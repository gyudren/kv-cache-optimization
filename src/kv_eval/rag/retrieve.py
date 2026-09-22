"""Dense(FAISS) + BM25 하이브리드 검색을 RRF로 융합하고 기술별 문서 필터를 적용한다.

설계 B-3): "Dense(FAISS) + BM25(키워드) → RRF 순위 융합, 기술별 문서 필터로 다른 기술
수치 혼입 방지". 필터를 먼저 적용해 허용된 문서만 순위 경쟁에 참여시키므로, 예를 들어
MLA 질의의 상위 k에 ITME 수치가 섞여 들어가지 않는다.
"""
from __future__ import annotations
from dataclasses import dataclass
from ..config import RRF_CONSTANT, RETRIEVAL_K
from .index import RetrievalStore, tokenize


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    page: int
    text: str
    technology: str
    score: float
    citation_number: int
    chunk_id: str


def permitted_technologies(technology_filter: str) -> set[str]:
    """질의 대상 기술에 따라 검색을 허용할 문서 그룹을 정한다.

    itme_baseline은 ITME 원문과 HW 베이스라인 2편(InfiniGen, CXL-PNM)을 함께 검색해
    선정 기술의 한계를 제3의 시각에서 교차 확인하기 위한 필터다(설계 B-3 ②).
    """
    if technology_filter == "mla":
        return {"mla"}
    if technology_filter == "itme":
        return {"itme"}
    if technology_filter == "itme_baseline":
        return {"itme", "baseline"}
    if technology_filter == "all":
        # 문서 4편 전체 검색. 운영 Agent는 쓰지 않고, 검색 순위 품질 측정(eval)에만 사용한다.
        return {"mla", "itme", "baseline"}
    raise ValueError(f"Unsupported research filter: {technology_filter!r}")


def rrf_fuse(dense_rank: list[str], lexical_rank: list[str], constant: int = RRF_CONSTANT) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in (dense_rank, lexical_rank):
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    def __init__(self, store: RetrievalStore):
        self.store = store

    def hybrid_search(self, query: str, technology_filter: str, k: int = RETRIEVAL_K) -> list[RetrievedChunk]:
        import faiss
        import numpy as np

        if k < 1:
            raise ValueError("k must be >=1")
        permitted = permitted_technologies(technology_filter)
        subset = [i for i, chunk in enumerate(self.store.chunks) if chunk["technology"] in permitted]
        if not subset:
            return []

        # Qwen3-Embedding은 검색 질의에 instruction을 붙여 임베딩하도록 학습돼 있다
        # (설계 B-4가 이 모델을 고른 근거). 문서는 instruction 없이 임베딩한다.
        query_embedding = np.asarray(
            self.store.model.encode([query], prompt_name="query"), dtype="float32"
        )
        faiss.normalize_L2(query_embedding)
        # 허용된 문서만 점수를 매긴다(제외된 기술이 top-k 자리를 차지하지 못하게).
        dense_scores = self.store.vectors[subset] @ query_embedding[0]
        lexical_scores = self.store.bm25.get_scores(tokenize(query))

        local_top = min(max(k, 1), len(subset))
        # 점수 동률이면 전역 인덱스 순으로 정렬해 실행마다 결과가 흔들리지 않게 한다.
        dense_order = sorted(range(len(subset)), key=lambda j: (-float(dense_scores[j]), subset[j]))[:local_top]
        lexical_order = sorted(subset, key=lambda i: (-float(lexical_scores[i]), i))[:local_top]

        fused = rrf_fuse(
            [self.store.chunks[subset[j]]["chunk_id"] for j in dense_order],
            [self.store.chunks[i]["chunk_id"] for i in lexical_order],
        )
        by_id = {self.store.chunks[i]["chunk_id"]: self.store.chunks[i] for i in subset}
        return [
            RetrievedChunk(
                **{field: chunk[field] for field in
                   ("doc_id", "page", "text", "technology", "citation_number", "chunk_id")},
                score=score,
            )
            for chunk_id, score in fused[:k]
            if (chunk := by_id.get(chunk_id))
        ]
