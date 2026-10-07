"""Dense(FAISS) + BM25 하이브리드 검색을 RRF로 융합한다.

기술별 문서 필터를 먼저 적용해 다른 기술의 청크가 상위 k에 섞이지 않게 한다.
"""
from __future__ import annotations
from dataclasses import dataclass
import threading
from ..config import RRF_CONSTANT, RETRIEVAL_K
from .index import RetrievalStore, tokenize

# 병렬 노드와 RAG 질문이 같은 임베딩 모델을 동시에 부른다. macOS MPS는 스레드 간 동시 encode에서
# 프로세스가 abort되므로 질의 임베딩을 직렬화한다.
_ENCODE_LOCK = threading.Lock()


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

    itme_baseline은 ITME 원문에 HW 베이스라인(InfiniGen, CXL-PNM)을 더해 한계를 교차 확인할 때 쓴다.
    """
    if technology_filter == "mla":
        return {"mla"}
    if technology_filter == "itme":
        return {"itme"}
    if technology_filter == "itme_baseline":
        return {"itme", "baseline"}
    if technology_filter == "all":
        # 검색 품질 측정(eval) 전용
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

        # Qwen3-Embedding은 질의에만 instruction을 붙이고 문서는 그대로 임베딩한다.
        with _ENCODE_LOCK:
            query_embedding = np.asarray(
                self.store.model.encode([query], prompt_name="query"), dtype="float32"
            )
        faiss.normalize_L2(query_embedding)
        dense_scores = self.store.vectors[subset] @ query_embedding[0]
        lexical_scores = self.store.bm25.get_scores(tokenize(query))

        local_top = min(max(k, 1), len(subset))
        # 동률은 전역 인덱스 순으로 정렬해 결과를 고정한다.
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
