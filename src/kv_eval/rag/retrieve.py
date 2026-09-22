"""Evidence-preserving retrieval: fixed RRF, prefilter per source paper."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
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


def permitted_docs(technology_filter: str) -> set[str]:
    if technology_filter == "mla":
        return {"deepseek_v2"}
    if technology_filter == "itme":
        return {"itme"}
    if technology_filter == "itme_baseline":
        return {"itme", "infinigen", "cxl_pnm"}
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
        permitted = permitted_docs(technology_filter)
        subset = [i for i, c in enumerate(self.store.chunks) if c["doc_id"] in permitted]
        if not subset:
            return []
        query_embedding = np.asarray(self.store.model.encode([query]), dtype="float32")
        faiss.normalize_L2(query_embedding)
        # Score only the permitted corpus; no discarded technology consumes top-k budget.
        vectors = np.vstack([self.store.dense_index.reconstruct(i) for i in subset]).astype("float32")
        dense_scores = vectors @ query_embedding[0]
        lexical_scores = self.store.bm25.get_scores(tokenize(query))
        local_top = min(max(k, 1), len(subset))
        dense_order = sorted(subset, key=lambda i: (-float(dense_scores[subset.index(i)]), i))[:local_top]
        lexical_order = sorted(subset, key=lambda i: (-float(lexical_scores[i]), i))[:local_top]
        fused = rrf_fuse([self.store.chunks[i]["chunk_id"] for i in dense_order],
                         [self.store.chunks[i]["chunk_id"] for i in lexical_order])
        indices = {c["chunk_id"]: c for c in self.store.chunks if c["doc_id"] in permitted}
        return [RetrievedChunk(**{field: chunk[field] for field in
                                ("doc_id", "page", "text", "technology", "citation_number", "chunk_id")}, score=score)
                for cid, score in fused[:k] if (chunk := indices.get(cid))]
