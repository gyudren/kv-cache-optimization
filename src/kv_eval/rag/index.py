"""로컬 Qwen3 임베딩 기반 FAISS Dense 색인 + BM25 Sparse 색인 (설계 B-3/B-4)."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any
from ..config import EMBEDDING_ID, EMBED_BATCH_SIZE


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), re.UNICODE)


@dataclass
class RetrievalStore:
    chunks: list[dict]
    model: Any
    dense_index: Any
    bm25: Any
    vectors: Any  # 정규화된 임베딩 행렬. 기술 필터 적용 시 부분집합 점수 계산에 재사용한다.


def build_index(chunks: list[dict]) -> RetrievalStore:
    """전처리된 청크 목록으로 Dense(FAISS)·Sparse(BM25) 색인을 만든다."""
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
    import numpy as np

    if not chunks:
        raise ValueError("색인할 청크가 없습니다")
    model = SentenceTransformer(EMBEDDING_ID, trust_remote_code=True)
    vectors = np.asarray(
        model.encode(
            [chunk["text"] for chunk in chunks],
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        ),
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(chunk["text"]) or ["<EMPTY>"] for chunk in chunks])
    return RetrievalStore(chunks, model, index, bm25, vectors)
