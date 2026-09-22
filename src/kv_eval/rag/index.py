"""Fixed local Qwen3 embeddings, FAISS dense index and BM25 lexical index.

청크는 raw PDF에서 다시 만들지 않고 데이터 전처리 담당(preprocessing/)의 산출물
(data/processed/chunks.jsonl)을 그대로 쓴다 — ingest.load_prepared_chunks() 참고.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any
from ..config import EMBEDDING_ID
from .ingest import load_prepared_chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), re.UNICODE)

@dataclass
class RetrievalStore:
    chunks: list[dict]
    model: Any
    dense_index: Any
    bm25: Any


def build_index(chunks: list[dict] | None = None) -> RetrievalStore:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
    import numpy as np
    chunks = chunks if chunks is not None else load_prepared_chunks()
    if not chunks:
        raise ValueError("Cannot build index: no usable text chunks")
    model = SentenceTransformer(EMBEDDING_ID, trust_remote_code=True)
    vectors = np.asarray(model.encode([p["text"] for p in chunks], show_progress_bar=False), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(c["text"]) or ["<EMPTY>"] for c in chunks])
    return RetrievalStore(chunks, model, index, bm25)
