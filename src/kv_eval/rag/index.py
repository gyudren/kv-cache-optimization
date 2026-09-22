"""Fixed local Qwen3 embeddings, FAISS dense index and BM25 lexical index."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any
from ..config import EMBEDDING_ID
from .ingest import chunk_pages


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), re.UNICODE)

@dataclass
class RetrievalStore:
    chunks: list[dict]
    model: Any
    dense_index: Any
    bm25: Any


def build_index(pages: list[dict]) -> RetrievalStore:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
    import numpy as np
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("Cannot build index: no usable text chunks")
    model = SentenceTransformer(EMBEDDING_ID, trust_remote_code=True)
    vectors = np.asarray(model.encode([p["text"] for p in chunks], show_progress_bar=False), dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(c["text"]) or ["<EMPTY>"] for c in chunks])
    return RetrievalStore(chunks, model, index, bm25)
