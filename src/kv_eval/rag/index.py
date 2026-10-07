"""로컬 Qwen3 임베딩 기반 FAISS Dense 색인 + BM25 Sparse 색인.

문서 임베딩은 `data/cache/index/{지문}.npy`에 캐시한다. 지문은 모델 ID와 청크로 만들어
둘 중 하나가 바뀌면 다시 임베딩한다.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
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
    vectors: Any  # 정규화된 문서 임베딩. 기술 필터 부분집합 점수 계산에 쓴다
    embeddings_cached: bool = False


def index_cache_dir() -> Path:
    return Path(os.getenv("RAG_CACHE_DIR", "data/cache")) / "index"


def corpus_fingerprint(chunks: list[dict]) -> str:
    digest = sha256(EMBEDDING_ID.encode())
    for chunk in chunks:
        digest.update(b"\0" + str(chunk.get("chunk_id", "")).encode() + b"\0" + chunk["text"].encode())
    return digest.hexdigest()[:16]


def load_or_embed(chunks: list[dict], model: Any, cache_dir: Path | None = None) -> tuple[Any, bool]:
    """정규화된 문서 임베딩과 캐시 재사용 여부를 돌려준다."""
    import faiss
    import numpy as np

    path = (cache_dir or index_cache_dir()) / f"{corpus_fingerprint(chunks)}.npy"
    if path.is_file():
        try:
            vectors = np.load(path)
            if vectors.ndim == 2 and vectors.shape[0] == len(chunks):
                return np.ascontiguousarray(vectors, dtype="float32"), True
        except (OSError, ValueError):
            pass  # 깨진 캐시는 다시 임베딩해 덮어쓴다
    vectors = np.asarray(
        model.encode([chunk["text"] for chunk in chunks], batch_size=EMBED_BATCH_SIZE, show_progress_bar=False),
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 저장 중 중단돼도 반쯤 쓴 파일을 읽지 않도록 임시 파일에 쓴 뒤 교체한다.
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, vectors)
    os.replace(tmp, path)
    return vectors, False


def build_index(chunks: list[dict]) -> RetrievalStore:
    """전처리된 청크 목록으로 Dense(FAISS)·Sparse(BM25) 색인을 만든다."""
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss

    if not chunks:
        raise ValueError("색인할 청크가 없습니다")
    # 질의 임베딩에 필요해서 캐시가 있어도 모델은 올린다.
    model = SentenceTransformer(EMBEDDING_ID, trust_remote_code=True)
    vectors, cached = load_or_embed(chunks, model)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(chunk["text"]) or ["<EMPTY>"] for chunk in chunks])
    return RetrievalStore(chunks, model, index, bm25, vectors, embeddings_cached=cached)
