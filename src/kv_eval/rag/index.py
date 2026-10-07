"""로컬 Qwen3 임베딩 기반 FAISS Dense 색인 + BM25 Sparse 색인 (설계 B-3/B-4).

문서 임베딩은 최초 1회만 계산해 `data/cache/index/{지문}.npy`에 저장하고, 이후 실행은 그 파일을 읽는다.
지문은 임베딩 모델 ID와 청크(chunk_id·본문)로 만들므로, 전처리를 다시 해 청크가 바뀌거나 모델을 바꾸면
자동으로 새로 임베딩한다. FAISS Flat 색인과 BM25는 저장된 벡터·청크로 매번 다시 만든다(수백 개라 즉시 끝남).
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
    vectors: Any  # 정규화된 임베딩 행렬. 기술 필터 적용 시 부분집합 점수 계산에 재사용한다.
    embeddings_cached: bool = False  # True면 저장된 문서 임베딩을 재사용했다(새로 임베딩하지 않음)


def index_cache_dir() -> Path:
    return Path(os.getenv("RAG_CACHE_DIR", "data/cache")) / "index"


def corpus_fingerprint(chunks: list[dict]) -> str:
    digest = sha256(EMBEDDING_ID.encode())
    for chunk in chunks:
        digest.update(b"\0" + str(chunk.get("chunk_id", "")).encode() + b"\0" + chunk["text"].encode())
    return digest.hexdigest()[:16]


def load_or_embed(chunks: list[dict], model: Any, cache_dir: Path | None = None) -> tuple[Any, bool]:
    """정규화된 문서 임베딩 행렬과 캐시 재사용 여부. 같은 청크·모델이면 저장된 벡터를 읽는다."""
    import faiss
    import numpy as np

    path = (cache_dir or index_cache_dir()) / f"{corpus_fingerprint(chunks)}.npy"
    if path.is_file():
        try:
            vectors = np.load(path)
            if vectors.ndim == 2 and vectors.shape[0] == len(chunks):
                return np.ascontiguousarray(vectors, dtype="float32"), True
        except (OSError, ValueError):
            pass  # 깨진 캐시는 무시하고 다시 임베딩해 덮어쓴다
    vectors = np.asarray(
        model.encode([chunk["text"] for chunk in chunks], batch_size=EMBED_BATCH_SIZE, show_progress_bar=False),
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 임시 파일에 쓴 뒤 교체한다. 저장 중 프로세스가 죽어도 다음 실행이 반쯤 쓴 파일을 읽지 않는다.
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
    # 질의 임베딩에 모델이 필요하므로 모델은 항상 올린다. 문서 임베딩만 캐시에서 읽는다.
    model = SentenceTransformer(EMBEDDING_ID, trust_remote_code=True)
    vectors, cached = load_or_embed(chunks, model)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(chunk["text"]) or ["<EMPTY>"] for chunk in chunks])
    return RetrievalStore(chunks, model, index, bm25, vectors, embeddings_cached=cached)
