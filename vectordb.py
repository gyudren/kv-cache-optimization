"""Embed data/processed/chunks.jsonl and load the chunks into a persistent ChromaDB collection.

Usage:
    python -m vectordb            # chunks.jsonl로 컬렉션을 안전하게 재구축

Environment variables:
    EMBEDDING_DEVICE=mps          # 기본값: mps
    EMBEDDING_BATCH_SIZE=4        # 기본값: 4
"""
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch


logger = logging.getLogger(__name__)

# 이 파일(vectordb.py)이 있는 폴더 경로 (= 프로젝트 루트, kv-cache-optimization/)
BASE_DIR = Path(__file__).resolve().parent
# 전처리 단계(preprocessing/pipeline.py)가 만들어 둔 청크 목록 파일
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
# ChromaDB가 인덱스(임베딩 벡터 + 원문 + 메타데이터)를 실제로 저장할 폴더
# PersistentClient를 쓰면 이 폴더에 sqlite/parquet 파일 형태로 남아서, 다음에 또 실행해도 재사용 가능
PERSIST_DIR = BASE_DIR / "data" / "chroma"
# ChromaDB 안에서 이 인덱스를 구분하는 이름 (여러 컬렉션을 만들 수도 있음)
COLLECTION_NAME = "kv_cache_chunks"
# 사용할 임베딩 모델. README의 Tech Stack에 적힌 모델(다국어·교차언어 검색 지원, 최대 32K 토큰)
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# MPS 통합 메모리 사용량을 낮추기 위해 기본 배치를 작게 유지한다.
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "4"))
STAGING_SUFFIX = "__rebuild"
BACKUP_SUFFIX = "__backup"


def format_duration(seconds: float) -> str:
    """초 단위 시간을 로그용 문자열로 변환한다."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def get_embedding_device() -> str:
    """요청한 임베딩 장치를 검증한다. 기본값은 Apple GPU인 MPS다."""
    device = os.getenv("EMBEDDING_DEVICE", "mps").lower()

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS를 사용할 수 없습니다. PyTorch/macOS 환경을 확인하거나 "
            "EMBEDDING_DEVICE=cpu로 명시해 실행하세요."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")
    if device not in {"mps", "cuda", "cpu"}:
        raise ValueError("EMBEDDING_DEVICE는 mps, cuda, cpu 중 하나여야 합니다.")

    return device


def get_collection_names(client: Any) -> set[str]:
    """설치된 Chroma 버전과 무관하게 컬렉션 이름 집합을 반환한다."""
    return {
        item.name if hasattr(item, "name") else str(item)
        for item in client.list_collections()
    }


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """ChromaDB가 "텍스트 → 벡터" 변환에 사용할 함수(클래스) 정의.

    ChromaDB는 기본 임베딩 함수를 내장하고 있지만, 우리는 README에서 지정한
    Qwen3-Embedding-0.6B를 쓰고 싶으므로 EmbeddingFunction을 상속받아 직접 구현한다.
    이렇게 만들어두면 collection.add()/upsert()/query() 할 때 ChromaDB가
    내부적으로 이 클래스를 호출해서 자동으로 텍스트를 벡터로 바꿔준다.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = get_embedding_device()
        local_files_only = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "0") == "1"
        logger.info("임베딩 모델 로딩 시작 | model=%s | device=%s", model_name, self.device)
        # sentence-transformers 라이브러리로 HuggingFace의 Qwen3 임베딩 모델을 불러옴
        # (처음 실행할 때는 모델 파일을 인터넷에서 다운로드하므로 시간이 좀 걸릴 수 있음)
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            local_files_only=local_files_only,
        )
        logger.info("임베딩 모델 로딩 완료 | device=%s", self.device)

    def __call__(self, input: Documents) -> Embeddings:
        """문서(청크 원문) 목록을 벡터 목록으로 변환. collection.add/upsert 시 자동 호출됨."""
        embeddings = self.model.encode(
            list(input),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,  # 벡터 길이를 1로 정규화 → 코사인 유사도 비교가 안정적
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, input: Documents) -> Embeddings:
        """검색어(질문)를 벡터로 변환할 때 호출됨. collection.query() 시 사용."""
        # Qwen3-Embedding은 "검색용 질문"에는 별도의 instruction(prompt_name="query")을
        # 붙여서 임베딩하도록 학습되어 있음 (문서 쪽엔 붙이지 않음). 그래야 검색 성능이 더 좋음.
        embeddings = self.model.encode(
            list(input),
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def default_space(self) -> str:
        # 벡터 간 거리를 잴 때 사용할 방식. 코사인 유사도 기준으로 검색하겠다는 뜻.
        return "cosine"

    @staticmethod
    def name() -> str:
        # ChromaDB가 이 임베딩 함수를 식별하기 위한 고유 이름표(필수 구현 항목)
        return "qwen3-embedding-0.6b"

    def get_config(self) -> Dict[str, Any]:
        # 이 임베딩 함수의 설정값을 저장. PersistentClient가 디스크에 기록해뒀다가
        # 나중에 다시 열 때 build_from_config()로 똑같은 임베딩 함수를 복원하는 데 쓰임.
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "Qwen3EmbeddingFunction":
        # get_config()로 저장해둔 설정값으로부터 이 클래스를 다시 만들어주는 복원 함수
        return Qwen3EmbeddingFunction(model_name=config["model_name"])


def load_chunks(path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """chunks.jsonl 파일을 읽어서 파이썬 딕셔너리 리스트로 변환.

    jsonl(JSON Lines)은 한 줄에 JSON 객체 하나씩 들어있는 포맷이라,
    한 줄씩 읽어서 json.loads()로 파싱하면 된다.
    """
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 빈 줄은 건너뜀
                chunks.append(json.loads(line))
    return chunks


def build_index(
    chunks_path: Path = CHUNKS_PATH,
    persist_dir: Path = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """새 컬렉션을 완성한 뒤 기존 컬렉션과 교체한다."""
    if BATCH_SIZE < 1:
        raise ValueError("EMBEDDING_BATCH_SIZE는 1 이상이어야 합니다.")

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")

    # 저장 폴더가 없으면 생성
    persist_dir.mkdir(parents=True, exist_ok=True)
    # 디스크에 영속적으로 저장되는 ChromaDB 클라이언트 생성 (메모리 전용이 아님)
    client = chromadb.PersistentClient(path=str(persist_dir))
    embedding_function = Qwen3EmbeddingFunction()
    staging_name = f"{collection_name}{STAGING_SUFFIX}"
    backup_name = f"{collection_name}{BACKUP_SUFFIX}"

    # 실패했던 이전 재구축의 임시 컬렉션만 제거한다. 운영 컬렉션은 새 인덱스 완성 전까지 유지한다.
    collection_names = get_collection_names(client)
    if staging_name in collection_names:
        logger.warning("미완료 임시 컬렉션 제거 | collection=%s", staging_name)
        client.delete_collection(name=staging_name)

    collection = client.create_collection(
        name=staging_name,
        embedding_function=embedding_function,
    )

    # ChromaDB에 넣을 세 가지 데이터를 각각 같은 순서의 리스트로 준비
    ids = [c["chunk_id"] for c in chunks]           # 각 청크의 고유 ID (예: "deepseek_v2_mla_text_0000")
    documents = [c["text"] for c in chunks]          # 실제로 임베딩할 원문 텍스트
    metadatas = [                                     # 검색 결과 필터링/표시에 쓸 부가 정보
        {
            "doc_id": c["doc_id"],           # 어떤 논문인지 (예: deepseek_v2_mla)
            "title": c["title"],             # 논문 제목
            "camp": c["camp"],               # SW / HW 진영 구분
            "role": c["role"],               # primary(선정 기술) / baseline(비교 대상)
            "content_type": c["content_type"],  # text / table
            "start_page": c["start_page"],
            "end_page": c["end_page"],
            "citation": c["citation"],       # 보고서 인용용 "[n, p.X]" 형식 문자열
            "char_count": c["char_count"],
        }
        for c in chunks
    ]

    total_chunks = len(chunks)
    total_batches = math.ceil(total_chunks / BATCH_SIZE)
    build_started_at = time.perf_counter()
    logger.info(
        "인덱스 재구축 시작 | chunks=%d | batches=%d | batch_size=%d | staging=%s",
        total_chunks,
        total_batches,
        BATCH_SIZE,
        staging_name,
    )

    # 각 배치를 임베딩한 직후 임시 컬렉션에 저장한다.
    for batch_number, start in enumerate(
        range(0, total_chunks, BATCH_SIZE),
        start=1,
    ):
        end = min(start + BATCH_SIZE, total_chunks)
        batch_started_at = time.perf_counter()
        batch_characters = sum(len(text) for text in documents[start:end])
        logger.info(
            "[배치 %d/%d] 시작 | chunks=%d-%d/%d | chars=%d",
            batch_number,
            total_batches,
            start + 1,
            end,
            total_chunks,
            batch_characters,
        )

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        if embedding_function.device == "mps":
            torch.mps.empty_cache()

        batch_elapsed = time.perf_counter() - batch_started_at
        total_elapsed = time.perf_counter() - build_started_at
        estimated_remaining = (
            total_elapsed / batch_number * (total_batches - batch_number)
        )
        logger.info(
            "[배치 %d/%d] 완료 | indexed=%d/%d (%.1f%%) | batch=%s | "
            "elapsed=%s | ETA=%s",
            batch_number,
            total_batches,
            end,
            total_chunks,
            end / total_chunks * 100,
            format_duration(batch_elapsed),
            format_duration(total_elapsed),
            format_duration(estimated_remaining),
        )

    indexed_count = collection.count()
    if indexed_count != total_chunks:
        raise RuntimeError(
            f"임시 컬렉션 검증 실패: expected={total_chunks}, actual={indexed_count}"
        )

    # 완성된 임시 컬렉션을 운영 이름으로 교체한다. 교체 전까지 기존 인덱스는 보존된다.
    collection_names = get_collection_names(client)
    if backup_name in collection_names:
        client.delete_collection(name=backup_name)

    previous_collection = None
    if collection_name in collection_names:
        previous_collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        previous_collection.modify(name=backup_name)

    try:
        collection.modify(name=collection_name)
    except Exception:
        if previous_collection is not None:
            previous_collection.modify(name=collection_name)
        raise

    if previous_collection is not None:
        client.delete_collection(name=backup_name)

    total_elapsed = time.perf_counter() - build_started_at
    logger.info(
        "인덱스 재구축 완료 | collection=%s | chunks=%d | elapsed=%s | path=%s",
        collection_name,
        indexed_count,
        format_duration(total_elapsed),
        persist_dir,
    )
    return collection


# 이 파일을 "python vectordb.py" 또는 "python -m vectordb"로 직접 실행했을 때만 build_index() 호출
# (다른 파일에서 "import vectordb"로 불러올 때는 자동 실행되지 않음)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    build_index()
