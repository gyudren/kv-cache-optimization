"""Embed data/processed/chunks.jsonl and load the chunks into a persistent ChromaDB collection.

Usage:
    python -m vectordb            # (re)build the collection from chunks.jsonl
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch


def log(msg: str) -> None:
    """타임스탬프를 붙여서 즉시 출력하는 로그 함수.

    일반 print()는 터미널이 아닌 곳(백그라운드 실행 등)으로 출력이 리다이렉트되면
    버퍼링 때문에 한참 있다가 한꺼번에 찍힐 수 있어서, flush=True로 강제로 즉시 내보낸다.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
# 한 번에 모델에 넣어서 임베딩할 청크 개수. 표 청크처럼 아주 긴 것도 섞여 있어서
# 배치를 너무 크게 잡으면 "진행 로그가 한참 안 찍히는" 것처럼 보이므로 작게(8개) 잡는다.
BATCH_SIZE = 8


class Qwen3EmbeddingFunction(EmbeddingFunction):
    """ChromaDB가 "텍스트 → 벡터" 변환에 사용할 함수(클래스) 정의.

    ChromaDB는 기본 임베딩 함수를 내장하고 있지만, 우리는 README에서 지정한
    Qwen3-Embedding-0.6B를 쓰고 싶으므로 EmbeddingFunction을 상속받아 직접 구현한다.
    이렇게 만들어두면 collection.add()/upsert()/query() 할 때 ChromaDB가
    내부적으로 이 클래스를 호출해서 자동으로 텍스트를 벡터로 바꿔준다.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        # NVIDIA GPU(cuda) > 맥 GPU(mps) > CPU 순으로 가능한 가속기를 사용.
        # 예전에는 표 청크가 최대 2만자까지 길어서 mps로 돌리면 긴 시퀀스 어텐션이
        # 메모리를 과하게 잡아먹다 "MPS backend out of memory"로 터졌는데, 이제 청킹이
        # 1,200자(겹침 200자) 기준으로 강제 분할돼서 가장 긴 청크도 ~1,400자 수준이라
        # mps로도 안전하게 돌아간다.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        log(f"임베딩 모델 로딩 시작: {model_name} (device={device})")
        # sentence-transformers 라이브러리로 HuggingFace의 Qwen3 임베딩 모델을 불러옴
        # (처음 실행할 때는 모델 파일을 인터넷에서 다운로드하므로 시간이 좀 걸릴 수 있음)
        self.model = SentenceTransformer(model_name, device=device)
        log("임베딩 모델 로딩 완료")

    def __call__(self, input: Documents) -> Embeddings:
        """문서(청크 원문) 목록을 벡터 목록으로 변환. collection.add/upsert 시 자동 호출됨."""
        texts = list(input)
        longest = max(len(t) for t in texts)
        log(f"  청크 {len(texts)}개 임베딩 중... (가장 긴 청크: {longest:,}자)")
        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,  # 벡터 길이를 1로 정규화 → 코사인 유사도 비교가 안정적
            show_progress_bar=True,  # 배치 내부 진행 상황을 tqdm 진행바로 표시
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


# 영문 알파벳 3자 이상이 붙어있는 "진짜 단어"를 찾는 패턴.
# 수식이 깨져서 나온 줄에는 이런 진짜 단어가 거의 없다는 점을 노이즈 판별 기준으로 삼는다.
_REAL_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _is_formula_noise_line(line: str) -> bool:
    """pdfplumber가 수식을 깨서 뽑아낸 줄인지 판단.

    PDF 안의 수식은 실제로 렌더링되지 않고 글자 글리프만 좌표 순서대로 뽑혀 나오기
    때문에, "𝐡𝐡𝑡𝑡", "𝐿𝐿", "4 …", "𝑁𝑁𝑟𝑟" 처럼 진짜 단어 없이 수식 기호·아래/위첨자
    문자·외톨이 숫자만 있는 줄이 생긴다. 이런 줄은 문장이 아니라서 임베딩에 넣으면
    의미 없는 토큰만 늘어 검색 관련성을 떨어뜨리므로, "3자 이상 영단어가 하나도 없는 줄"을
    노이즈로 보고 제거한다. 실제 문장에는 거의 항상 이런 단어가 있어서 오탐 위험은 낮다.
    """
    stripped = line.strip()
    if not stripped:
        return False
    return not _REAL_WORD_RE.search(stripped)


def clean_chunk_text(text: str) -> str:
    """임베딩 직전에 청크 텍스트에서 깨진 수식 노이즈 줄을 제거한다.

    표(content_type="table") 청크는 대상이 아니다 — 표는 원래 숫자·짧은 라벨 위주라
    이 기준을 그대로 적용하면 정상적인 표 데이터까지 지워질 수 있기 때문.
    ChromaDB에는 이렇게 정제한 텍스트만 저장·임베딩하고, data/processed/chunks.jsonl
    원본 파일 자체는 건드리지 않는다(원문 확인·재처리용으로 그대로 둠).
    """
    lines = text.split("\n")
    kept_lines = [line for line in lines if not _is_formula_noise_line(line)]
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # 줄을 지우고 남은 빈 줄 뭉치 정리
    return cleaned.strip()


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
    """chunks.jsonl → 임베딩 → ChromaDB 컬렉션 적재까지 한 번에 실행하는 메인 함수."""
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found at {chunks_path}")
    log(f"chunks.jsonl에서 청크 {len(chunks)}개 로드 완료")

    # 임베딩 전에 텍스트 청크에서 깨진 수식 노이즈 줄을 제거한다.
    # 제거 후 남는 내용이 거의 없는 청크(예: 수식으로만 가득 찬 청크)는 검색에 도움이
    # 안 되므로 아예 색인에서 뺀다.
    MIN_CLEANED_LENGTH = 20
    cleaned_chunks = []
    dropped = 0
    for c in chunks:
        if c["content_type"] == "text":
            cleaned_text = clean_chunk_text(c["text"])
        else:  # table 청크는 정제 대상이 아님
            cleaned_text = c["text"]

        if len(cleaned_text) < MIN_CLEANED_LENGTH:
            dropped += 1
            continue

        c = {**c, "text": cleaned_text, "char_count": len(cleaned_text)}
        cleaned_chunks.append(c)

    if dropped:
        log(f"수식 노이즈 제거 후 내용이 거의 안 남은 청크 {dropped}개는 색인에서 제외")
    chunks = cleaned_chunks

    # 글자 수가 짧은 청크부터 처리하도록 정렬.
    # 그렇지 않으면 짧은 청크와 긴 청크(표 등, 최대 2만자)가 한 배치에 섞여서
    # 짧은 것도 긴 것 길이에 맞춰 패딩되어 배치 전체가 느려지고, 진행 로그도 한참 안 보이게 됨.
    # 결과(색인 내용)에는 영향 없고, 단지 처리 순서만 "짧은 것 → 긴 것"으로 바뀜.
    chunks = sorted(chunks, key=lambda c: c["char_count"])

    # 저장 폴더가 없으면 생성
    persist_dir.mkdir(parents=True, exist_ok=True)
    # 디스크에 영속적으로 저장되는 ChromaDB 클라이언트 생성 (메모리 전용이 아님)
    client = chromadb.PersistentClient(path=str(persist_dir))
    # 컬렉션이 이미 있으면 가져오고, 없으면 새로 만듦.
    # embedding_function을 넘겨주면 이후 add/upsert/query 할 때 텍스트를 자동으로 벡터화해줌.
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=Qwen3EmbeddingFunction(),
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

    # 청크가 많을 수 있으니 BATCH_SIZE개씩 나눠서 upsert
    # upsert = "id가 이미 있으면 덮어쓰고, 없으면 새로 추가" (같은 스크립트를 다시 돌려도 안전함)
    num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx, start in enumerate(range(0, len(chunks), BATCH_SIZE), start=1):
        end = start + BATCH_SIZE
        batch_start_time = time.time()
        log(f"배치 {batch_idx}/{num_batches} 시작 (청크 {ids[start:end][0]} ~ {ids[start:end][-1]})")
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        elapsed = time.time() - batch_start_time
        log(
            f"배치 {batch_idx}/{num_batches} 완료 "
            f"({min(end, len(chunks))}/{len(chunks)}개 누적, {elapsed:.1f}초 소요)"
        )

    log(f"done: {collection.count()} chunks in collection '{collection_name}' at {persist_dir}")
    return collection


# 이 파일을 "python vectordb.py" 또는 "python -m vectordb"로 직접 실행했을 때만 build_index() 호출
# (다른 파일에서 "import vectordb"로 불러올 때는 자동 실행되지 않음)
if __name__ == "__main__":
    build_index()
