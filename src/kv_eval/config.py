"""Fixed project configuration. No auto-selection or model fallback."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

MODEL_ID = "gpt-5.6-terra"
DOMAIN = "데이터센터·클라우드 기반 장문맥 LLM 서빙"
TECHNOLOGIES = {"mla": "DeepSeek-V2 MLA", "itme": "ITME"}
EMBEDDING_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_BATCH_SIZE = 8  # 청크 길이 편차가 커서 배치를 크게 잡으면 패딩 때문에 느려진다
MAX_PAGES = 200
DECLARED_PAGES = {"deepseek_v2": 52, "itme": 13, "infinigen": 18, "cxl_pnm": 13}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_INDEXED_CHARS = 20  # 정제 후 이보다 짧게 남는 청크는 색인하지 않는다
# 깨진 수식 줄 제거 여부. 켜면 표·수식 줄의 근거 용어까지 지워져 필수 용어 커버리지가 떨어지므로 기본은 끈다.
CLEAN_FORMULA_NOISE = os.getenv("CLEAN_FORMULA_NOISE", "0") == "1"
RETRIEVAL_K = 6
RRF_CONSTANT = 60
MIN_RELEVANT = 2
RAG_REWRITES = 2
# 질문별 RAG 호출 동시 실행 수. 질문끼리 독립이라 결과에는 영향이 없다.
MAX_PARALLEL_QUESTIONS = int(os.getenv("MAX_PARALLEL_QUESTIONS", "12"))
PERSPECTIVES = ("tech", "market", "stakeholder", "domain")
# 함께 할당된 관점 노드를 동시에 몇 개 돌릴지(max_concurrency).
# 관점마다 근거 원문과 임베딩 호출이 메모리에 같이 올라오고 MPS 임베딩 경합도 생겨서 기본은 1이다.
AGENT_CONCURRENCY = max(1, int(os.getenv("AGENT_CONCURRENCY", "1")))
# 무한 루프를 막는 재작업 상한. 관점 값은 충분성 재조사에만 쓴다.
RETRY_LIMITS = {"tech": 2, "market": 2, "stakeholder": 2, "domain": 2, "synthesis": 1, "report": 2,
                "quality_evaluator": 1}
# 종합·평가가 관점을 다시 부를 때 쓰는 후속 재조사 한도. RETRY_LIMITS와 따로 센다.
FOLLOWUP_LIMITS = {name: 1 for name in ("tech", "market", "stakeholder", "domain")}
# Supervisor 진입 상한. 넘으면 추가 조사를 멈추고 보고서까지 마무리한 뒤 끝낸다.
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
# 상한 도달 후 마무리(종합, 보고서, 평가, 종료 4회)에 재개 직후 재진입 여유 1회를 더한 값.
FINALIZE_STEPS = 5


def recursion_limit_for(max_steps: int) -> int:
    """Supervisor 단계 상한으로 LangGraph recursion_limit을 계산한다.

    Supervisor 한 번이 supervisor와 작업 노드 2 superstep이라 2배 하고, 여유 10을 더한다.
    """
    return (max_steps + FINALIZE_STEPS) * 2 + 10
# State에 남기는 발췌 길이. 원문은 evidence_store에 둔다.
STATE_EXCERPT_CHARS = 160
# 편향 통제: 기술·관점별 최소 고유 출처 수, 단일 발행처 비중 상한
MIN_DISTINCT_SOURCES = 2
MAX_SINGLE_SOURCE_SHARE = 0.6
MAX_REPORT_PAGES = 10
# 넘으면 경고만 하고, MAX_REPORT_PAGES를 넘으면 이슈로 처리한다.
TARGET_REPORT_PAGES = 9
REPORT_STEM = "Agent_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리"
# 이전 과제(RAG) 산출물 파일명. 덮어쓰지 않도록 구분만 해 둔다.
LEGACY_REPORT_STEM = "RAG-Output_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리"
LANGSMITH_RUN_NAME = "kv-eval-supervisor"
LANGSMITH_TAGS = ["pattern:supervisor", "kv-cache-eval"]

@dataclass(frozen=True)
class Settings:
    manifest_path: Path
    chunks_path: Path
    summary_path: Path
    output_dir: Path
    openai_key: str
    tavily_key: str
    model: str = MODEL_ID

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import dotenv_values, load_dotenv
        # 환경변수가 .env보다 우선한다. 셸에 남은 옛 키가 .env를 가리는 경우를 알 수 있게 값이 다르면 경고한다.
        load_dotenv()
        for name, file_value in dotenv_values().items():
            env_value = os.getenv(name)
            if file_value and env_value and env_value.strip() != file_value.strip():
                print(f"[config] 경고: 환경변수 {name} 이(가) .env 값을 덮어씁니다. "
                      f"환경변수 값이 사용됩니다(unset 하면 .env 값 사용).")
        # 키 끝의 개행·공백이 HTTP 헤더를 깨뜨리면 "Connection error"로 보이므로 strip한다.
        model = os.getenv("OPENAI_MODEL", MODEL_ID).strip()
        if model != MODEL_ID:
            raise ValueError(f"Generator/Judge model must be {MODEL_ID}; got {model!r}")
        processed = Path(os.getenv("PROCESSED_DIR", "data/processed"))
        return cls(
            manifest_path=Path(os.getenv("MANIFEST_PATH", "data/manifest.json")),
            chunks_path=processed / "chunks.jsonl",
            summary_path=processed / "summary.json",
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            openai_key=os.getenv("OPENAI_API_KEY", "").strip(),
            tavily_key=os.getenv("TAVILY_API_KEY", "").strip(),
            model=model,
        )

    def require_credentials(self) -> None:
        if not self.openai_key or not self.tavily_key:
            raise RuntimeError("OPENAI_API_KEY and TAVILY_API_KEY are required; see .env.example")
