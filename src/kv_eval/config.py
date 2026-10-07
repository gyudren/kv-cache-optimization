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
# 색인 직전에 "3자 이상 영단어가 없는 줄"(깨진 수식 글리프)을 제거할지 여부.
# 동일 평가 세트(92케이스) A/B 측정 결과 끄는 쪽이 합격 기준을 통과해 기본값을 0으로 둔다.
#   ON  : Hit@1 0.86 / MRR 0.93 / 필수 용어 커버리지 0.75 → FAIL
#   OFF : Hit@1 0.84 / MRR 0.92 / 필수 용어 커버리지 0.92 → PASS
# 순위는 거의 같은데 표·수식 줄의 근거 용어가 함께 지워지는 손해가 더 크다.
CLEAN_FORMULA_NOISE = os.getenv("CLEAN_FORMULA_NOISE", "0") == "1"
RETRIEVAL_K = 6  # Implementation detail; same setting across technologies.
RRF_CONSTANT = 60
MIN_RELEVANT = 2
RAG_REWRITES = 2
# 질문 단위 RAG 호출 동시 실행 수. 질문끼리 독립이라 결과는 같고 대기 시간만 줄어든다.
MAX_PARALLEL_QUESTIONS = int(os.getenv("MAX_PARALLEL_QUESTIONS", "12"))
# 4관점(기술 성숙도·시장성·이해관계자·도메인). Supervisor가 State를 보고 이 중 필요한 것만 고른다.
PERSPECTIVES = ("tech", "market", "stakeholder", "domain")
# 한 superstep에서 동시에 실행할 작업 노드 수(LangGraph max_concurrency). Supervisor는 부족한 관점을
# Send로 한 번에 할당하지만, 실행은 기본 1개씩 순차로 한다. 관점마다 근거 원문·프롬프트·임베딩 호출이 함께
# 메모리에 올라오므로 동시에 돌리면 최대 메모리가 관점 수만큼 커지고(OOM), MPS 임베딩 경합도 생긴다.
AGENT_CONCURRENCY = max(1, int(os.getenv("AGENT_CONCURRENCY", "1")))
# 재작업 상한(안전장치). 근거 충분성·품질 평가 통과가 1차 종료 조건이고, 상한은 무한 루프만 막는다.
RETRY_LIMITS = {"tech": 2, "market": 2, "stakeholder": 2, "domain": 2, "synthesis": 1, "report": 2,
                "quality_evaluator": 1}
# Supervisor 진입 횟수 상한. 넘으면 조사·재작성을 멈추고 종합→보고서→평가만 마친 뒤 정상 종료한다.
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))
# 상한 도달 후 마무리에 필요한 Supervisor 진입 수: 종합 → 보고서 → 평가 → 종료 = 4, 여유 1을 더해 5.
# (마무리 중 실패한 노드는 재시도하지 않고 공백으로 넘기므로 4회를 넘지 않는다. 여유분은 재개 직후의 재진입용)
FINALIZE_STEPS = 5


def recursion_limit_for(max_steps: int) -> int:
    """LangGraph recursion_limit은 Supervisor 단계 상한에서만 계산한다(단일 출처).

    Supervisor 1회 = supervisor + 작업 노드 = 2 superstep. 마무리 단계와 여유 10을 더한다.
    build_graph가 실제로 쓰는 Policy.max_steps로 이 값을 계산해 그래프 기본 설정에 넣는다.
    """
    return (max_steps + FINALIZE_STEPS) * 2 + 10
# State에 남기는 Evidence 발췌 길이. 원문은 evidence_store(디스크)에 두고 excerpt_ref로 참조한다.
STATE_EXCERPT_CHARS = 160
# 편향 통제 규칙: 기술·관점별 최소 고유 출처 수, 웹 근거의 단일 발행처 비중 상한
MIN_DISTINCT_SOURCES = 2
MAX_SINGLE_SOURCE_SHARE = 0.6
MAX_REPORT_PAGES = 10
# 목표 페이지(여유 1p). 넘으면 경고만 남긴다(상한 초과는 이슈).
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
        # 환경변수가 .env보다 우선한다(12-factor, CI 주입 허용). 다만 셸에 남아 있는 옛 키가
        # .env를 가리면 "키를 바꿨는데 401이 난다"는 오진이 나오므로, 값이 서로 다르면 알린다.
        load_dotenv()
        for name, file_value in dotenv_values().items():
            env_value = os.getenv(name)
            if file_value and env_value and env_value.strip() != file_value.strip():
                print(f"[config] 경고: 환경변수 {name} 이(가) .env 값을 덮어씁니다. "
                      f"환경변수 값이 사용됩니다(unset 하면 .env 값 사용).")
        # 키 끝에 붙은 개행·공백은 그대로 쓰면 HTTP 헤더가 불법값이 되어
        # 인증 오류가 아니라 "Connection error"로 보이므로 여기서 제거한다.
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
