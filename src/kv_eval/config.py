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
RETRY_LIMITS = {"tech": 2, "market": 2, "stakeholder": 2, "domain": 2, "synthesis": 1, "report": 2}
REPORT_STEM = "RAG-Output_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리"

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
