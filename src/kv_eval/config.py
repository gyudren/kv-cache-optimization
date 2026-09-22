"""Fixed project configuration. No auto-selection or model fallback."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

MODEL_ID = "gpt-5.6-sol"
DOMAIN = "데이터센터·클라우드 기반 장문맥 LLM 서빙"
TECHNOLOGIES = {"mla": "DeepSeek-V2 MLA", "itme": "ITME"}
EMBEDDING_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_PAGES = 200
DECLARED_PAGES = {"deepseek_v2": 52, "itme": 13, "infinigen": 18, "cxl_pnm": 13}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_K = 6  # Implementation detail; same setting across technologies.
RRF_CONSTANT = 60
MIN_RELEVANT = 2
RAG_REWRITES = 2
RETRY_LIMITS = {"tech": 2, "market": 2, "stakeholder": 2, "domain": 2, "synthesis": 1, "report": 2}
REPORT_STEM = "RAG-Output_판교_9반_김민정_김태동_임동건_김동욱_이재겸_박규리"

@dataclass(frozen=True)
class Settings:
    papers_dir: Path
    output_dir: Path
    openai_key: str
    tavily_key: str
    model: str = MODEL_ID

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv()
        model = os.getenv("OPENAI_MODEL", MODEL_ID)
        if model != MODEL_ID:
            raise ValueError(f"Generator/Judge model must be {MODEL_ID}; got {model!r}")
        return cls(
            papers_dir=Path(os.getenv("PAPERS_DIR", "data/papers")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            openai_key=os.getenv("OPENAI_API_KEY", ""),
            tavily_key=os.getenv("TAVILY_API_KEY", ""),
            model=model,
        )

    def require_credentials(self) -> None:
        if not self.openai_key or not self.tavily_key:
            raise RuntimeError("OPENAI_API_KEY and TAVILY_API_KEY are required; see .env.example")
