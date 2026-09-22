"""실행 진입점.

RAG 색인(FAISS+BM25)을 만들고, D-2 그래프를 처음부터 끝까지 실행해 평가 보고서를
outputs/ 에 Markdown + PDF로 저장한다.

사용법:
    python -m kv_eval.main
    (환경변수) OPENAI_API_KEY 필수. TAVILY_API_KEY는 선택
    — 없으면 tools.web_search가 DuckDuckGo(ddgs)로 자동 대체한다.
"""
from __future__ import annotations

import argparse
import sys

from .config import REPORT_STEM, Settings
from .graph import build_graph
from .llm import StructuredLLM
from .rag.index import build_index
from .rag.ingest import load_prepared_chunks
from .rag.retrieve import HybridRetriever
from .rag.workflow import RAGWorkflow
from .reporting.export import export_report
from .state import initial_state
from .tools.web_search import WebClient

DEFAULT_USER_QUERY = (
    "동일한 KV cache 메모리 병목을 해결하는 SW 압축 방식(DeepSeek-V2 MLA)과 "
    "HW 메모리 확장 방식(ITME)은 기술 성숙도, 시장성, 이해관계자 및 데이터센터 적용성 "
    "관점에서 각각 어떻게 평가되는가?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache 최적화 기술 다관점 평가 Agentic RAG")
    parser.add_argument("--query", default=DEFAULT_USER_QUERY, help="평가 요청 문구(기본값: A-3 핵심 질문)")
    args = parser.parse_args()

    settings = Settings.from_env()
    if not settings.openai_key:
        print("OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        raise SystemExit(1)
    if not settings.tavily_key:
        print("TAVILY_API_KEY가 없어 시장·이해관계자 평가는 DuckDuckGo로 대체합니다.", file=sys.stderr)

    print("[1/4] RAG 색인 준비 중 (Qwen3-Embedding-0.6B + FAISS + BM25)...")
    chunks = load_prepared_chunks()
    store = build_index(chunks)
    retriever = HybridRetriever(store)

    llm = StructuredLLM(api_key=settings.openai_key)
    rag = RAGWorkflow(retriever, llm)
    web = WebClient()

    print("[2/4] 그래프 실행 중 (Master → 기술조사 → 병렬평가 → 종합 → 보고서)...")
    graph = build_graph(rag, web, llm)
    final_state = graph.invoke(initial_state(args.query), config={"recursion_limit": 200})

    if final_state.get("status") != "completed":
        print(f"경고: status={final_state.get('status')} (완료되지 못했습니다)", file=sys.stderr)

    print("[3/4] 보고서 내보내는 중 (Markdown + PDF)...")
    paths = export_report(final_state["report_draft"], str(settings.output_dir), REPORT_STEM)

    print("[4/4] 완료:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
