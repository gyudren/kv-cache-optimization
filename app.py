"""Single CLI entry point. No simulated reports if dependencies/keys/papers missing."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

# Source checkout invocation (also supports pip editable install).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from kv_eval.config import Settings, REPORT_STEM
from kv_eval.rag.ingest import paper_manifest, load_processed_chunks, corpus_stats
from kv_eval.rag.index import build_index
from kv_eval.rag.retrieve import HybridRetriever
from kv_eval.rag.workflow import RAGWorkflow
from kv_eval.llm import StructuredLLM
from kv_eval.tools.web_search import WebSearch
from kv_eval.graph import build_graph
from kv_eval.state import initial_state
from kv_eval.reporting.sections import validate_report
from kv_eval.reporting.export import export_report

DEFAULT_QUERY = ("데이터센터·클라우드 장문맥 LLM 서빙에서 DeepSeek-V2 MLA와 ITME를 "
                 "TRL, 시장성, 이해관계자, 도메인 적용성 관점에서 근거 중심으로 비교 평가하라.")


def run(query: str = DEFAULT_QUERY) -> dict:
    started_at = time.time()
    settings = Settings.from_env()
    settings.require_credentials()
    # 파싱·노이즈 제거·청킹은 preprocessing 파이프라인이 이미 수행했다(python -m preprocessing.pipeline).
    manifest = paper_manifest(settings.manifest_path)
    chunks = load_processed_chunks(settings.chunks_path, manifest)
    print(f"[     0s] 청크 {len(chunks)}개 적재 · 색인 생성 중(FAISS + BM25)...", flush=True)
    stats = corpus_stats(chunks, manifest, settings.summary_path)
    # Do not fabricate the design's 333 chunks or 11 bibliography pages.
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    rag = RAGWorkflow(HybridRetriever(build_index(chunks)), StructuredLLM(settings.openai_key))
    print(f"[{time.time()-started_at:6.0f}s] 색인 완료 · 그래프 실행 시작", flush=True)
    web = WebSearch(settings.tavily_key)
    llm = rag.llm
    graph = build_graph(rag, web, llm)
    # invoke() 대신 stream()을 쓰면 노드가 끝날 때마다 상태를 받아볼 수 있다.
    # LLM 호출이 100회 이상 순차로 일어나므로 어느 단계인지 보이지 않으면 멈춘 것과 구분할 수 없다.
    state = None
    seen_logs = 0
    for state in graph.stream(initial_state(query), config={"recursion_limit": 100},
                              stream_mode="values"):
        for entry in state.get("logs", [])[seen_logs:]:
            elapsed = time.time() - started_at
            detail = f" ({entry['result']})" if entry.get("result") else ""
            print(f"[{elapsed:6.0f}s] {entry['node']}{detail}", flush=True)
        seen_logs = len(state.get("logs", []))
    if state is None:
        raise RuntimeError("Graph produced no state")
    # 최종 State를 남겨 두면 LLM을 다시 호출하지 않고 보고서만 재내보내기할 수 있다(--export-only).
    (settings.output_dir / "final_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return finalize(state, settings.output_dir)


def finalize(state: dict, output_dir: Path) -> dict:
    """검증 → validation.json / 보고서 .md·.pdf / run_logs.json 저장."""
    validation = validate_report(state["report_draft"], state["evidence"])
    gate = next((x for x in reversed(state["logs"]) if x.get("node") == "master_report_gate"), {})
    validation["passed"] = bool(validation["passed"] and gate.get("gate") is True)
    if not gate.get("gate"):
        validation["issues"] = list(dict.fromkeys(validation["issues"] + state.get("review_feedback", {}).get("report", {}).get("issues", [])))
    (output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    export = export_report(state["report_draft"], str(output_dir), REPORT_STEM)
    (output_dir / "run_logs.json").write_text(json.dumps(state["logs"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": state["status"], "verified": validation["passed"],
            "issues": validation["issues"], **export}


def export_only() -> dict:
    """직전 실행의 outputs/final_state.json으로 검증·내보내기만 다시 한다(LLM·웹 호출 없음)."""
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    state_path = output_dir / "final_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"{state_path} 없음: 먼저 python app.py 로 전체 파이프라인을 실행하세요")
    return finalize(json.loads(state_path.read_text(encoding="utf-8")), output_dir)


def report_only() -> dict:
    """직전 실행의 조사·평가·종합 결과(final_state.json)를 그대로 두고 보고서→검수 루프만 다시 돈다.

    그래프의 report / master_report_gate 노드를 같은 순서·같은 재시도 한도(report ≤ 2)로 실행한다.
    새 검색은 하지 않으므로 근거는 전체 실행에서 수집·검증된 Evidence와 동일하다.
    """
    from kv_eval.agents import master, report
    settings = Settings.from_env()
    if not settings.openai_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    state_path = settings.output_dir / "final_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"{state_path} 없음: 먼저 python app.py 로 전체 파이프라인을 실행하세요")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["retry_counts"]["report"] = 0
    state.setdefault("review_feedback", {}).pop("report", None)
    state["phase"], state["status"] = "report", "running"
    llm = StructuredLLM(settings.openai_key)
    started_at = time.time()
    while True:
        for node in (lambda st: report.report_node(st, llm), lambda st: master.master_report_gate_node(st, llm)):
            update = node(state)
            state["logs"] = state["logs"] + update.pop("logs", [])
            state.update(update)
            last = state["logs"][-1]
            print(f"[{time.time()-started_at:6.0f}s] {last['node']} ({last.get('result')})", flush=True)
        if master.route_report(state) == "end":
            break
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return finalize(state, settings.output_dir)


if __name__ == "__main__":
    try:
        args = sys.argv[1:]
        result = export_only() if "--export-only" in args else report_only() if "--report-only" in args else run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["verified"]:
            print("WARNING: report generated but NOT verified for submission", file=sys.stderr)
            sys.exit(2)
    except Exception as exc:
        # No API secrets, raw requests, or private prompts in the failure report.
        message = f"{type(exc).__name__}: {exc}"
        try:
            output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "failure.json").write_text(
                json.dumps({"status": "failed", "error_type": type(exc).__name__},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        print(f"Backend failed: {message}", file=sys.stderr)
        sys.exit(1)
