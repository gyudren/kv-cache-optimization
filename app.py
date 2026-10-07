"""Single CLI entry point. No simulated reports if dependencies/keys/papers missing.

    python app.py                      # 새 실행 (trace_id 발급)
    python app.py --resume <trace_id>  # 체크포인트(SQLite)에서 이어서 실행
    python app.py --report-only        # 직전 결과로 보고서 → 품질 평가 루프만 다시 실행
    python app.py --export-only        # LLM 호출 없이 검증·내보내기만 다시 실행
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Source checkout invocation (also supports pip editable install).
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from kv_eval.config import PERSPECTIVES, REPORT_STEM, RETRY_LIMITS, Settings
from kv_eval.graph import build_graph
from kv_eval.observability import (langsmith_enabled, new_trace_id, read_decisions, run_config)
from kv_eval.state import initial_state
from kv_eval.supervisor.policy import Policy

DEFAULT_QUERY = ("데이터센터·클라우드 장문맥 LLM 서빙에서 DeepSeek-V2 MLA와 ITME를 "
                 "TRL, 시장성, 이해관계자, 도메인 적용성 관점에서 근거 중심으로 비교 평가하라.")


def configure_tracing() -> None:
    """LANGSMITH_TRACING=true인데 키가 없으면 전송 실패 경고만 쌓이므로 트레이싱을 끄고 알린다."""
    if os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and not langsmith_enabled():
        os.environ["LANGSMITH_TRACING"] = "false"
        print("[trace] LANGSMITH_API_KEY 없음: LangSmith 트레이싱 비활성(결정 로그 JSONL은 계속 기록)", flush=True)


def checkpoint_path(output_dir: Path) -> Path:
    return Path(os.getenv("CHECKPOINT_PATH", str(output_dir / "checkpoints.sqlite")))


def open_checkpointer(path: Path):
    """파일 기반 체크포인터. thread_id=trace_id로 저장되어 프로세스가 죽어도 --resume으로 이어 간다."""
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def build_runtime(settings: Settings, started_at: float):
    """RAG(FAISS+BM25)·웹 검색·LLM을 준비한다. 무거운 import는 여기서만 한다."""
    from kv_eval.llm import StructuredLLM
    from kv_eval.rag.index import build_index
    from kv_eval.rag.ingest import corpus_stats, load_processed_chunks, paper_manifest
    from kv_eval.rag.retrieve import HybridRetriever
    from kv_eval.rag.workflow import RAGWorkflow
    from kv_eval.tools.web_search import WebSearch
    # 파싱·노이즈 제거·청킹은 preprocessing 파이프라인이 이미 수행했다(python -m preprocessing.pipeline).
    manifest = paper_manifest(settings.manifest_path)
    chunks = load_processed_chunks(settings.chunks_path, manifest)
    print(f"[     0s] 청크 {len(chunks)}개 적재 · 색인 생성 중(FAISS + BM25)...", flush=True)
    stats = corpus_stats(chunks, manifest, settings.summary_path)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "corpus_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    rag = RAGWorkflow(HybridRetriever(build_index(chunks)), StructuredLLM(settings.openai_key))
    print(f"[{time.time()-started_at:6.0f}s] 색인 완료 · 그래프 실행 시작", flush=True)
    return rag, WebSearch(settings.tavily_key), rag.llm


def execute(graph, trace_id: str, initial: dict | None, started_at: float) -> dict:
    """그래프를 stream으로 실행한다. initial=None이면 체크포인트에서 이어서 실행한다(resume).

    LLM 호출이 100회 이상 일어나므로 노드가 끝날 때마다 진행 상황과 Supervisor 결정 사유를 출력한다.
    """
    config = run_config(trace_id)
    for update in graph.stream(initial, config=config, stream_mode="updates"):
        for node, delta in update.items():
            elapsed = time.time() - started_at
            if node == "supervisor" and isinstance(delta, dict):
                decision = delta.get("last_decision", {})
                print(f"[{elapsed:6.0f}s] supervisor#{decision.get('step')} → {decision.get('decision')} "
                      f"({decision.get('reason', '')[:140]})", flush=True)
            elif isinstance(delta, dict):
                status = (delta.get("node_status") or {}).get(node, "")
                error = (delta.get("last_error") or {}).get(node, "")
                print(f"[{elapsed:6.0f}s] {node} {status}{f' ({error[:120]})' if error else ''}", flush=True)
    return graph.get_state(config).values


def run(query: str = DEFAULT_QUERY, resume: str | None = None) -> dict:
    started_at = time.time()
    settings = Settings.from_env()
    settings.require_credentials()
    trace_id = resume or new_trace_id()
    configure_tracing()
    print(f"[trace] trace_id={trace_id} · LangSmith "
          + (f"ON (project={os.getenv('LANGSMITH_PROJECT', 'default')})" if langsmith_enabled() else "OFF"), flush=True)
    checkpointer = open_checkpointer(checkpoint_path(settings.output_dir))
    if resume:
        # 색인(수 분)을 만들기 전에 체크포인트부터 확인한다. 상태 조회에는 런타임이 필요 없다.
        snapshot = build_graph(None, None, None, checkpointer=checkpointer).get_state(run_config(trace_id))
        if not snapshot.values:
            raise ValueError(f"체크포인트에 trace_id={trace_id} 실행이 없습니다")
        if not snapshot.next:
            print("[resume] 이미 종료된 실행입니다. 결과만 다시 내보냅니다.", flush=True)
            return save_and_finalize(snapshot.values, settings.output_dir)
    rag, web, llm = build_runtime(settings, started_at)
    graph = build_graph(rag, web, llm, checkpointer=checkpointer)
    if resume:
        # 입력 None = 마지막 체크포인트부터 이어서 실행(완료된 노드는 다시 실행하지 않음)
        state = execute(graph, trace_id, None, started_at)
    else:
        state = execute(graph, trace_id, initial_state(query, trace_id), started_at)
    return save_and_finalize(state, settings.output_dir)


def save_and_finalize(state: dict, output_dir: Path) -> dict:
    # 최종 State를 남겨 두면 LLM을 다시 호출하지 않고 보고서만 재내보내기할 수 있다(--export-only).
    # RAG 캐시와 결정 로그 본문은 State 밖에 있으므로 이 파일은 작게 유지된다.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return finalize(state, output_dir)


def from_legacy(state: dict) -> dict:
    """이전 과제(RAG) 실행의 final_state.json(필드명 *_result/report_draft)을 현재 State 형태로 바꾼다."""
    if "perspectives" in state:
        return state
    names = {"tech": "tech_result", "market": "market_result", "stakeholder": "stakeholder_result", "domain": "domain_result"}
    return {"user_query": state.get("user_query", DEFAULT_QUERY), "trace_id": state.get("trace_id", "legacy"),
            "perspectives": {k: state.get(v, {}) for k, v in names.items()},
            "synthesis": state.get("synthesis_result", {}), "report": state.get("report_draft", ""),
            "evidence": state.get("evidence", []), "gaps": [], "eval_result": {}, "status": state.get("status", "")}


def finalize(state: dict, output_dir: Path) -> dict:
    """검증 → validation.json / 보고서 .md·.pdf / run_logs.json(결정 로그) 저장."""
    from kv_eval.reporting.export import export_report
    from kv_eval.reporting.sections import validate_report
    (output_dir / "failure.json").unlink(missing_ok=True)  # 이전 실행의 실패 기록이 남아 혼동되지 않게
    trace_id = state.get("trace_id", "")
    report = state.get("report", "")
    eval_result = state.get("eval_result") or {}
    decisions = read_decisions(trace_id) if trace_id else []
    if decisions:  # 결정 로그가 없는 실행(이전 형식 State 내보내기 등)은 기존 run_logs.json을 덮어쓰지 않는다
        (output_dir / "run_logs.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report.strip():
        validation = {"passed": False, "issues": ["보고서 미생성"], "gaps": state.get("gaps", []), "trace_id": trace_id}
        (output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": state.get("status"), "trace_id": trace_id, "verified": False, "issues": validation["issues"]}
    validation = validate_report(report, state.get("evidence", []))
    criteria = eval_result.get("criteria", {})
    failed = [f"품질 평가 {name} 미달: {c.get('reason', '')[:300]}" for name, c in criteria.items() if not c.get("passed")]
    if not eval_result:
        failed.append("품질 평가 결과 없음")
    validation.update({
        "passed": bool(validation["passed"] and eval_result.get("passed")),
        "issues": list(dict.fromkeys(validation["issues"] + failed)),
        "quality_eval": {name: {k: c.get(k) for k in ("passed", "score", "reason", "target_agents")}
                         for name, c in criteria.items()},
        "gaps": state.get("gaps", []), "trace_id": trace_id, "status": state.get("status"),
        "supervisor_steps": state.get("step_count"), "retry_counts": state.get("retry_counts", {}),
    })
    from kv_eval.evidence_store import missing_full_text
    store_missing = missing_full_text(state.get("evidence", []))
    validation["evidence_store"] = {"items": len(state.get("evidence", [])), "missing_full_text": store_missing,
                                    "evaluator": eval_result.get("evidence_store", {})}
    validation["warnings"] = list(dict.fromkeys(
        validation.get("warnings", []) + eval_result.get("warnings", [])
        + ([f"Evidence {store_missing}건의 원문이 저장소(data/cache/)에 없음: 축약 발췌만 남아 있음"] if store_missing else [])))
    # 10p 상한은 validate_report가 같은 조판으로 이미 검사했다. 실제 파일의 페이지 수를 함께 남긴다.
    export = export_report(report, str(output_dir), REPORT_STEM)
    validation["pdf_pages"] = export["pages"]
    (output_dir / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": state.get("status"), "trace_id": trace_id, "verified": validation["passed"],
            "issues": validation["issues"], **export}


def _load_final_state(output_dir: Path) -> dict:
    state_path = output_dir / "final_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"{state_path} 없음: 먼저 python app.py 로 전체 파이프라인을 실행하세요")
    return from_legacy(json.loads(state_path.read_text(encoding="utf-8")))


def export_only() -> dict:
    """직전 실행의 outputs/final_state.json으로 검증·내보내기만 다시 한다(LLM·웹 호출 없음)."""
    return finalize(_load_final_state(Path(os.getenv("OUTPUT_DIR", "outputs"))), Path(os.getenv("OUTPUT_DIR", "outputs")))


def report_only() -> dict:
    """직전 실행의 조사·평가·종합 결과를 그대로 두고 보고서 → 품질 평가 루프만 다시 돈다.

    같은 Supervisor 그래프를 쓰되 관점 재조사 한도를 0으로 둔다(새 검색 없음). 평가가 관점 재조사를
    요구하면 근거 공백으로 기록하고 보고서 재작성으로 처리한다.
    """
    from kv_eval import evidence_store
    from kv_eval.llm import StructuredLLM
    settings = Settings.from_env()
    if not settings.openai_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    configure_tracing()  # LLM 클라이언트(wrap_openai 여부)를 만들기 전에 트레이싱 설정을 확정한다
    previous = _load_final_state(settings.output_dir)
    # 보고서·평가는 발췌 원문이 필요하다. final_state에는 축약본과 참조만 있으므로 원문 저장소가 없으면
    # (새로 clone한 저장소 등) 축약 발췌로 조용히 다시 쓰지 않고 여기서 멈춘다.
    store_stats: dict = {}
    hydrated = evidence_store.hydrate(previous.get("evidence", []), store_stats)
    if store_stats.get("missing_full_text"):
        raise RuntimeError(
            f"--report-only 불가: Evidence {store_stats['missing_full_text']}/{len(hydrated)}건의 원문이 저장소에 없습니다 "
            "(data/cache/<trace_id>/evidence_*.json은 git에 포함되지 않음). 원래 실행한 머신에서 다시 하거나 "
            "python app.py로 전체 실행하세요.")
    trace_id = new_trace_id()
    seed = {key: previous.get(key) for key in ("perspectives", "synthesis", "evidence", "gaps", "cache_keys")
            if previous.get(key) is not None}
    # 이전 형식 State의 원문 발췌도 저장소로 옮겨 State에는 축약본만 넣는다(보고서는 hydrate로 원문 사용).
    seed["evidence"] = evidence_store.offload(trace_id, "seed", hydrated)
    seed["perspective_status"] = {name: "sufficient" if (previous["perspectives"].get(name) or {}).get("sufficient")
                                  else "excluded" for name in PERSPECTIVES}
    seed["node_status"] = {**{name: "done" for name in (*PERSPECTIVES, "synthesis")},
                           "report": "pending", "quality_evaluator": "pending"}
    policy = Policy(retry_limits={**RETRY_LIMITS, **{name: 0 for name in PERSPECTIVES}, "synthesis": 0})
    graph = build_graph(None, None, StructuredLLM(settings.openai_key),
                        checkpointer=open_checkpointer(checkpoint_path(settings.output_dir)), policy=policy)
    print(f"[trace] trace_id={trace_id} (report-only)", flush=True)
    state = execute(graph, trace_id, initial_state(previous.get("user_query", DEFAULT_QUERY), trace_id, seed), time.time())
    return save_and_finalize(state, settings.output_dir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KV cache 기술 다관점 평가 (LangGraph Supervisor)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", metavar="TRACE_ID", help="체크포인트에서 해당 trace_id 실행을 이어서 진행")
    mode.add_argument("--report-only", action="store_true", help="직전 결과로 보고서·품질 평가 루프만 재실행")
    mode.add_argument("--export-only", action="store_true", help="LLM 호출 없이 검증·내보내기만 재실행")
    parser.add_argument("--query", default=None, help="평가 요청 문장(새 실행에만 적용)")
    args = parser.parse_args(argv)
    if args.query is not None and (args.resume or args.report_only or args.export_only):
        parser.error("--query는 새 실행에만 쓸 수 있습니다(재개·재내보내기는 저장된 요청을 그대로 사용)")
    args.query = args.query or DEFAULT_QUERY
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    try:
        result = (export_only() if args.export_only else report_only() if args.report_only
                  else run(args.query, resume=args.resume))
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
