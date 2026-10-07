"""그래프 구조·reducer·실패 처리·재개·관측성 검증."""
from __future__ import annotations
import asyncio
import json

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from fakes import INF, FakeLLM, FakeRAG, FakeWeb, SimulatedCrash
from kv_eval.config import STATE_EXCERPT_CHARS, LANGSMITH_TAGS, PERSPECTIVES, RETRY_LIMITS
from kv_eval.graph import AGENT_NODES, WORKER_NODES, build_graph
from kv_eval.observability import decision_log_path, new_trace_id, read_decisions, run_config
from kv_eval.state import GraphState, initial_state, merge_dict, merge_evidence
from kv_eval.supervisor.policy import Policy, decide
from kv_eval.supervisor.router import route


# ---- 패턴 정합성 ------------------------------------------------------------------------------
def test_no_agent_to_agent_edges_and_single_router():
    compiled = build_graph(None, None, None)
    edges = [(e.source, e.target, e.conditional) for e in compiled.get_graph().edges]
    workers = set(WORKER_NODES)
    assert set(AGENT_NODES) < workers and "quality_evaluator" in workers
    assert not [e for e in edges if e[0] in workers and e[1] in workers], "작업 노드 간 직접 엣지 금지"
    # 모든 작업 노드(보고서·품질 평가 포함)의 유일한 후속 노드는 supervisor다
    for worker in workers:
        assert {t for s, t, _ in edges if s == worker} == {"supervisor"}, worker
    assert {t for s, t, _ in edges if s == START} == {"supervisor"}
    # 분기는 supervisor의 conditional edge 하나뿐이다
    conditional_sources = {s for s, _, c in edges if c}
    assert conditional_sources == {"supervisor"}
    assert len(compiled.builder.branches["supervisor"]) == 1
    assert {t for s, t, c in edges if s == "supervisor" and c} == workers | {END}


def test_routing_is_decided_by_state_only():
    base = initial_state("q", "t-route")
    first = decide(base)
    assert first.targets == list(PERSPECTIVES)
    # 같은 State면 같은 결정, market만 부족하면 market만
    evidence = [{"agent": p, "technology": t, "source_type": "web", "url": f"https://x/{p}/{t}/{i}", "claim": "c",
                 "source_id": f"{p}{t}{i}"} for p in PERSPECTIVES for t in ("mla", "itme") for i in range(2)]
    state = {**base, "evidence": evidence,
             "node_status": {**base["node_status"], **{p: "done" for p in PERSPECTIVES}},
             "perspectives": {p: {"sufficient": p != "market", "missing": ["x"]} for p in PERSPECTIVES}}
    assert decide(state).targets == ["market"]
    assert decide(state).targets == ["market"]
    sends = route({**state, "next_agents": ["market"]})
    assert isinstance(sends, list) and all(isinstance(s, Send) for s in sends) and sends[0].node == "market"
    assert route({**state, "next_agents": ["__end__"]}) == END
    assert route({**state, "next_agents": ["synthesis"]}) == "synthesis"


# ---- reducer 동시 쓰기 --------------------------------------------------------------------------
def test_reducers_merge_and_dedup():
    assert merge_dict({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    long = {"source_id": "s1", "claim": "c", "page": 1, "excerpt": "x" * (STATE_EXCERPT_CHARS + 500)}
    merged = merge_evidence([long], [dict(long), {"source_id": "s2", "claim": "c", "url": "u", "excerpt": "y"}])
    assert len(merged) == 2
    assert len(merge_evidence([], [long])[0]["excerpt"]) == STATE_EXCERPT_CHARS


def test_parallel_send_writes_are_merged():
    """4개 노드가 같은 superstep에서 perspectives·node_status·evidence에 동시에 써도 모두 남는다."""
    def worker(name):
        def node(state):
            return {"perspectives": {name: {"sufficient": True}}, "node_status": {name: "done"},
                    "evidence": [{"source_id": f"{name}-1", "claim": name, "url": name, "excerpt": "e"},
                                 {"source_id": "shared", "claim": "same", "url": "same", "excerpt": "e"}]}
        return node

    graph = StateGraph(GraphState)
    graph.add_node("fan", lambda state: {})
    for name in PERSPECTIVES:
        graph.add_node(name, worker(name))
        graph.add_edge(name, END)
    graph.add_edge(START, "fan")
    graph.add_conditional_edges("fan", lambda s: [Send(n, s) for n in PERSPECTIVES], list(PERSPECTIVES))
    out = graph.compile().invoke(initial_state("q", "t-par"))
    assert set(out["perspectives"]) == set(PERSPECTIVES)
    assert all(out["node_status"][p] == "done" for p in PERSPECTIVES)
    assert len(out["evidence"]) == len(PERSPECTIVES) + 1  # 공통 근거는 한 번만


# ---- 실패 처리(fallback) ----------------------------------------------------------------------
def test_agent_exception_is_retried_then_recovers(run_graph):
    llm = FakeLLM(raise_on={"market": 1})
    state = run_graph(llm)
    log = read_decisions(state["trace_id"])
    assert any("실행 실패" in d["reason"] and d["decision"] == "dispatch:market" for d in log)
    assert state["last_error"]["market"] == "" and state["node_status"]["market"] == "done"
    assert state["status"] == "completed"


def test_agent_always_failing_is_excluded_and_reported_as_gap(run_graph):
    llm = FakeLLM(raise_on={"stakeholder": INF})
    state = run_graph(llm)  # 예외가 그래프 밖으로 나오지 않는다
    assert llm.runs["stakeholder"] == 1 + RETRY_LIMITS["stakeholder"]
    assert state["perspective_status"]["stakeholder"] == "excluded"
    gap = next(g for g in state["gaps"] if g.startswith("stakeholder:"))
    assert "실행 실패" in gap and "RuntimeError" in gap
    section = state["report"].split("#### 근거 공백 (Supervisor 기록)", 1)[1]
    assert "**이해관계자**" in section and "실행 실패" in section  # 7장 한계점에 읽을 수 있는 형태로 기록
    assert state["status"] == "completed_with_gaps"


# ---- 재개(체크포인트) --------------------------------------------------------------------------
def test_resume_from_sqlite_checkpoint_skips_completed_nodes(tmp_path):
    llm = FakeLLM(crash_on={"synthesis": 1})
    path = str(tmp_path / "ckpt.sqlite")
    trace_id = new_trace_id()
    config = run_config(trace_id)

    async def first_process():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            graph = build_graph(FakeRAG(), FakeWeb(), llm, checkpointer=saver)
            with pytest.raises(SimulatedCrash):
                await graph.ainvoke(initial_state("q", trace_id), config=config)
            return await graph.aget_state(config)

    snapshot = asyncio.run(first_process())
    assert snapshot.next == ("synthesis",)
    assert set(snapshot.values["perspectives"]) == set(PERSPECTIVES)

    # 새 프로세스를 흉내: 같은 파일로 체크포인터와 그래프를 다시 만든 뒤 입력 None으로 이어서 실행
    async def second_process():
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            return await build_graph(FakeRAG(), FakeWeb(), llm, checkpointer=saver).ainvoke(None, config=config)

    state = asyncio.run(second_process())
    assert state["status"] == "completed" and state["trace_id"] == trace_id
    assert all(llm.runs[name] == 1 for name in PERSPECTIVES)  # 완료된 관점은 다시 실행하지 않음
    assert llm.runs["synthesis"] == 2


# ---- 관측성 -----------------------------------------------------------------------------------
def test_decision_log_and_langsmith_config(run_graph):
    state = run_graph(FakeLLM())
    path = decision_log_path(state["trace_id"])
    assert path.name == f"decisions_{state['trace_id']}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows and all({"trace_id", "step", "node", "decision", "reason", "ts"} <= set(r) for r in rows)
    assert all(r["trace_id"] == state["trace_id"] for r in rows)
    assert state["last_decision"] == rows[-1]  # State에는 마지막 결정 1건만
    config = run_config(state["trace_id"])
    assert config["configurable"]["thread_id"] == config["metadata"]["trace_id"] == state["trace_id"]
    assert config["tags"] == LANGSMITH_TAGS and "pattern:supervisor" in config["tags"]
    assert config["run_name"]


def test_supervisor_verifies_sufficiency_deterministically():
    """에이전트가 sufficient=True라고 해도 고유 출처가 부족하면 Supervisor가 그 관점만 재조사시킨다."""
    base = initial_state("q", "t-suff")
    evidence = [{"agent": p, "technology": t, "source_type": "web", "url": f"https://x/{p}/{t}/{i}", "claim": "c",
                 "source_id": f"{p}{t}{i}"} for p in PERSPECTIVES for t in ("mla", "itme") for i in range(2)]
    evidence = [ev for ev in evidence if not (ev["agent"] == "market" and ev["technology"] == "itme" and ev["url"].endswith("/1"))]
    state = {**base, "evidence": evidence,
             "node_status": {**base["node_status"], **{p: "done" for p in PERSPECTIVES}},
             "perspectives": {p: {"sufficient": True, "missing": []} for p in PERSPECTIVES}}
    decision = decide(state)
    assert decision.targets == ["market"]
    assert any("market/itme: 고유 출처 1개" in m for m in decision.updates["feedback"]["market"]["missing"])


def test_recursion_limit_follows_policy_instance():
    """Policy(max_steps=30)으로 상한까지 도는 실행도 GraphRecursionError 없이 끝난다(D-20)."""
    from kv_eval.config import FINALIZE_STEPS
    policy = Policy(max_steps=30)
    graph = build_graph(FakeRAG(), FakeWeb(), FakeLLM(insufficient={p: INF for p in PERSPECTIVES}), policy=policy)
    assert graph.config["recursion_limit"] == policy.recursion_limit == (30 + FINALIZE_STEPS) * 2 + 10
    assert "recursion_limit" not in run_config("t")
    # 재조사 한도를 크게 열어 max_steps까지 실제로 진행시킨다
    wide = Policy(max_steps=30, retry_limits={**RETRY_LIMITS, **{p: 100 for p in PERSPECTIVES}})
    graph = build_graph(FakeRAG(), FakeWeb(), FakeLLM(insufficient={p: INF for p in PERSPECTIVES}), policy=wide)
    trace_id = new_trace_id()
    state = asyncio.run(graph.ainvoke(initial_state("q", trace_id), config=run_config(trace_id)))
    assert state["step_count"] > 30 and state["status"] in ("completed_with_gaps", "unverified")
    assert any("단계 상한" in gap for gap in state["gaps"])
