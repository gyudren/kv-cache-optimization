from kv_eval.state import initial_state, deduplicate_evidence
from kv_eval.config import RETRY_LIMITS
from kv_eval.agents import master


def test_state_contract():
    s = initial_state("test")
    assert list(s["retry_counts"]) == list(RETRY_LIMITS)
    assert s["retry_counts"]["synthesis"] == 0
    assert s["phase"] == "init" and s["status"] == "running"
    assert "judge_count" not in s


def test_tech_gate_exhaust_preserves_missing():
    state = initial_state("query")
    state["tech_result"] = {"sufficient": False, "missing": ["TRL 근거 부족"]}
    for attempt in range(3):
        state.update(master.master_tech_gate_node(state))
        if attempt < 2:
            assert master.route_tech(state) == "master_query_rewrite"
            state.update(master.master_query_rewrite_node(state))
    assert state["retry_counts"]["tech"] == 2
    assert state["next_agents"] == ["market", "stakeholder", "domain"]
    assert state["tech_result"]["missing"] == ["TRL 근거 부족"]


def test_query_rewrite_passes_missing_items_to_technology():
    state = initial_state("query")
    state["tech_result"] = {"sufficient": False, "missing": ["MLA KV 감소율 근거 없음"]}
    state.update(master.master_tech_gate_node(state))
    state.update(master.master_query_rewrite_node(state))
    assert state["retry_counts"]["tech"] == 1
    assert state["review_feedback"]["tech"]["rewritten_queries"] == ["MLA KV 감소율 근거 없음"]


def test_join_retries_only_missing_agent_then_proceeds():
    state = initial_state("query")
    state["phase"] = "parallel_eval"
    state["market_result"] = {"sufficient": True}
    state["stakeholder_result"] = {"sufficient": False, "missing": ["발언"]}
    state["domain_result"] = {"sufficient": True}
    state.update(master.master_join_node(state))
    assert master.route_join(state) == "master_retry"
    assert state["next_agents"] == ["stakeholder"]
    state.update(master.master_retry_node(state))
    assert state["retry_counts"]["stakeholder"] == 1
    assert state["retry_counts"]["market"] == 0
    assert state["review_feedback"]["stakeholder"]["missing"] == ["발언"]
    state["stakeholder_result"] = {"sufficient": True}
    state.update(master.master_join_node(state))
    assert state["phase"] == "synthesis"
    assert state["next_agents"] == []
    assert master.route_join(state) == "synthesis"


def test_join_stops_retrying_after_limit():
    state = initial_state("query")
    state["phase"] = "parallel_eval"
    state["market_result"] = {"sufficient": False, "missing": ["시장 근거 없음"]}
    state["stakeholder_result"] = {"sufficient": True}
    state["domain_result"] = {"sufficient": True}
    for _ in range(RETRY_LIMITS["market"]):
        state.update(master.master_join_node(state))
        assert state["next_agents"] == ["market"]
        state.update(master.master_retry_node(state))
    state.update(master.master_join_node(state))
    assert state["retry_counts"]["market"] == RETRY_LIMITS["market"]
    assert state["phase"] == "synthesis"


def test_synthesis_gate_retries_once_then_reports():
    state = initial_state("query")
    state["synthesis_result"] = {"needs_revision": True, "evidence_gaps": ["상충 사례 부족"]}
    state.update(master.master_synthesis_gate_node(state))
    assert master.route_synthesis(state) == "synthesis"
    assert state["retry_counts"]["synthesis"] == RETRY_LIMITS["synthesis"] == 1
    # 한도(1회)를 넘으면 설계대로 보고서 단계로 넘어간다.
    state.update(master.master_synthesis_gate_node(state))
    assert master.route_synthesis(state) == "report"
    assert state["phase"] == "report"


def test_synthesis_gate_never_redispatches_perspective_agents():
    """설계 D-2에 없는 '종합 → 관점 Agent 재할당' 경로가 없는지 확인한다."""
    state = initial_state("query")
    state["synthesis_result"] = {"needs_source_agents": ["market"], "evidence_gaps": ["채택 근거 없음"]}
    state.update(master.master_synthesis_gate_node(state))
    assert state["next_agents"] == []
    assert state["retry_counts"]["market"] == 0
    assert master.route_synthesis(state) == "report"
    # 재검색 대신 근거 공백으로 남겨 보고서 한계점에 드러나게 한다.
    assert any("market" in gap for gap in state["synthesis_result"]["evidence_gaps"])


def test_dedup_evidence_preserves_distinct_claim():
    a = {"source_id": "a", "claim": "one", "page": 1}
    b = {**a, "claim": "two"}
    assert deduplicate_evidence([a, a, b]) == [a, b]


def test_langgraph_compile_if_installed():
    import pytest
    pytest.importorskip("langgraph")
    from kv_eval.graph import build_graph
    dummy = object()
    graph = build_graph(dummy, dummy, dummy)
    assert graph is not None

class JudgeStub:
    def generate_structured(self, prompt, schema):
        from kv_eval.schemas import ReportAssessment
        assert schema is ReportAssessment
        return ReportAssessment(passed=True, issues=[])


def test_report_gate_exhaust_stays_unverified():
    state = initial_state("query")
    state["report_draft"] = "## SUMMARY\n근거 부족"
    for attempt in range(3):
        state.update(master.master_report_gate_node(state, JudgeStub()))
        if attempt < 2:
            assert master.route_report(state) == "report"
    assert master.route_report(state) == "end"
    assert state["retry_counts"]["report"] == 2
    assert state["status"] == "completed"
    assert state["logs"][-1]["gate"] is False
