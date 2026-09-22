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
        update = master.master_tech_gate_node(state)
        state.update(update)
        if attempt < 2:
            assert master.route_tech(state) == "technology"
    assert state["retry_counts"]["tech"] == 2
    assert state["next_agents"] == ["market", "stakeholder", "domain"]
    assert state["tech_result"]["missing"] == ["TRL 근거 부족"]


def test_join_retries_only_missing_agent_then_proceeds():
    state = initial_state("query")
    state["phase"] = "parallel_eval"
    state["market_result"] = {"sufficient": True}
    state["stakeholder_result"] = {"sufficient": False, "missing": ["발언"]}
    state["domain_result"] = {"sufficient": True}
    state.update(master.master_join_node(state))
    assert state["next_agents"] == ["stakeholder"]
    assert state["retry_counts"]["stakeholder"] == 1
    assert state["retry_counts"]["market"] == 0
    state["stakeholder_result"] = {"sufficient": True}
    state.update(master.master_join_node(state))
    assert state["phase"] == "synthesis"
    assert state["next_agents"] == []


def test_synthesis_source_reassign_and_limit():
    state = initial_state("query")
    state["synthesis_result"] = {"needs_source_agents": ["market"], "evidence_gaps": ["adoption missing"]}
    state.update(master.master_synthesis_gate_node(state))
    assert state["next_agents"] == ["market"]
    assert state["retry_counts"]["market"] == 1
    state.update(master.master_synthesis_gate_node(state))
    assert state["retry_counts"]["market"] == 2
    state.update(master.master_synthesis_gate_node(state))
    assert state["phase"] == "report"


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
