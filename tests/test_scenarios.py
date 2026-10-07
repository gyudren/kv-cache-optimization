"""Fake LLM·Web·RAG 시나리오: 라우팅이 State에 따라 달라지고, 항상 유한하게 끝나는지 검증한다."""
from __future__ import annotations

from fakes import INF, FakeLLM, FakeWeb
from kv_eval.config import PERSPECTIVES, RETRY_LIMITS
from kv_eval.observability import read_decisions
from kv_eval.supervisor.policy import Policy


def decisions(state):
    return [d["decision"] for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]


# (1) 정상 통과 ------------------------------------------------------------------------------
def test_normal_pass(run_graph):
    llm = FakeLLM()
    state = run_graph(llm)
    assert state["status"] == "completed"
    assert state["eval_result"]["passed"] is True
    assert set(state["perspectives"]) == set(PERSPECTIVES)
    assert dict(llm.runs) == {"tech": 1, "market": 1, "stakeholder": 1, "domain": 1,
                              "synthesis": 1, "report": 1, "judge": 1}
    # 4관점은 기술 선행 없이 한 번에 fan-out되고, 이후 종합 → 보고서 → 종료
    assert decisions(state) == ["dispatch:tech,market,stakeholder,domain", "synthesis", "report", "end:passed"]
    for criterion in ("groundedness", "neutrality", "bias_control", "coverage"):
        c = state["eval_result"]["criteria"][criterion]
        assert set(c) >= {"passed", "score", "reason", "target_agents"}
    assert "rag_cache" not in str(state["perspectives"])  # 원문 캐시는 State 밖(디스크)
    assert set(state["cache_keys"]) == {"tech", "domain"}


# (2) 시장 근거 부족 → 시장만 재호출 ---------------------------------------------------------------
def test_market_insufficient_reruns_only_market(run_graph):
    llm, web = FakeLLM(insufficient={"market": 1}), FakeWeb()
    state = run_graph(llm, web=web)
    assert llm.runs["market"] == 2
    assert all(llm.runs[name] == 1 for name in ("tech", "stakeholder", "domain"))
    assert state["retry_counts"]["market"] == 1
    assert decisions(state)[:2] == ["dispatch:tech,market,stakeholder,domain", "dispatch:market"]
    # 부족 항목(missing)이 재검색 질의로 전달됐다(retry_queries 재사용)
    assert any("시장 규모 정량 근거" in q for q in web.queries)
    assert state["status"] == "completed"


# (3) 품질 평가 미달 → 원인별 루프 --------------------------------------------------------------
def test_eval_neutrality_failure_rewrites_report_only(run_graph):
    llm = FakeLLM(banned_report=1)
    state = run_graph(llm)
    assert llm.runs["report"] == 2 and llm.runs["judge"] == 2
    assert all(llm.runs[name] == 1 for name in (*PERSPECTIVES, "synthesis"))
    assert "rewrite:report" in decisions(state)
    assert state["eval_result"]["passed"] is True
    assert "우열" in str(state["feedback"]["report"]["issues"])


def test_eval_bias_failure_reinvestigates_responsible_perspective(run_graph):
    llm = FakeLLM(one_sided={"market": 1})
    state = run_graph(llm)
    assert "reinvestigate:market" in decisions(state)
    assert llm.runs["market"] == 2
    assert all(llm.runs[name] == 1 for name in ("tech", "stakeholder", "domain"))
    # 관점이 바뀌었으므로 종합·보고서·평가는 다시 만든다
    assert llm.runs["synthesis"] == 2 and llm.runs["report"] == 2 and llm.runs["judge"] == 2
    assert state["eval_result"]["passed"] is True
    # 한쪽(긍정) 판정뿐이었으므로 재조사 질의는 반대 방향(우려) 근거를 찾도록 바뀐다
    assert "concerns" in " ".join(state["feedback"]["market"]["rewritten_queries"])


def test_llm_judge_groundedness_failure_rewrites_report(run_graph):
    llm = FakeLLM(judge_fail={"groundedness": (1, "report")})
    state = run_graph(llm)
    assert "rewrite:report" in decisions(state)
    assert llm.runs["report"] == 2 and llm.runs["market"] == 1


def test_llm_judge_coverage_failure_reinvestigates_target(run_graph):
    llm = FakeLLM(judge_fail={"coverage": (1, "domain")})
    state = run_graph(llm)
    assert "reinvestigate:domain" in decisions(state)
    assert llm.runs["domain"] == 2 and llm.runs["market"] == 1


def test_rule_failure_cannot_be_overruled_by_judge(run_graph):
    # Judge는 항상 통과를 주지만 규칙(금지 표현)이 실패하면 항목은 미달이어야 한다.
    llm = FakeLLM(banned_report=INF)
    state = run_graph(llm)
    neutrality = state["eval_result"]["criteria"]["neutrality"]
    assert neutrality["judge"]["passed"] is True and neutrality["passed"] is False
    assert state["status"] == "unverified"
    assert decisions(state)[-1] == "end:unverified"
    assert llm.runs["report"] == 1 + RETRY_LIMITS["report"]


def test_synthesis_requests_more_evidence_for_one_perspective(run_graph):
    llm = FakeLLM(needs_source=["stakeholder"])
    state = run_graph(llm)
    assert "dispatch:stakeholder" in decisions(state)
    assert llm.runs["stakeholder"] == 2 and llm.runs["market"] == 1 and llm.runs["synthesis"] == 2


# (4) 항상 부족 → 유한 스텝 END + 근거 공백 --------------------------------------------------------
def test_always_insufficient_terminates_with_gaps(run_graph):
    llm = FakeLLM(insufficient={name: INF for name in PERSPECTIVES})
    state = run_graph(llm)
    for name in PERSPECTIVES:
        assert llm.runs[name] == 1 + RETRY_LIMITS[name]
        assert state["perspective_status"][name] == "excluded"
        assert any(gap.startswith(f"{name}:") for gap in state["gaps"])
    assert state["status"] == "completed_with_gaps"
    assert state["report"] and "근거 공백 (Supervisor 기록)" in state["report"]
    assert decisions(state)[-1].startswith("end:")
    assert state["step_count"] <= Policy().max_steps


def test_step_limit_ends_gracefully_with_report(run_graph):
    llm = FakeLLM(insufficient={name: INF for name in PERSPECTIVES})
    state = run_graph(llm, policy=Policy(max_steps=1))
    assert all(llm.runs[name] == 1 for name in PERSPECTIVES)  # 상한 도달 후에는 재조사하지 않는다
    assert any("단계 상한" in gap for gap in state["gaps"])
    assert state["report"] and state["status"] in ("completed_with_gaps", "unverified")
    log = read_decisions(state["trace_id"])
    assert any("단계 상한" in d["reason"] for d in log if d["node"] == "supervisor")
    assert log[-1]["decision"].startswith("end:")
