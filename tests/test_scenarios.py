"""Fake 시나리오별 라우팅과 종료 테스트."""
from __future__ import annotations

from fakes import INF, FakeLLM, FakeWeb
from kv_eval.config import PERSPECTIVES, RETRY_LIMITS
from kv_eval.observability import read_decisions
from kv_eval.supervisor.policy import Policy


def decisions(state):
    return [d["decision"] for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]


def test_normal_pass(run_graph):
    llm = FakeLLM()
    state = run_graph(llm)
    assert state["status"] == "completed"
    assert state["eval_result"]["passed"] is True
    assert set(state["perspectives"]) == set(PERSPECTIVES)
    assert dict(llm.runs) == {"tech": 1, "market": 1, "stakeholder": 1, "domain": 1,
                              "synthesis": 1, "report": 1, "judge": 1}
    # 4관점을 한 번에 fan-out한 뒤 종합, 보고서, 평가 순으로 끝난다
    assert decisions(state) == ["dispatch:tech,market,stakeholder,domain", "synthesis", "report", "evaluate",
                                "end:passed"]
    for criterion in ("groundedness", "neutrality", "bias_control", "coverage"):
        c = state["eval_result"]["criteria"][criterion]
        assert set(c) >= {"passed", "score", "reason", "target_agents"}
    assert "rag_cache" not in str(state["perspectives"])  # 원문 캐시는 디스크에 둔다
    assert set(state["cache_keys"]) == {"tech", "domain"}


def test_market_insufficient_reruns_only_market(run_graph):
    llm, web = FakeLLM(insufficient={"market": 1}), FakeWeb()
    state = run_graph(llm, web=web)
    assert llm.runs["market"] == 2
    assert all(llm.runs[name] == 1 for name in ("tech", "stakeholder", "domain"))
    assert state["retry_counts"]["market"] == 1
    assert decisions(state)[:2] == ["dispatch:tech,market,stakeholder,domain", "dispatch:market"]
    # missing 항목이 재검색 질의로 들어간다
    assert any("시장 규모 정량 근거" in q for q in web.queries)
    assert state["status"] == "completed"


def test_eval_neutrality_failure_rewrites_report_only(run_graph):
    llm = FakeLLM(banned_report=1)
    state = run_graph(llm)
    assert llm.runs["report"] == 2 and llm.runs["judge"] == 2
    assert all(llm.runs[name] == 1 for name in (*PERSPECTIVES, "synthesis"))
    assert "rewrite:report" in decisions(state)
    assert state["eval_result"]["passed"] is True
    assert "우열" in str(state["feedback"]["report"]["issues"])


def test_one_sided_verdict_is_reworked_before_report(run_graph):
    # 판정이 긍정 한쪽뿐이면 보고서를 쓰기 전에 충분성 단계에서 그 관점만 반대 방향으로 재조사한다
    llm = FakeLLM(one_sided={"market": 1})
    state = run_graph(llm)
    assert decisions(state)[:3] == ["dispatch:tech,market,stakeholder,domain", "dispatch:market", "synthesis"]
    assert llm.runs["market"] == 2
    assert all(llm.runs[name] == 1 for name in ("tech", "stakeholder", "domain"))
    # 보고서 전에 고쳤으므로 평가 미달·재작성 없이 한 번에 통과한다
    assert llm.runs["synthesis"] == 1 and llm.runs["report"] == 1 and llm.runs["judge"] == 1
    assert state["eval_result"]["passed"] is True and state["retry_counts"]["market"] == 1
    # 긍정 일색이었으니 재조사는 우려 쪽 근거를 찾는다
    feedback = state["feedback"]["market"]
    assert "concerns" in " ".join(feedback["rewritten_queries"])
    assert any("우려 방향 근거를 찾아" in m for m in feedback["missing"])


def test_eval_bias_failure_reinvestigates_responsible_perspective(run_graph):
    llm = FakeLLM(judge_fail={"bias_control": (1, "market")})
    state = run_graph(llm)
    assert "reinvestigate:market" in decisions(state)
    assert llm.runs["market"] == 2
    assert all(llm.runs[name] == 1 for name in ("tech", "stakeholder", "domain"))
    # 관점이 바뀌었으므로 종합·보고서·평가는 다시 만든다
    assert llm.runs["synthesis"] == 2 and llm.runs["report"] == 2 and llm.runs["judge"] == 2
    assert state["eval_result"]["passed"] is True


def test_eval_reinvestigation_still_available_after_synthesis_followup(run_graph):
    # 실제 실행 77cb99ef의 실패 경로: 종합이 market 후속 재조사를 먼저 쓰고, 평가가 market을 원인으로 지목했다.
    # 두 한도가 같은 카운터였을 때는 재조사하지 못하고 재작성만 반복하다 unverified로 끝났다.
    llm = FakeLLM(needs_source=["market"], judge_fail={"bias_control": (1, "market")})
    state = run_graph(llm)
    log = decisions(state)
    first_synthesis = log.index("synthesis")
    assert log[first_synthesis + 1] == "dispatch:market"          # 종합 요청 후속 재조사
    assert "reinvestigate:market" in log[log.index("evaluate"):]   # 평가 요청 재조사도 가능
    assert state["followup_counts"]["market"] == 1 and state["reinvestigate_counts"]["market"] == 1
    assert "rewrite:report" not in log
    assert log[-1] == "end:passed" and state["status"] == "completed"


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
    # Judge가 통과를 줘도 금지 표현 규칙이 실패하면 미달
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


def test_always_insufficient_terminates_with_gaps(run_graph):
    llm = FakeLLM(insufficient={name: INF for name in PERSPECTIVES})
    state = run_graph(llm)
    for name in PERSPECTIVES:
        assert llm.runs[name] == 1 + RETRY_LIMITS[name]
        assert state["perspective_status"][name] == "excluded"
        assert any(gap["perspective"] == name for gap in state["gaps"])
    assert state["status"] == "completed_with_gaps"
    assert state["report"] and "근거 공백 (Supervisor 기록)" in state["report"]
    assert decisions(state)[-1].startswith("end:")
    assert state["step_count"] <= Policy().max_steps


def test_step_limit_ends_gracefully_with_report(run_graph):
    llm = FakeLLM(insufficient={name: INF for name in PERSPECTIVES})
    state = run_graph(llm, policy=Policy(max_steps=1))
    assert all(llm.runs[name] == 1 for name in PERSPECTIVES)  # 상한 도달 후에는 재조사하지 않는다
    assert any(gap["kind"] == "step_limit" and "단계 상한" in gap["detail"] for gap in state["gaps"])
    assert state["report"] and state["status"] in ("completed_with_gaps", "unverified")
    log = read_decisions(state["trace_id"])
    assert any("단계 상한" in d["reason"] for d in log if d["node"] == "supervisor")
    assert log[-1]["decision"].startswith("end:")


def _assert_passed_only_after_evaluation(state):
    """end:passed면 마지막 보고서 뒤에 evaluate가 있어야 한다."""
    log = [d["decision"] for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]
    if log[-1] != "end:passed":
        return
    last_report = max(i for i, d in enumerate(log) if d in ("report", "rewrite:report", "retry:report"))
    assert "evaluate" in log[last_report + 1:], log


def test_end_passed_requires_evaluation_after_last_report(run_graph):
    for llm in (FakeLLM(), FakeLLM(banned_report=1), FakeLLM(one_sided={"market": 1}),
                FakeLLM(raise_on={"report": 1}), FakeLLM(judge_fail={"groundedness": (1, "report")})):
        state = run_graph(llm)
        assert decisions(state)[-1] == "end:passed"
        _assert_passed_only_after_evaluation(state)
        assert llm.runs["judge"] == decisions(state).count("evaluate")


def test_failed_report_is_not_evaluated(run_graph):
    llm = FakeLLM(raise_on={"report": 1})
    state = run_graph(llm)
    log = decisions(state)
    i = log.index("report")
    assert log[i + 1] == "retry:report"          # 실패한 보고서는 평가로 가지 않는다
    assert llm.runs["judge"] == 1                  # 재시도한 보고서만 평가
    assert state["status"] == "completed"


def test_report_always_failing_ends_without_evaluation(run_graph):
    llm = FakeLLM(raise_on={"report": INF})
    state = run_graph(llm)
    assert llm.runs["judge"] == 0 and "evaluate" not in decisions(state)
    assert decisions(state)[-1] == "end:report_failed" and state["status"] == "unverified"
