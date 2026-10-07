"""후속 재조사 한도, 출처 단위, [D] 인용 범위, 근거 교체, 실패 주입 종료 테스트."""
from __future__ import annotations

import pytest

from fakes import INF, FakeLLM
from kv_eval.agents.domain import SINGLE_DOC_NOTE, downgrade_single_document
from kv_eval.config import PERSPECTIVES, RETRY_LIMITS
from kv_eval.evaluation.quality import CRITERIA, check_groundedness, evidence_shortfalls
from kv_eval.evidence_store import source_unit, source_units
from kv_eval.observability import read_decisions
from kv_eval.state import initial_state, merge_evidence
from kv_eval.supervisor.policy import Policy, decide


def decisions(state: dict) -> list[str]:
    return [d["decision"] for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]


def evaluated_state(failing: dict[str, list[str]] | None = None, retry: dict | None = None,
                    followup: dict | None = None, reinvestigate: dict | None = None, **extra) -> dict:
    """평가까지 끝난 State. failing={평가 항목: 원인 에이전트 목록}.

    followup = 종합 요청 후속 재조사 횟수, reinvestigate = 평가 요청 재조사 횟수.
    """
    failing = failing or {}
    base = initial_state("q", "t-followup")
    evidence = [{"agent": p, "technology": t, "source_type": "web", "url": f"https://x/{p}/{t}/{i}", "claim": "c",
                 "source_id": f"{p}{t}{i}"} for p in PERSPECTIVES for t in ("mla", "itme") for i in range(2)]
    criteria = {name: {"passed": name not in failing, "target_agents": failing.get(name, []),
                       "reason": f"{name} 미달 사유", "rule": {"passed": True, "issues": []}} for name in CRITERIA}
    return {**base, "evidence": evidence,
            "node_status": {name: "done" for name in base["node_status"]},
            "perspectives": {p: {"sufficient": True, "missing": []} for p in PERSPECTIVES},
            "perspective_status": {p: "sufficient" for p in PERSPECTIVES},
            "synthesis": {"needs_source_agents": [], "evidence_gaps": []}, "report": "## SUMMARY\n본문",
            "retry_counts": {**base["retry_counts"], **(retry or {})},
            "followup_counts": {**base["followup_counts"], **(followup or {})},
            "reinvestigate_counts": {**base["reinvestigate_counts"], **(reinvestigate or {})},
            "eval_result": {"passed": not failing, "criteria": criteria}, **extra}


def test_policy_judge_blames_tech_after_sufficiency_retries_exhausted():
    state = evaluated_state({"groundedness": ["tech"]}, retry={"tech": RETRY_LIMITS["tech"]})
    decision = decide(state)
    assert decision.decision == "reinvestigate:tech" and decision.targets == ["tech"]
    assert decision.updates["reinvestigate_counts"] == {"tech": 1} and "retry_counts" not in decision.updates
    assert "followup_counts" not in decision.updates  # 종합 후속 재조사 한도는 쓰지 않는다
    assert decision.updates["node_status"]["tech"] == "running"
    assert {decision.updates["node_status"][n] for n in ("synthesis", "report", "quality_evaluator")} == {"pending"}


def test_judge_blames_tech_reinvestigates_even_when_sufficiency_retries_used_up(run_graph):
    limit = RETRY_LIMITS["tech"]
    llm = FakeLLM(insufficient={"tech": limit}, judge_fail={"groundedness": (1, "tech")})
    state = run_graph(llm)
    log = decisions(state)
    assert log[:limit + 1] == ["dispatch:tech,market,stakeholder,domain", *["dispatch:tech"] * limit]
    assert log[log.index("evaluate") + 1] == "reinvestigate:tech"
    assert state["retry_counts"]["tech"] == limit and state["reinvestigate_counts"]["tech"] == 1
    assert llm.runs["tech"] == limit + 2
    assert all(llm.runs[name] == 1 for name in ("market", "stakeholder", "domain"))
    assert log[-1] == "end:passed" and state["status"] == "completed"


def test_synthesis_request_after_retries_exhausted_runs_once_then_becomes_gap(run_graph):
    limit = RETRY_LIMITS["market"]
    llm = FakeLLM(insufficient={"market": limit}, needs_source=["market"], needs_source_runs=2)
    state = run_graph(llm)
    log = decisions(state)
    first, second = [i for i, d in enumerate(log) if d == "synthesis"]
    assert log[first + 1] == "dispatch:market"                 # 첫 요청: 후속 재조사
    assert log[first + 1:].count("dispatch:market") == 1
    assert log[second + 1] == "report"                         # 두 번째 요청: 공백 기록 후 보고서로
    assert state["retry_counts"]["market"] == limit and state["followup_counts"]["market"] == 1
    assert llm.runs["market"] == limit + 2 and llm.runs["synthesis"] == 2
    assert any(g["perspective"] == "market" and g["kind"] == "followup_exhausted" for g in state["gaps"])
    assert "**시장성** — 근거 부족:" in state["report"]
    assert log[-1].startswith("end:") and state["status"] == "completed_with_gaps"


def test_policy_second_synthesis_request_is_recorded_as_gap():
    state = evaluated_state(retry={"market": RETRY_LIMITS["market"]}, followup={"market": 1},
                            synthesis={"needs_source_agents": ["market"], "evidence_gaps": ["market: 시장 규모 근거"]})
    state["node_status"].update(report="pending", quality_evaluator="pending")
    decision = decide(state)
    assert decision.decision == "report"
    assert any(g["perspective"] == "market" and g["kind"] == "followup_exhausted" and "종합 단계" in g["detail"]
               for g in decision.updates["gaps"])


def test_source_unit_is_document_not_page():
    page3 = {"source_type": "paper", "doc_id": "deepseek_v2", "page": 3, "technology": "mla"}
    page16 = {**page3, "page": 16}
    assert source_unit(page3) == source_unit(page16)
    assert len(source_units([page3, page16])["mla"]) == 1
    assert source_unit({"source_type": "web", "url": "https://a.example/1"}) == "https://a.example/1"


def test_domain_is_not_subject_to_distinct_source_rule():
    one_doc = source_units([{"source_type": "paper", "doc_id": d, "page": p, "technology": t}
                            for t, d in (("mla", "deepseek_v2"), ("itme", "itme")) for p in (2, 9)])
    state = {"perspectives": {"tech": {"source_units": one_doc}, "domain": {"source_units": one_doc}}}
    assert {x["technology"] for x in evidence_shortfalls(state, "tech")} == {"mla", "itme"}  # 같은 문서 2쪽 = 출처 1개
    assert evidence_shortfalls(state, "domain") == []


def test_single_document_fit_is_downgraded_to_conditional(run_graph):
    evidence = [{"source_id": "p3", "doc_id": "itme"}, {"source_id": "p9", "doc_id": "itme"},
                {"source_id": "b1", "doc_id": "infinigen"}]
    out = downgrade_single_document([{"verdict": "적합", "explanation": "e", "cited_ids": ["p3", "p9"]},
                                     {"verdict": "적합", "explanation": "e", "cited_ids": ["p3", "b1"]}], evidence)
    assert [i["verdict"] for i in out] == ["조건부", "적합"]
    # Fake 도메인은 짝수 항목을 '적합'으로 내지만 모두 원문 1편 인용이라 하향된다
    items = run_graph(FakeLLM())["perspectives"]["domain"]["items"]
    assert not [i for i in items if i["verdict"] == "적합"]
    assert sum(SINGLE_DOC_NOTE in i["explanation"] for i in items) == 8


def test_design_citation_is_number_evidence_only_in_summary_and_chapters_1_2(run_graph):
    state = run_graph(FakeLLM())
    assert check_groundedness(state)["passed"] is True

    def insert(heading: str, line: str) -> dict:
        assert heading in state["report"]
        return {**state, "report": state["report"].replace(heading, f"{heading}{line}\n", 1)}

    for heading in ("## SUMMARY\n", "## 1. 분석 배경\n", "## 2. 기술 선정\n"):
        assert check_groundedness(insert(heading, "128K 문맥 요청 1건의 KV cache는 40 GiB다 [D]."))["passed"], heading
    result = check_groundedness(insert("## 4. 관점별 평가\n", "MLA는 KV cache를 93.3% 줄인다 [D]."))
    assert result["passed"] is False
    assert any("93.3%" in issue and "[D]" in issue for issue in result["issues"])


def test_merge_evidence_replaces_same_agent_and_appends_agentless_seed():
    old = [{"source_id": "w1", "claim": "c", "url": "u1", "agent": "market", "attempt": 0},
           {"source_id": "w2", "claim": "c", "url": "u2", "agent": "market", "attempt": 0},
           {"source_id": "p1", "claim": "c", "page": 1, "agent": "domain", "attempt": 0}]
    merged = merge_evidence(old, [{"source_id": "w3", "claim": "c", "url": "u3", "agent": "market", "attempt": 1}])
    assert [(ev["source_id"], ev["attempt"]) for ev in merged] == [("p1", 0), ("w3", 1)]
    seeded = merge_evidence(merged, [{"source_id": "s1", "claim": "c", "url": "u9"}])
    assert [ev["source_id"] for ev in seeded] == ["p1", "w3", "s1"]


def test_reinvestigation_leaves_only_latest_attempt_evidence(run_graph):
    state = run_graph(FakeLLM(judge_fail={"bias_control": (1, "market")}))
    assert "reinvestigate:market" in decisions(state)
    attempts = {}
    for ev in state["evidence"]:
        attempts.setdefault(ev["agent"], set()).add(ev["attempt"])
    assert attempts["market"] == {1} and all(attempts[n] == {0} for n in ("tech", "stakeholder", "domain"))


FAILURE_SCENARIOS = {
    "항상 근거 부족(4관점)": dict(insufficient={n: INF for n in PERSPECTIVES}),
    "항상 판정 한쪽뿐": dict(one_sided={"market": INF, "stakeholder": INF}),
    "항상 Judge 미달(원인 report)": dict(judge_fail={c: (INF, "report") for c in CRITERIA}),
    "항상 Judge 미달(원인 관점)": dict(judge_fail={"groundedness": (INF, "tech"), "bias_control": (INF, "market"),
                                              "coverage": (INF, "domain"), "neutrality": (INF, "stakeholder")}),
    "항상 금지 표현(규칙 미달)": dict(banned_report=INF),
    **{f"항상 예외({n})": dict(raise_on={n: INF}) for n in (*PERSPECTIVES, "synthesis", "judge")},
    "항상 예외(4관점 동시)": dict(raise_on={n: INF for n in PERSPECTIVES}),
}


@pytest.mark.parametrize("name", list(FAILURE_SCENARIOS))
def test_failure_injection_ends_within_max_steps_with_report(run_graph, name):
    state = run_graph(FakeLLM(**FAILURE_SCENARIOS[name]))
    log = decisions(state)
    assert log[-1].startswith("end:") and log[-1] != "end:hard_limit"
    assert state["step_count"] <= Policy().max_steps, (name, state["step_count"])
    assert state["report"].strip() and "## REFERENCE" in state["report"]
    assert state["status"] in ("completed", "completed_with_gaps", "unverified")


def test_report_agent_always_failing_ends_within_max_steps_with_gap(run_graph):
    """보고서 에이전트가 항상 실패해도 상한 안에서 끝나고 공백을 남긴다."""
    state = run_graph(FakeLLM(raise_on={"report": INF}))
    assert decisions(state)[-1] == "end:report_failed" and state["step_count"] <= Policy().max_steps
    assert any(g["perspective"] == "report" for g in state["gaps"]) and state["status"] == "unverified"


def test_report_only_policy_never_reinvestigates():
    # --report-only는 새 검색을 하지 않는다: 평가가 관점을 지목해도 재조사 대신 판정 한계를 적는 재작성으로 간다
    policy = Policy(retry_limits={**RETRY_LIMITS, **{name: 0 for name in PERSPECTIVES}, "synthesis": 0},
                    followup_limits={name: 0 for name in PERSPECTIVES},
                    reinvestigate_limits={name: 0 for name in PERSPECTIVES})
    decision = decide(evaluated_state({"bias_control": ["market"]}), policy)
    assert decision.decision == "rewrite:report"
    assert decision.updates["feedback"]["report"]["verdict_limits"][0]["section"] == "4.2"


def test_one_sided_gap_only_when_counter_direction_was_actually_searched():
    # 출처 부족으로 재시도를 다 쓴 뒤 마지막 시도에서 처음 한쪽 판정이 나오면 반대 방향은 탐색하지 않았다.
    # 이때는 "재검색했으나 없음"(one_sided)이 아니라 일반 근거 부족으로 남겨야 한다.
    one_sided = {"technologies": {t: {"market_size_growth_verdict": "긍정", "adoption_verdict": "긍정"}
                                  for t in ("mla", "itme")}, "sufficient": True, "missing": [],
                 "source_units": {"mla": ["a", "b"], "itme": ["c", "d"]}}
    base = evaluated_state(retry={"market": RETRY_LIMITS["market"]})
    base["node_status"].update(synthesis="pending", report="pending", quality_evaluator="pending")
    base["perspectives"] = {**base["perspectives"], "market": one_sided}
    not_searched = decide({**base, "feedback": {"market": {"queries_by_tech": {"mla": ["additional independent sources analysis"]}}}})
    kinds = {(g["kind"], g["technology"]) for g in not_searched.updates["gaps"] if g["perspective"] == "market"}
    assert ("one_sided", "mla") not in kinds and ("insufficient", "mla") in kinds
    searched = decide({**base, "feedback": {"market": {"queries_by_tech": {
        "mla": ["limitations risks concerns criticism"], "itme": ["limitations risks concerns criticism"]}}}})
    kinds = {(g["kind"], g["technology"]) for g in searched.updates["gaps"] if g["perspective"] == "market"}
    assert {("one_sided", "mla"), ("one_sided", "itme")} <= kinds
