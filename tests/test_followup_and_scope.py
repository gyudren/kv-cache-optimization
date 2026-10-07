"""후속 재조사 한도·출처 단위·[D] 범위·근거 교체·실패 주입 종료를 검증한다(P0-2).

그래프는 Fake로 실제 실행하거나(run_graph), policy.decide를 직접 호출한다.
"""
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
                    followup: dict | None = None, **extra) -> dict:
    """4관점 충분 → 종합·보고서·평가까지 끝난 State. failing = {평가 항목: 원인 에이전트 목록}."""
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
            "eval_result": {"passed": not failing, "criteria": criteria}, **extra}


# ---- (a) Judge가 groundedness 원인을 tech로 지목 → tech 후속 재조사 ------------------------------
def test_policy_judge_blames_tech_after_sufficiency_retries_exhausted():
    state = evaluated_state({"groundedness": ["tech"]}, retry={"tech": RETRY_LIMITS["tech"]})
    decision = decide(state)
    assert decision.decision == "reinvestigate:tech" and decision.targets == ["tech"]
    assert decision.updates["followup_counts"] == {"tech": 1} and "retry_counts" not in decision.updates
    assert decision.updates["node_status"]["tech"] == "running"
    assert {decision.updates["node_status"][n] for n in ("synthesis", "report", "quality_evaluator")} == {"pending"}


def test_judge_blames_tech_reinvestigates_even_when_sufficiency_retries_used_up(run_graph):
    limit = RETRY_LIMITS["tech"]
    llm = FakeLLM(insufficient={"tech": limit}, judge_fail={"groundedness": (1, "tech")})
    state = run_graph(llm)
    log = decisions(state)
    assert log[:limit + 1] == ["dispatch:tech,market,stakeholder,domain", *["dispatch:tech"] * limit]
    assert log[log.index("evaluate") + 1] == "reinvestigate:tech"
    assert state["retry_counts"]["tech"] == limit and state["followup_counts"]["tech"] == 1
    assert llm.runs["tech"] == limit + 2
    assert all(llm.runs[name] == 1 for name in ("market", "stakeholder", "domain"))
    assert log[-1] == "end:passed" and state["status"] == "completed"


# ---- (b) 종합이 market 추가 근거 요청, market 충분성 재시도 소진 -----------------------------------
def test_synthesis_request_after_retries_exhausted_runs_once_then_becomes_gap(run_graph):
    limit = RETRY_LIMITS["market"]
    llm = FakeLLM(insufficient={"market": limit}, needs_source=["market"], needs_source_runs=2)
    state = run_graph(llm)
    log = decisions(state)
    first, second = [i for i, d in enumerate(log) if d == "synthesis"]
    assert log[first + 1] == "dispatch:market"                 # 1회차 요청: 후속 재조사
    assert log[first + 1:].count("dispatch:market") == 1
    assert log[second + 1] == "report"                         # 2회차 요청: 공백 기록 후 보고서로 진행
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


# ---- (c) 고유 출처는 문서 단위, domain은 출처 수 규칙 대신 단일 문헌 하향 ------------------------------
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
    # 실제 그래프에서도 원문 1편만 인용한 '적합'은 남지 않는다(Fake 도메인 에이전트는 짝수 항목을 '적합'으로 판정)
    items = run_graph(FakeLLM())["perspectives"]["domain"]["items"]
    assert not [i for i in items if i["verdict"] == "적합"]
    assert sum(SINGLE_DOC_NOTE in i["explanation"] for i in items) == 8


# ---- (d) [D]는 SUMMARY·1·2장에서만 수치 근거 ------------------------------------------------------
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


# ---- (e) merge_evidence: 같은 agent의 새 쓰기는 이전 근거를 교체, agent 없는 seed는 이어 붙임 -----------
def test_merge_evidence_replaces_same_agent_and_appends_agentless_seed():
    old = [{"source_id": "w1", "claim": "c", "url": "u1", "agent": "market", "attempt": 0},
           {"source_id": "w2", "claim": "c", "url": "u2", "agent": "market", "attempt": 0},
           {"source_id": "p1", "claim": "c", "page": 1, "agent": "domain", "attempt": 0}]
    merged = merge_evidence(old, [{"source_id": "w3", "claim": "c", "url": "u3", "agent": "market", "attempt": 1}])
    assert [(ev["source_id"], ev["attempt"]) for ev in merged] == [("p1", 0), ("w3", 1)]
    seeded = merge_evidence(merged, [{"source_id": "s1", "claim": "c", "url": "u9"}])
    assert [ev["source_id"] for ev in seeded] == ["p1", "w3", "s1"]


def test_reinvestigation_leaves_only_latest_attempt_evidence(run_graph):
    state = run_graph(FakeLLM(one_sided={"market": 1}))
    assert "reinvestigate:market" in decisions(state)
    attempts = {}
    for ev in state["evidence"]:
        attempts.setdefault(ev["agent"], set()).add(ev["attempt"])
    assert attempts["market"] == {1} and all(attempts[n] == {0} for n in ("tech", "stakeholder", "domain"))


# ---- (f) 실패 주입 시나리오는 모두 MAX_STEPS 안에서 end:*로 끝나고 보고서가 남는다 ------------------
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
    """보고서 에이전트 자체가 항상 실패하면 보고서는 만들 수 없다. 이때도 상한 안에서 끝나고 공백을 남긴다."""
    state = run_graph(FakeLLM(raise_on={"report": INF}))
    assert decisions(state)[-1] == "end:report_failed" and state["step_count"] <= Policy().max_steps
    assert any(g["perspective"] == "report" for g in state["gaps"]) and state["status"] == "unverified"
