"""후속 재조사 한도가 소진된 관점 원인 평가 미달의 경로(P0-3).

재조사할 수 없는 관점 원인 이슈는 원문 그대로 보고서 재작성 지시에 들어가고, 보고서는 해당 판정표 바로 아래에
'한계:' 문장을 쓴다. 그 재작성 후에도 같은 항목이 같은 사유로 미달이면 더 재작성하지 않고 end:unverified로 끝낸다.
"""
from __future__ import annotations

from fakes import INF, FakeLLM
from kv_eval.agents.report import verdict_limit_note
from kv_eval.config import RETRY_LIMITS
from kv_eval.evaluation.quality import CRITERIA
from kv_eval.observability import read_decisions
from kv_eval.supervisor.policy import decide
from test_followup_and_scope import evaluated_state


def decisions(state: dict) -> list[dict]:
    return [d for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]


def narrative_after_table(report: str, section: str) -> str:
    """'### 4.x' 절에서 판정표 다음 첫 서술 줄."""
    body = report.split(f"### {section}", 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
    return next(line for line in body.splitlines()[1:] if line.strip() and not line.lstrip().startswith("|"))


# ---- decide 직접 호출 ------------------------------------------------------------------------------
def test_exhausted_perspective_issue_is_carried_into_rewrite_with_original_text():
    state = evaluated_state({"groundedness": ["tech"]}, followup={"tech": 1})
    decision = decide(state)
    assert decision.decision == "rewrite:report"
    feedback = decision.updates["feedback"]["report"]
    assert "[4.1 판정 한계 명시] groundedness: groundedness 미달 사유" in feedback["issues"]
    assert feedback["verdict_limits"] == [{"perspective": "tech", "section": "4.1", "criterion": "groundedness",
                                           "issue": "groundedness: groundedness 미달 사유"}]
    assert feedback["carried_keys"] == ["groundedness|tech|judge"]
    assert any(g["perspective"] == "tech" and g["kind"] == "evaluation" and g["criterion"] == "groundedness"
               for g in decision.updates["gaps"])


def test_same_item_same_reason_after_carried_rewrite_ends_unverified():
    first = decide(evaluated_state({"groundedness": ["tech"]}, followup={"tech": 1}))
    # 재작성 1회 후 같은 항목·같은 원인으로 다시 미달(Judge 사유 문장은 달라도 같은 사유로 본다)
    again = evaluated_state({"groundedness": ["tech"]}, retry={"report": 1}, followup={"tech": 1},
                            feedback={"report": first.updates["feedback"]["report"]},
                            gaps=first.updates["gaps"])
    again["eval_result"]["criteria"]["groundedness"]["reason"] = "다른 문장으로 쓴 같은 지적"
    decision = decide(again)
    assert decision.decision == "end:unverified" and decision.updates["status"] == "unverified"
    assert "재작성 중단" in decision.reason
    assert RETRY_LIMITS["report"] > 1  # 보고서 재시도 한도가 남아 있어도 멈춘다
    assert not decision.updates.get("gaps")  # 같은 항목 공백은 사유 문장이 달라도 한 번만 기록


def test_new_carried_item_after_rewrite_still_gets_one_rewrite():
    first = decide(evaluated_state({"groundedness": ["tech"]}, followup={"tech": 1}))
    other = evaluated_state({"groundedness": ["tech"], "bias_control": ["market"]},
                            retry={"report": 1}, followup={"tech": 1, "market": 1},
                            feedback={"report": first.updates["feedback"]["report"]})
    decision = decide(other)
    assert decision.decision == "rewrite:report"
    assert {limit["section"] for limit in decision.updates["feedback"]["report"]["verdict_limits"]} == {"4.1", "4.2"}


def test_verdict_limit_note_targets_the_field_under_the_table():
    note = verdict_limit_note([{"perspective": "market", "section": "4.2", "criterion": "bias_control",
                                "issue": "market/mla: 판정이 긍정 한쪽뿐(반대 방향 근거 미탐색)"}])
    assert "'한계: '" in note and "Do not change or contradict the table verdict" in note
    assert "- 4.2 (perspective_market): market/mla: 판정이 긍정 한쪽뿐" in note
    assert verdict_limit_note([]) == ""


# ---- 그래프 실행 ----------------------------------------------------------------------------------
def test_judge_blaming_tech_forever_rewrites_once_with_limit_then_stops(run_graph):
    # 실제 Judge처럼 같은 지적을 실행마다 다른 문장으로 쓴다(이전에는 매번 새 공백으로 보여 재작성 2회를 모두 소모했다)
    llm = FakeLLM(judge_fail={"groundedness": (INF, "tech")}, judge_reason_varies=True)
    state = run_graph(llm)
    log = decisions(state)
    assert [d["decision"] for d in log] == [
        "dispatch:tech,market,stakeholder,domain", "synthesis", "report", "evaluate",
        "reinvestigate:tech", "synthesis", "report", "evaluate",
        "rewrite:report", "evaluate", "end:unverified"]
    assert "재작성 중단" in log[-1]["reason"]
    assert state["retry_counts"]["report"] == 1 < RETRY_LIMITS["report"]
    feedback = state["feedback"]["report"]
    assert any(i.startswith("[4.1 판정 한계 명시] groundedness: groundedness 판정") for i in feedback["issues"])
    assert "VERDICT LIMITATIONS" in llm.prompts["report"][-1] and "- 4.1 (perspective_trl):" in llm.prompts["report"][-1]
    assert narrative_after_table(state["report"], "4.1").startswith("한계:")
    assert sum(g["perspective"] == "tech" and g["kind"] == "evaluation" for g in state["gaps"]) == 1
    assert state["status"] == "unverified"


def test_rule_issue_of_exhausted_perspective_is_stated_under_table_and_passes(run_graph):
    # 시장 판정이 계속 긍정 한쪽뿐 → 후속 재조사 1회 후에도 같음 → 재작성에서 4.2 표 아래 한계 명시 → 공백 인정으로 통과
    llm = FakeLLM(one_sided={"market": INF})
    state = run_graph(llm)
    log = [d["decision"] for d in decisions(state)]
    assert log.count("reinvestigate:market") == 1 and log.count("rewrite:report") == 1
    assert any(i.startswith("[4.2 판정 한계 명시] market/mla: ") and "한쪽뿐" in i for i in state["feedback"]["report"]["issues"])
    assert narrative_after_table(state["report"], "4.2").startswith("한계:")
    assert narrative_after_table(state["report"], "4.1").startswith("한계:") is False
    assert log[-1] == "end:passed" and state["status"] == "completed_with_gaps"


def test_report_only_issues_keep_normal_rewrite_budget(run_graph):
    """원인이 보고서 서술뿐인 미달은 이 규칙 대상이 아니다(재작성 한도까지 고칠 기회를 준다)."""
    state = run_graph(FakeLLM(judge_fail={c: (INF, "report") for c in CRITERIA}, judge_reason_varies=True))
    log = [d["decision"] for d in decisions(state)]
    assert log.count("rewrite:report") == RETRY_LIMITS["report"] and log[-1] == "end:unverified"
    assert "verdict_limits" in state["feedback"]["report"] and not state["feedback"]["report"]["verdict_limits"]
