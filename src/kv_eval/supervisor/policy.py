"""Supervisor 결정 정책: State 제어 필드만 읽고 다음 실행 노드를 정한다(순수 함수).

입력: perspective_status · node_status · retry_counts · eval_result · step_count (+ 결과의 sufficient/missing)
출력: Decision(targets, decision, reason, updates)

고정된 실행 순서는 없다. 매 진입마다 아래 우선순위로 "지금 State에서 해야 할 일"을 고른다.
1) 근거가 없거나 부족하거나 실패한 관점 → 해당 관점만 (재)할당 (재시도 상한 안에서)
2) 모든 관점이 결론 상태 → 종합 (종합이 특정 관점의 추가 근거를 요구하면 그 관점만 재조사)
3) 종합 완료 → 보고서 (보고서는 품질 평가 노드를 거쳐 돌아온다)
4) 평가 결과 → 통과면 종료, 미달이면 원인별 경로(관점 재조사 / 보고서 재작성)
상한(MAX_STEPS·재시도 한도)은 안전장치다. 도달하면 근거 공백을 기록하고 보고서까지 만든 뒤 종료한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from ..config import FINALIZE_STEPS, MAX_STEPS, PERSPECTIVES, RETRY_LIMITS
from ..evaluation.quality import REINVESTIGATE_CRITERIA, REWRITE_CRITERIA, evidence_shortfalls
from ..state import perspective

END_NODE = "__end__"
DOWNSTREAM = ("synthesis", "report", "quality_evaluator")
LABEL = {"tech": "기술 성숙도(TRL)", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용성"}


@dataclass(frozen=True)
class Policy:
    max_steps: int = MAX_STEPS
    retry_limits: Mapping[str, int] = field(default_factory=lambda: MappingProxyType(dict(RETRY_LIMITS)))

    def limit(self, name: str) -> int:
        return self.retry_limits.get(name, 0)


@dataclass
class Decision:
    targets: list[str]
    decision: str
    reason: str
    updates: dict[str, Any] = field(default_factory=dict)


def classify(state: dict, name: str) -> str:
    """관점 1개의 근거 충분도. excluded는 한 번 정해지면 유지한다(재시도 상한 소진)."""
    if (state.get("perspective_status") or {}).get(name) == "excluded":
        return "excluded"
    status = (state.get("node_status") or {}).get(name, "pending")
    if status == "failed":
        return "failed"
    if status != "done":
        return "pending"  # 미실행(또는 중단 후 재개로 running이 남은 경우)
    # 에이전트의 자기 보고(sufficient)만 믿지 않고 Supervisor가 출처 수를 결정적으로 다시 확인한다.
    if not perspective(state, name).get("sufficient") or evidence_shortfalls(state, name):
        return "insufficient"
    return "sufficient"


def _feedback(missing: list[str], *, search: bool = True) -> dict:
    missing = [str(m) for m in missing if str(m).strip()][:6]
    return {"missing": missing, "rewritten_queries": missing if search else []}


# 품질 평가(편향·커버리지) 미달 사유 → 재검색 질의 힌트. 사유 문장을 그대로 검색하면 결과가 나오지 않는다.
EVAL_QUERY_HINTS = (
    ("긍정 한쪽뿐", "limitations risks concerns criticism"),
    ("우려 한쪽뿐", "adoption benefits positive outlook support"),
    ("단일 발행처", "independent analysis industry report"),
    ("고유 출처", "additional independent sources analysis"),
)


GENERIC_EVAL_HINT = "independent sources limitations adoption evidence"


def _eval_feedback(issues: list[str]) -> dict:
    """평가 미달 사유는 missing으로 그대로 넘기고, 재검색 질의는 반대 방향·독립 출처 힌트로 바꾼다."""
    queries = [hint for issue in issues for key, hint in EVAL_QUERY_HINTS if key in issue]
    # 규칙 사유가 없는(LLM Judge만 미달) 경우에도 사유 문장으로 검색하지 않고 일반 힌트를 쓴다.
    return {"missing": issues[:6], "rewritten_queries": list(dict.fromkeys(queries or [GENERIC_EVAL_HINT]))[:3]}


class _Builder:
    """한 번의 결정에서 State 갱신분을 모은다."""

    def __init__(self, state: dict, policy: Policy):
        self.state, self.policy = state, policy
        self.retry = dict(state.get("retry_counts") or {})
        self.updates: dict[str, Any] = {"retry_counts": {}, "feedback": {}, "node_status": {},
                                        "perspective_status": {}, "gaps": []}

    def can_retry(self, name: str) -> bool:
        return self.retry.get(name, 0) < self.policy.limit(name)

    def bump(self, name: str, feedback: dict | None = None) -> None:
        self.retry[name] = self.retry.get(name, 0) + 1
        self.updates["retry_counts"][name] = self.retry[name]
        if feedback is not None:
            self.updates["feedback"][name] = feedback

    def run(self, *names: str) -> None:
        for name in names:
            self.updates["node_status"][name] = "running"

    def invalidate_downstream(self) -> None:
        # 관점 결과가 바뀌면 종합·보고서·평가는 다시 만들어야 한다.
        for name in DOWNSTREAM:
            if (self.state.get("node_status") or {}).get(name, "pending") != "pending":
                self.updates["node_status"].setdefault(name, "pending")

    def gap(self, text: str) -> None:
        if text not in (self.state.get("gaps") or []) and text not in self.updates["gaps"]:
            self.updates["gaps"].append(text)

    def done(self, targets: list[str], decision: str, reason: str, **extra) -> Decision:
        updates = {k: v for k, v in self.updates.items() if v}
        updates.update(extra)
        return Decision(targets, decision, reason, updates)

    def end(self, decision: str, reason: str, verified: bool) -> Decision:
        # completed = 평가 통과·공백 없음 / completed_with_gaps = 평가 통과·근거 공백 명시 / unverified = 평가 미통과
        has_gaps = bool(self.state.get("gaps") or self.updates["gaps"])
        status = "unverified" if not verified else "completed_with_gaps" if has_gaps else "completed"
        return self.done([END_NODE], decision, reason, status=status)


def _perspective_step(b: _Builder, finalize: bool) -> Decision | None:
    state, policy = b.state, b.policy
    to_run, reasons = [], []
    for name in PERSPECTIVES:
        status = classify(state, name)
        b.updates["perspective_status"][name] = status
        if status in ("sufficient", "excluded"):
            continue
        if status == "pending" and not finalize:
            to_run.append(name)
            reasons.append(f"{name}: 미수집")
            continue
        missing = [*evidence_shortfalls(state, name), *perspective(state, name).get("missing", [])]
        error = (state.get("last_error") or {}).get(name, "")
        if not finalize and b.can_retry(name):
            if status == "failed":
                b.bump(name, {**_feedback(state.get("feedback", {}).get(name, {}).get("missing", []), search=True),
                              "last_error": error})
                reasons.append(f"{name}: 실행 실패 재시도 {b.retry[name]}/{policy.limit(name)} ({error[:80]})")
            else:
                b.bump(name, _feedback(missing))
                reasons.append(f"{name}: 근거 부족 재조사 {b.retry[name]}/{policy.limit(name)} "
                               f"({'; '.join(map(str, missing[:2]))[:120]})")
            to_run.append(name)
            continue
        # 상한 소진 또는 단계 상한: 제외하고 근거 공백으로 명시한다.
        b.updates["perspective_status"][name] = "excluded"
        if finalize:
            b.gap(f"{name}: 단계 상한(MAX_STEPS) 도달로 {LABEL[name]} 추가 조사 중단 — 근거 부족")
        elif status == "failed":
            b.gap(f"{name}: {LABEL[name]} 에이전트 실행 실패로 제외({error[:120]}) — 근거 부족")
        else:
            b.gap(f"{name}: {LABEL[name]} 재조사 {policy.limit(name)}회 후에도 근거 부족 — "
                  + "; ".join(map(str, missing[:3]))[:240])
        reasons.append(f"{name}: 제외(근거 공백 기록)")
    if not to_run:
        return None
    b.run(*to_run)
    b.invalidate_downstream()
    for name in to_run:
        b.updates["perspective_status"][name] = "pending"
    return b.done(to_run, "dispatch:" + ",".join(to_run), "; ".join(reasons))


def _synthesis_step(b: _Builder, finalize: bool) -> Decision | None:
    state = b.state
    status = (state.get("node_status") or {}).get("synthesis", "pending")
    if status in ("pending", "running"):
        b.run("synthesis")
        return b.done(["synthesis"], "synthesis", "4관점이 결론 상태(충분 또는 근거 공백 기록) → 종합")
    if status == "failed":
        if not finalize and b.can_retry("synthesis"):
            b.bump("synthesis")
            b.run("synthesis")
            return b.done(["synthesis"], "retry:synthesis",
                          f"종합 실행 실패 재시도 ({(state.get('last_error') or {}).get('synthesis', '')[:100]})")
        b.gap("synthesis: 종합 에이전트 실행 실패 — 관점 간 일치·상충 정리 근거 부족")
        b.updates["node_status"]["synthesis"] = "skipped"
        return None
    if status != "done" or finalize:
        return None
    synthesis = state.get("synthesis") or {}
    # 종합이 특정 관점의 추가 근거를 요구하면 그 관점만 재조사한다(한도 안에서).
    requested = [n for n in synthesis.get("needs_source_agents", []) if n in PERSPECTIVES]
    rerun = [n for n in requested if classify(state, n) != "excluded" and b.can_retry(n)]
    for name in requested:
        if name not in rerun:
            b.gap(f"{name}: 종합 단계에서 추가 근거 요청, 재조사 한도 소진 — 근거 부족")
    if rerun:
        gaps = synthesis.get("evidence_gaps", [])
        for name in rerun:
            b.bump(name, _feedback([g for g in gaps if name in str(g)] or gaps))
        b.run(*rerun)
        b.invalidate_downstream()
        return b.done(rerun, "dispatch:" + ",".join(rerun), "종합이 추가 근거를 요청한 관점만 재조사")
    if synthesis.get("needs_revision") and b.can_retry("synthesis"):
        b.bump("synthesis", {"issues": synthesis.get("evidence_gaps", [])})
        b.run("synthesis")
        return b.done(["synthesis"], "retry:synthesis", "종합 표현 보완 필요(needs_revision)")
    return None


def _report_and_eval_step(b: _Builder, finalize: bool) -> Decision:
    state = b.state
    node_status = state.get("node_status") or {}
    report_status = node_status.get("report", "pending")
    if report_status in ("pending", "running"):
        b.run("report")
        b.updates["node_status"]["quality_evaluator"] = "pending"
        return b.done(["report"], "report", "종합 완료 → 보고서 작성")
    if report_status == "failed":
        if not finalize and b.can_retry("report"):
            b.bump("report")
            b.run("report")
            return b.done(["report"], "retry:report",
                          f"보고서 실행 실패 재시도 ({(state.get('last_error') or {}).get('report', '')[:100]})")
        b.gap("report: 보고서 에이전트 실행 실패 — 보고서 미생성")
        return b.end("end:report_failed", "보고서 재시도 한도 소진", verified=False)

    if not (state.get("report") or "").strip():
        # done인데 본문이 비었으면 평가하지 않고 보고서 실패와 같이 처리한다(빈 보고서에 Judge를 부르지 않음).
        if not finalize and b.can_retry("report"):
            b.bump("report")
            b.run("report")
            return b.done(["report"], "retry:report", "보고서 본문 없음 → 재작성")
        b.gap("report: 보고서 본문 미생성")
        return b.end("end:report_failed", "보고서 본문 없음, 재시도 한도 소진", verified=False)

    eval_status = node_status.get("quality_evaluator", "pending")
    if eval_status in ("pending", "running"):
        b.run("quality_evaluator")
        return b.done(["quality_evaluator"], "evaluate",
                      f"보고서 완료(시도 {b.retry.get('report', 0)}) → 품질 평가 4항목 판정")
    if eval_status == "failed":
        if not finalize and b.can_retry("quality_evaluator"):
            b.bump("quality_evaluator")
            b.run("quality_evaluator")
            return b.done(["quality_evaluator"], "retry:quality_evaluator", "품질 평가 실행 실패 재시도")
        b.gap("quality_evaluator: 품질 평가 실행 실패 — 보고서 미검증")
        return b.end("end:unverified", "품질 평가 재시도 한도 소진", verified=False)

    result = state.get("eval_result") or {}
    if result.get("passed"):
        return b.end("end:passed", "품질 평가 4항목 통과", verified=True)
    criteria = result.get("criteria", {})
    failed = [n for n, c in criteria.items() if not c.get("passed")]
    if finalize:
        return b.end("end:step_limit", f"단계 상한 도달 — 품질 평가 미달 항목 {failed} 남김", verified=False)

    # (a) 편향 통제·관점 커버리지 미달 → 원인 관점만 재조사
    issues_by_agent: dict[str, list[str]] = {}
    for name in REINVESTIGATE_CRITERIA:
        c = criteria.get(name, {})
        if c.get("passed"):
            continue
        for agent in c.get("target_agents", []):
            if agent in PERSPECTIVES:
                own = [i for i in c.get("rule", {}).get("issues", []) if i.startswith(agent)]
                issues_by_agent.setdefault(agent, []).extend(own or [f"{name}: {c.get('reason', '')[:200]}"])
    rerun = [a for a in issues_by_agent if classify(state, a) != "excluded" and b.can_retry(a)]
    for agent, issues in issues_by_agent.items():
        if agent in rerun:
            continue
        for issue in issues:  # 재조사 불가: 근거 공백으로 명시하고 보고서에 드러낸다
            b.gap(issue if issue.startswith(f"{agent}/") or issue.startswith(f"{agent}:")
                  else f"{agent}: 품질 평가 미달, 재조사 한도 소진 — {issue[:200]}")
    if rerun:
        for agent in rerun:
            b.bump(agent, _eval_feedback(issues_by_agent[agent]))
        b.run(*rerun)
        b.invalidate_downstream()
        return b.done(rerun, "reinvestigate:" + ",".join(rerun),
                      "품질 평가 편향·커버리지 미달 → 원인 관점만 재조사: "
                      + "; ".join(i for a in rerun for i in issues_by_agent[a])[:300])

    # (b) Groundedness·중립성 미달(또는 새 근거 공백 반영 필요) → 보고서 재작성
    rewrite_issues = []
    for name in (*REWRITE_CRITERIA, *REINVESTIGATE_CRITERIA):
        c = criteria.get(name, {})
        if not c.get("passed") and (name in REWRITE_CRITERIA or "report" in c.get("target_agents", [])):
            rewrite_issues.extend(c.get("rule", {}).get("issues", []) or [c.get("reason", "")])
    if b.updates["gaps"]:
        rewrite_issues.append("Supervisor가 새로 기록한 근거 공백을 7장 한계점에 근거 부족으로 명시할 것")
    if rewrite_issues and b.can_retry("report"):
        b.bump("report", {"issues": [i for i in rewrite_issues if i][:12]})
        b.run("report")
        b.updates["node_status"]["quality_evaluator"] = "pending"
        return b.done(["report"], "rewrite:report",
                      f"품질 평가 미달({','.join(failed)}) → 보고서 재작성: " + "; ".join(rewrite_issues)[:300])
    return b.end("end:unverified", f"재작업 한도 소진 — 품질 평가 미달 항목 {failed} 남김", verified=False)


def decide(state: dict, policy: Policy | None = None) -> Decision:
    policy = policy or Policy()
    step = int(state.get("step_count", 0)) + 1
    b = _Builder(state, policy)
    if step > policy.max_steps + FINALIZE_STEPS:
        return b.end("end:hard_limit", f"Supervisor 진입 {step}회 — 최종 상한 초과로 즉시 종료", verified=False)
    finalize = step > policy.max_steps
    for stage in (_perspective_step, _synthesis_step):
        decision = stage(b, finalize)
        if decision is not None:
            if finalize:
                decision.reason = f"[단계 상한 {policy.max_steps} 도달: 마무리 모드] " + decision.reason
            return decision
    decision = _report_and_eval_step(b, finalize)
    if finalize and not decision.decision.startswith("end"):
        decision.reason = f"[단계 상한 {policy.max_steps} 도달: 마무리 모드] " + decision.reason
    return decision
