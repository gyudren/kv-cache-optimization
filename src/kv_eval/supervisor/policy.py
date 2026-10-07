"""Supervisor 결정 정책: State 제어 필드만 읽고 다음 실행 노드를 정한다(순수 함수).

입력: perspective_status · node_status · retry_counts · eval_result · step_count (+ 결과의 sufficient/missing)
출력: Decision(targets, decision, reason, updates)

고정된 실행 순서는 없다. 매 진입마다 아래 우선순위로 "지금 State에서 해야 할 일"을 고른다.
1) 근거가 없거나 부족하거나 실패한 관점 → 해당 관점만 (재)할당 (재시도 상한 안에서)
2) 모든 관점이 결론 상태 → 종합 (종합이 특정 관점의 추가 근거를 요구하면 그 관점만 재조사)
3) 종합 완료 → 보고서 (보고서는 품질 평가 노드를 거쳐 돌아온다)
4) 평가 결과 → 통과면 종료, 미달이면 원인별 경로(원인이 관점이면 그 관점 재조사 / report면 보고서 재작성).
   원인이 관점인데 후속 재조사 한도가 소진됐으면 그 이슈 원문을 보고서 재작성 지시에 넣어 해당 판정표 아래에
   '한계:'로 명시하게 하고, 그 재작성 후에도 같은 항목이 같은 사유로 미달이면 재작성을 멈추고 미검증으로 끝낸다.
상한(MAX_STEPS·재시도 한도)은 안전장치다. 도달하면 근거 공백을 기록하고 보고서까지 만든 뒤 종료한다.
재시도 한도는 둘로 나뉜다. 충분성 재조사(1)는 RETRY_LIMITS, 종합·평가가 요청한 후속 재조사(2·4)는
FOLLOWUP_LIMITS를 쓴다. 충분성 재조사가 한도를 다 써도 종합·평가가 지목한 관점은 따로 한 번 더 조사할 수 있다.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from ..config import FINALIZE_STEPS, FOLLOWUP_LIMITS, MAX_STEPS, PERSPECTIVES, RETRY_LIMITS, recursion_limit_for
from ..evaluation.quality import CRITERIA, SECTION_PERSPECTIVE, evidence_shortfalls
from ..state import followup_key, perspective
from ..tools import TECH_TOKENS, mentioned_techs

END_NODE = "__end__"
DOWNSTREAM = ("synthesis", "report", "quality_evaluator")
LABEL = {"tech": "기술 성숙도(TRL)", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용성"}
SECTION_OF = {name: number for number, name in SECTION_PERSPECTIVE.items()}  # 관점 → 판정표가 있는 보고서 절(4.1~4.4)
EVAL_GAP = "품질 평가 미달, 후속 조사 한도 소진"


@dataclass(frozen=True)
class _AgentIssue:
    """원인이 관점 에이전트인 평가 미달 1건. key는 '같은 항목·같은 사유'를 판정하는 비교 키다."""
    agent: str
    criterion: str
    text: str
    reason_key: str

    @property
    def key(self) -> str:
        return f"{self.criterion}|{self.agent}|{self.reason_key}"


@dataclass(frozen=True)
class Policy:
    max_steps: int = MAX_STEPS
    retry_limits: Mapping[str, int] = field(default_factory=lambda: MappingProxyType(dict(RETRY_LIMITS)))
    followup_limits: Mapping[str, int] = field(default_factory=lambda: MappingProxyType(dict(FOLLOWUP_LIMITS)))

    def limit(self, name: str) -> int:
        return self.retry_limits.get(name, 0)

    def followup_limit(self, name: str) -> int:
        return self.followup_limits.get(name, 0)

    @property
    def recursion_limit(self) -> int:
        return recursion_limit_for(self.max_steps)


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


# 사유 → 재검색 힌트. 사유 문장은 사람이 읽는 missing으로만 넘기고 검색어로 쓰지 않는다.
EVAL_QUERY_HINTS = (
    ("긍정 한쪽뿐", "limitations risks concerns criticism"),
    ("우려 한쪽뿐", "adoption benefits positive outlook support"),
    ("단일 발행처", "independent analysis industry report"),
    ("고유 출처", "additional independent sources analysis"),
    ("TRL", "production deployment commercial service availability"),
    ("상용", "production deployment commercial service availability"),
)
GENERIC_EVAL_HINT = "independent sources limitations adoption evidence"
SHORTFALL_HINT = "additional independent sources analysis"
TECHS = ("mla", "itme")


def _item_tech(text: str) -> list[str]:
    """`관점/기술:` 접두어가 있으면 그 기술, 없으면 본문에 언급된 기술(둘 다/없음이면 두 기술 모두)."""
    prefix = re.match(r"^\w+/(mla|itme):", str(text))
    if prefix:
        return [prefix.group(1)]
    techs = sorted(mentioned_techs(text))
    return techs if len(techs) == 1 else list(TECHS)


def _agent_missing_query(text: str) -> str:
    """에이전트가 보고한 부족 항목을 검색어로 다듬는다(접두어·기술명·괄호 주석 제거)."""
    text = re.sub(r"^\w+(/\w+)?:\s*", "", str(text))
    text = re.sub(r"\([^)]*\)", " ", text)
    for tokens in TECH_TOKENS.values():
        for token in tokens:
            text = re.sub(token, " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())[:100]


def _rework(items: list[tuple[list[str], str, str]]) -> dict:
    """재작업 지시를 구조화한다. items = [(대상 기술 목록, 사유, 검색 힌트)].

    - missing: 사람이 읽는 사유(프롬프트에 그대로 들어감)
    - queries_by_tech: 기술별 검색 힌트. 실제로 부족한 기술에만 붙고 다른 기술 이름은 섞이지 않는다.
    """
    missing, by_tech = [], {tech: [] for tech in TECHS}
    for techs, reason, query in items:
        if reason and reason not in missing:
            missing.append(reason)
        for tech in techs:
            if query and query not in by_tech[tech]:
                by_tech[tech].append(query)
    by_tech = {tech: queries[:3] for tech, queries in by_tech.items() if queries}
    return {"missing": missing[:8], "queries_by_tech": by_tech,
            "rewritten_queries": list(dict.fromkeys(q for qs in by_tech.values() for q in qs))}


def _sufficiency_feedback(shortfalls: list[dict], missing: list[str], optional: list[str] = ()) -> dict:
    """재작업 지시: 사유는 결정적 검사와 필수 결함, 검색 힌트는 에이전트가 확인하지 못한 세부 항목에서 만든다.

    필수 결함 문장("검증 가능한 출처 인용 없음")은 검색어가 되지 못하므로 사유로만 넘기고, 같은 기술의 선택 항목
    (예: "MLA 시장 규모 정량 근거")을 검색 힌트로 쓴다. 힌트가 하나도 없는 기술에는 일반 힌트를 붙인다.
    """
    items = [([x["technology"]], x["reason"], SHORTFALL_HINT) for x in shortfalls]
    items += [(_item_tech(m), str(m), "") for m in missing if str(m).strip()]
    items += [(_item_tech(o), "", _agent_missing_query(o)) for o in optional if str(o).strip()]
    hinted = {tech for techs, _, query in items if query for tech in techs}
    needy = {tech for techs, reason, _ in items if reason for tech in techs}
    items += [([tech], "", SHORTFALL_HINT) for tech in sorted(needy - hinted)]
    return _rework(items)


def _eval_feedback(issues: list[str]) -> dict:
    """평가 미달 사유는 missing으로 넘기고, 검색어는 사유 유형별 힌트로만 만든다(사유 문장으로 검색하지 않음)."""
    items = []
    for issue in issues:
        hints = [hint for key, hint in EVAL_QUERY_HINTS if key in issue] or [GENERIC_EVAL_HINT]
        items += [(_item_tech(issue) if re.match(r"^\w+/(mla|itme):", issue) else list(TECHS), issue, hint)
                  for hint in hints]
    return _rework(items)


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

    def can_followup(self, name: str) -> bool:
        return self.retry.get(followup_key(name), 0) < self.policy.followup_limit(name)

    def followup(self, names: list[str], feedback_by_name: dict[str, dict]) -> None:
        """종합·평가가 지목한 관점을 후속 재조사 한도로 다시 보낸다(제외됐던 관점도 다시 조사 대상이 된다)."""
        for name in names:
            key = followup_key(name)
            self.retry[key] = self.retry.get(key, 0) + 1
            self.updates["retry_counts"][key] = self.retry[key]
            self.updates["feedback"][name] = feedback_by_name[name]
            self.updates["perspective_status"][name] = "pending"
        self.run(*names)
        self.invalidate_downstream()

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

    def has_gap(self, prefix: str) -> bool:
        return any(g.startswith(prefix) for g in [*(self.state.get("gaps") or []), *self.updates["gaps"]])

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
        shortfalls = evidence_shortfalls(state, name)
        missing = [*(x["reason"] for x in shortfalls), *perspective(state, name).get("missing", [])]
        error = (state.get("last_error") or {}).get(name, "")
        if not finalize and b.can_retry(name):
            if status == "failed":
                # 실행 실패 재시도: 직전 지시를 그대로 다시 넘긴다(새 검색 사유는 없음).
                b.bump(name, {**state.get("feedback", {}).get(name, {}), "last_error": error})
                reasons.append(f"{name}: 실행 실패 재시도 {b.retry[name]}/{policy.limit(name)} ({error[:80]})")
            else:
                result = perspective(state, name)
                b.bump(name, _sufficiency_feedback(shortfalls, result.get("missing", []), result.get("missing_optional", [])))
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
    # 종합이 특정 관점의 추가 근거를 요구하면 그 관점만 재조사한다(후속 재조사 한도 안에서).
    requested = [n for n in synthesis.get("needs_source_agents", []) if n in PERSPECTIVES]
    rerun = [n for n in requested if b.can_followup(n)]
    gaps = synthesis.get("evidence_gaps", [])
    for name in requested:
        if name not in rerun:
            reason = next((str(g) for g in gaps if name in str(g)), "")
            b.gap(f"{name}: 종합 단계에서 추가 근거가 필요하다고 판단했으나 후속 조사 한도 소진"
                  + (f" — {reason[:160]}" if reason else ""))
    if rerun:
        b.followup(rerun, {name: _sufficiency_feedback([], [g for g in gaps if name in str(g)] or gaps)
                           for name in rerun})
        return b.done(rerun, "dispatch:" + ",".join(rerun), "종합이 추가 근거를 요청한 관점만 재조사(후속 재조사)")
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

    # (a) 미달 원인이 관점 에이전트인 항목(편향·커버리지, 또는 에이전트 판정을 옮긴 표의 근거·중립성 결함) → 그 관점만 재조사
    issues_by_agent: dict[str, list[_AgentIssue]] = {}
    for name in CRITERIA:
        c = criteria.get(name, {})
        if c.get("passed"):
            continue
        for agent in c.get("target_agents", []):
            if agent in PERSPECTIVES:
                own = [_AgentIssue(agent, name, i, i) for i in c.get("rule", {}).get("issues", []) if i.startswith(agent)]
                # Judge 사유는 실행마다 문장이 달라지므로 "같은 사유"는 (항목, 원인 관점)으로 본다.
                judged = _AgentIssue(agent, name, f"{name}: {c.get('reason', '')[:300]}", "judge")
                issues_by_agent.setdefault(agent, []).extend(own or [judged])
    rerun = [a for a in issues_by_agent if b.can_followup(a)]
    carried: list[_AgentIssue] = []  # 후속 한도 소진으로 재조사하지 못한 관점 원인 이슈 → 보고서에서 판정의 한계로 명시
    for agent, issues in issues_by_agent.items():
        if agent in rerun:
            continue
        for issue in issues:  # 재조사 불가: 근거 공백으로 명시하고 보고서에 드러낸다
            carried.append(issue)
            if issue.text.startswith((f"{agent}/", f"{agent}:")):
                b.gap(issue.text)
            elif not b.has_gap(f"{agent}: {EVAL_GAP} — {issue.criterion}:"):  # 같은 항목은 사유 문장이 달라도 한 번만
                b.gap(f"{agent}: {EVAL_GAP} — {issue.text[:200]}")
    if rerun:
        b.followup(rerun, {agent: _eval_feedback([i.text for i in issues_by_agent[agent]]) for agent in rerun})
        return b.done(rerun, "reinvestigate:" + ",".join(rerun),
                      "품질 평가 미달 원인 관점만 재조사: "
                      + "; ".join(i.text for a in rerun for i in issues_by_agent[a])[:300])

    # 재조사하지 못한 관점 원인 이슈로 이미 한 번 재작성했는데 같은 항목이 같은 사유로 또 미달이면 재작성을 멈춘다.
    # 판정의 한계를 명시한 보고서로도 해소되지 않는 결함이라 재작성을 반복해도 통과할 수 없다(무의미한 루프 방지).
    carried_keys = sorted({issue.key for issue in carried})
    previous = (state.get("feedback") or {}).get("report") or {}
    if carried_keys and set(carried_keys) <= set(previous.get("carried_keys", [])):
        return b.end("end:unverified",
                     "재조사할 수 없는 관점 원인 미달이 판정 한계를 명시한 재작성 후에도 같은 사유로 반복 → 재작성 중단: "
                     + "; ".join(i.text for i in carried)[:240], verified=False)

    # (b) 원인이 보고서 서술인 항목, 재조사하지 못한 관점 원인 항목(해당 표 아래 판정 한계 명시), 새 근거 공백 → 보고서 재작성
    rewrite_issues = []
    for name in CRITERIA:
        c = criteria.get(name, {})
        if not c.get("passed") and "report" in c.get("target_agents", []):
            rewrite_issues.extend(c.get("rule", {}).get("issues", []) or [c.get("reason", "")])
    verdict_limits = [{"perspective": i.agent, "section": SECTION_OF[i.agent], "criterion": i.criterion, "issue": i.text}
                      for i in carried]
    rewrite_issues += [f"[{limit['section']} 판정 한계 명시] {limit['issue']}" for limit in verdict_limits]
    if b.updates["gaps"]:
        rewrite_issues.append("Supervisor가 새로 기록한 근거 공백을 7장 한계점에 근거 부족으로 명시할 것")
    if rewrite_issues and b.can_retry("report"):
        b.bump("report", {"issues": [i for i in rewrite_issues if i][:12],
                          "verdict_limits": verdict_limits, "carried_keys": carried_keys})
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
