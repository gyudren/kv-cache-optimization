"""Supervisor 결정 정책. State 제어 필드만 보고 다음 노드를 고르는 순수 함수다.

매 진입마다 관점 조사, 종합, 보고서, 품질 평가 순으로 지금 할 일을 고른다.
충분성 재조사는 RETRY_LIMITS, 종합이 요청한 후속 재조사는 FOLLOWUP_LIMITS, 품질 평가가 원인으로 지목한
재조사는 REINVESTIGATE_LIMITS로 따로 센다.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from ..config import (FINALIZE_STEPS, FOLLOWUP_LIMITS, MAX_STEPS, PERSPECTIVES, REINVESTIGATE_LIMITS, RETRY_LIMITS,
                      recursion_limit_for)
from ..evaluation.quality import CRITERIA, SECTION_PERSPECTIVE, direction_shortfalls, evidence_shortfalls
from ..state import Gap, make_gap, perspective, split_scope
from ..tools import TECH_TOKENS, mentioned_techs

END_NODE = "__end__"
DOWNSTREAM = ("synthesis", "report", "quality_evaluator")
LABEL = {"tech": "기술 성숙도(TRL)", "market": "시장성", "stakeholder": "이해관계자", "domain": "도메인 적용성"}
SECTION_OF = {name: number for number, name in SECTION_PERSPECTIVE.items()}  # 관점별 판정표 절(4.1~4.4)


@dataclass(frozen=True)
class _AgentIssue:
    """원인이 관점 에이전트인 평가 미달 항목. key가 같으면 같은 항목·같은 사유로 본다."""
    agent: str
    criterion: str
    text: str
    reason_key: str
    technology: str | None = None

    @property
    def key(self) -> str:
        return f"{self.criterion}|{self.agent}|{self.reason_key}"


@dataclass(frozen=True)
class Policy:
    max_steps: int = MAX_STEPS
    retry_limits: Mapping[str, int] = field(default_factory=lambda: MappingProxyType(dict(RETRY_LIMITS)))
    followup_limits: Mapping[str, int] = field(default_factory=lambda: MappingProxyType(dict(FOLLOWUP_LIMITS)))
    reinvestigate_limits: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType(dict(REINVESTIGATE_LIMITS)))

    def limit(self, name: str) -> int:
        return self.retry_limits.get(name, 0)

    def followup_limit(self, name: str) -> int:
        return self.followup_limits.get(name, 0)

    def reinvestigate_limit(self, name: str) -> int:
        return self.reinvestigate_limits.get(name, 0)

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
    """관점의 근거 충분도. 한 번 excluded가 되면 그대로 유지한다."""
    if (state.get("perspective_status") or {}).get(name) == "excluded":
        return "excluded"
    status = (state.get("node_status") or {}).get(name, "pending")
    if status == "failed":
        return "failed"
    if status != "done":
        return "pending"  # 재개 직후 running으로 남은 경우도 포함
    # 에이전트의 자기 보고와 별개로 출처 수와 판정 방향(긍정·우려 한쪽뿐인가)을 다시 확인한다.
    # 방향 검사는 품질 평가의 편향 규칙과 같은 함수다. 보고서 전에 걸러야 반대 방향 근거를 재조사할 수 있다.
    if (not perspective(state, name).get("sufficient") or evidence_shortfalls(state, name)
            or direction_shortfalls(state, name)):
        return "insufficient"
    return "sufficient"


# 평가 사유 유형별 재검색 힌트. 사유 문장 자체는 검색어로 쓰지 않는다.
EVAL_QUERY_HINTS = (
    ("긍정 한쪽뿐", "limitations risks concerns criticism"),
    ("우려 한쪽뿐", "adoption benefits positive outlook support"),
    ("단일 발행처", "independent analysis industry report"),
    ("고유 출처", "additional independent sources analysis"),
    ("TRL", "production deployment commercial service availability"),
    ("상용", "production deployment commercial service availability"),
)
GENERIC_EVAL_HINT = "independent sources limitations adoption evidence"
# 판정이 한쪽뿐일 때 반대 방향을 찾는 검색 힌트(지금 판정 방향 → 찾을 방향의 검색어)
DIRECTION_HINTS = {"긍정": "limitations risks concerns criticism", "우려": "adoption benefits positive outlook support"}
SHORTFALL_HINT = "additional independent sources analysis"
TECHS = ("mla", "itme")


def _item_tech(text: str) -> list[str]:
    """항목이 가리키는 기술. 접두어나 본문 언급으로 하나로 정해지지 않으면 두 기술 모두."""
    prefix = re.match(r"^\w+/(mla|itme):", str(text))
    if prefix:
        return [prefix.group(1)]
    techs = sorted(mentioned_techs(text))
    return techs if len(techs) == 1 else list(TECHS)


def _agent_missing_query(text: str) -> str:
    """에이전트가 보고한 부족 항목을 검색어로 다듬는다."""
    text = re.sub(r"^\w+(/\w+)?:\s*", "", str(text))
    text = re.sub(r"\([^)]*\)", " ", text)
    for tokens in TECH_TOKENS.values():
        for token in tokens:
            text = re.sub(token, " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())[:100]


def _rework(items: list[tuple[list[str], str, str]]) -> dict:
    """(대상 기술, 사유, 검색 힌트) 목록을 재작업 지시로 묶는다. 검색 힌트는 기술별로 따로 둔다."""
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


def _sufficiency_feedback(shortfalls: list[dict], missing: list[str], optional: list[str] = (),
                          directions: list[dict] = ()) -> dict:
    """충분성 재조사 지시를 만든다.

    필수 결함 문장은 검색어로 쓸 수 없어 사유로만 넘기고, 검색 힌트는 선택 항목에서 만든다.
    판정이 한쪽뿐인 기술에는 반대 방향 검색 힌트를 붙인다. 힌트가 하나도 없는 기술에는 일반 힌트를 붙인다.
    """
    items = [([x["technology"]], x["reason"], SHORTFALL_HINT) for x in shortfalls]
    items += [([x["technology"]], f"{x['reason']} — {x['opposite']} 방향 근거를 찾아 판정에 반영할 것",
               DIRECTION_HINTS[x["side"]]) for x in directions]
    items += [(_item_tech(m), str(m), "") for m in missing if str(m).strip()]
    items += [(_item_tech(o), "", _agent_missing_query(o)) for o in optional if str(o).strip()]
    hinted = {tech for techs, _, query in items if query for tech in techs}
    needy = {tech for techs, reason, _ in items if reason for tech in techs}
    items += [([tech], "", SHORTFALL_HINT) for tech in sorted(needy - hinted)]
    return _rework(items)


def _eval_feedback(issues: list[str]) -> dict:
    """평가 미달 사유로 재조사 지시를 만든다. 검색어는 EVAL_QUERY_HINTS에서 고른다."""
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
        self.followups = dict(state.get("followup_counts") or {})
        self.reinvestigations = dict(state.get("reinvestigate_counts") or {})
        self.updates: dict[str, Any] = {"retry_counts": {}, "followup_counts": {}, "reinvestigate_counts": {},
                                        "feedback": {}, "node_status": {}, "perspective_status": {}, "gaps": []}

    def can_retry(self, name: str) -> bool:
        return self.retry.get(name, 0) < self.policy.limit(name)

    def bump(self, name: str, feedback: dict | None = None) -> None:
        self.retry[name] = self.retry.get(name, 0) + 1
        self.updates["retry_counts"][name] = self.retry[name]
        if feedback is not None:
            self.updates["feedback"][name] = feedback

    def can_followup(self, name: str) -> bool:
        return self.followups.get(name, 0) < self.policy.followup_limit(name)

    def can_reinvestigate(self, name: str) -> bool:
        return self.reinvestigations.get(name, 0) < self.policy.reinvestigate_limit(name)

    def followup(self, names: list[str], feedback_by_name: dict[str, dict], counter: str = "followup_counts") -> None:
        """종합(followup_counts)·평가(reinvestigate_counts)가 지목한 관점을 다시 보낸다.

        제외됐던 관점도 pending으로 되돌린다.
        """
        counts = self.followups if counter == "followup_counts" else self.reinvestigations
        for name in names:
            counts[name] = counts.get(name, 0) + 1
            self.updates[counter][name] = counts[name]
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

    def all_gaps(self) -> list[Gap]:
        return [*(self.state.get("gaps") or []), *self.updates["gaps"]]

    def gap(self, gap: Gap) -> None:
        if gap not in self.all_gaps():
            self.updates["gaps"].append(gap)

    def done(self, targets: list[str], decision: str, reason: str, **extra) -> Decision:
        updates = {k: v for k, v in self.updates.items() if v}
        updates.update(extra)
        return Decision(targets, decision, reason, updates)

    def end(self, decision: str, reason: str, verified: bool) -> Decision:
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
        directions = direction_shortfalls(state, name)
        missing = [*(x["reason"] for x in shortfalls), *perspective(state, name).get("missing", []),
                   *(x["reason"] for x in directions)]
        error = (state.get("last_error") or {}).get(name, "")
        if not finalize and b.can_retry(name):
            if status == "failed":
                # 실행 실패는 직전 지시를 그대로 다시 넘긴다.
                b.bump(name, {**state.get("feedback", {}).get(name, {}), "last_error": error})
                reasons.append(f"{name}: 실행 실패 재시도 {b.retry[name]}/{policy.limit(name)} ({error[:80]})")
            else:
                result = perspective(state, name)
                b.bump(name, _sufficiency_feedback(shortfalls, result.get("missing", []), result.get("missing_optional", []),
                                                   directions))
                reasons.append(f"{name}: 근거 부족 재조사 {b.retry[name]}/{policy.limit(name)} "
                               f"({'; '.join(map(str, missing[:2]))[:120]})")
            to_run.append(name)
            continue
        # 재시도 한도나 단계 상한에 걸리면 제외하고 근거 공백으로 남긴다.
        b.updates["perspective_status"][name] = "excluded"
        if finalize:
            b.gap(make_gap(name, "step_limit", "조사 단계 상한에 도달해 추가 조사를 멈춤"))
        elif status == "failed":
            b.gap(make_gap(name, "agent_failed", f"에이전트 실행 실패로 결과 없음 ({error[:120]})"))
        else:
            # 반대 방향을 다시 찾고도 한쪽뿐이면 그 사실을 따로 남긴다. 보고서가 판정표 아래에 한계로 밝히고,
            # 편향 규칙과 Judge는 "탐색했으나 없음"을 확증편향(탐색 안 함)과 구분해 판정한다.
            # 직전 재조사 지시에 그 방향 힌트가 실제로 들어갔을 때만 "재검색했으나 없음"이다. 다른 사유로 한도를
            # 다 쓴 뒤에 처음 한쪽 판정이 나온 경우는 탐색하지 않았으므로 일반 근거 부족으로 남긴다.
            last_hints = (state.get("feedback", {}).get(name, {}).get("queries_by_tech") or {})
            searched = [x for x in directions if DIRECTION_HINTS[x["side"]] in last_hints.get(x["technology"], [])]
            for x in searched:
                b.gap(make_gap(name, "one_sided", f"{x['opposite']} 방향 근거를 재검색했으나 찾지 못해 {x['side']} 판정만 남음",
                               x["technology"]))
            one_sided = {x["reason"] for x in searched}
            for item in [m for m in missing if m not in one_sided][:3]:
                technology, detail = split_scope(item)
                b.gap(make_gap(name, "insufficient", detail, technology))
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
        b.gap(make_gap("synthesis", "node_failed", "종합 에이전트 실행 실패로 관점 간 일치·상충 정리가 없음"))
        b.updates["node_status"]["synthesis"] = "skipped"
        return None
    if status != "done" or finalize:
        return None
    synthesis = state.get("synthesis") or {}
    # 종합이 추가 근거를 요구한 관점만 후속 재조사한다.
    requested = [n for n in synthesis.get("needs_source_agents", []) if n in PERSPECTIVES]
    rerun = [n for n in requested if b.can_followup(n)]
    gaps = synthesis.get("evidence_gaps", [])
    for name in requested:
        if name not in rerun:
            reason = next((str(g) for g in gaps if name in str(g)), "")
            b.gap(make_gap(name, "followup_exhausted", "종합 단계에서 추가 근거가 필요하다고 판단함"
                           + (f": {split_scope(reason)[1][:160]}" if reason else "")))
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
        b.gap(make_gap("report", "node_failed", "보고서 에이전트 실행 실패로 보고서 없음"))
        return b.end("end:report_failed", "보고서 재시도 한도 소진", verified=False)

    if not (state.get("report") or "").strip():
        # done인데 본문이 비었으면 Judge를 부르지 않고 실패로 처리한다.
        if not finalize and b.can_retry("report"):
            b.bump("report")
            b.run("report")
            return b.done(["report"], "retry:report", "보고서 본문 없음 → 재작성")
        b.gap(make_gap("report", "node_failed", "보고서 본문이 생성되지 않음"))
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
        b.gap(make_gap("quality_evaluator", "node_failed", "품질 평가 실행 실패로 보고서 미검증"))
        return b.end("end:unverified", "품질 평가 재시도 한도 소진", verified=False)

    result = state.get("eval_result") or {}
    if result.get("passed"):
        return b.end("end:passed", "품질 평가 4항목 통과", verified=True)
    criteria = result.get("criteria", {})
    failed = [n for n, c in criteria.items() if not c.get("passed")]
    if finalize:
        return b.end("end:step_limit", f"단계 상한 도달 — 품질 평가 미달 항목 {failed} 남김", verified=False)

    # 미달 원인이 관점 에이전트면 그 관점만 재조사한다.
    issues_by_agent: dict[str, list[_AgentIssue]] = {}
    for name in CRITERIA:
        c = criteria.get(name, {})
        if c.get("passed"):
            continue
        for agent in c.get("target_agents", []):
            if agent in PERSPECTIVES:
                own = [_AgentIssue(agent, name, f"{agent}/{i['technology']}: {i['text']}" if i["technology"]
                                   else f"{agent}: {i['text']}", i["text"], i["technology"])
                       for i in c.get("rule", {}).get("items", []) if i["agent"] == agent]
                # Judge 사유 문장은 실행마다 달라서 항목과 관점만으로 같은 사유를 판단한다.
                judged = _AgentIssue(agent, name, f"{name}: {c.get('reason', '')[:300]}", "judge")
                issues_by_agent.setdefault(agent, []).extend(own or [judged])
    rerun = [a for a in issues_by_agent if b.can_reinvestigate(a)]
    carried: list[_AgentIssue] = []  # 재조사하지 못해 보고서에 판정 한계로 적을 이슈
    for agent, issues in issues_by_agent.items():
        if agent in rerun:
            continue
        for issue in issues:
            carried.append(issue)
            # 같은 관점·항목의 평가 공백은 한 번만 남긴다.
            if not any(g["perspective"] == agent and g["kind"] == "evaluation" and g["criterion"] == issue.criterion
                       and g["technology"] == issue.technology
                       and (issue.reason_key == "judge" or g["detail"] == issue.reason_key) for g in b.all_gaps()):
                detail = issue.reason_key if issue.reason_key != "judge" else issue.text.split(": ", 1)[-1]
                b.gap(make_gap(agent, "evaluation", detail[:200], issue.technology, issue.criterion))
    if rerun:
        b.followup(rerun, {agent: _eval_feedback([i.text for i in issues_by_agent[agent]]) for agent in rerun},
                   counter="reinvestigate_counts")
        return b.done(rerun, "reinvestigate:" + ",".join(rerun),
                      "품질 평가 미달 원인 관점만 재조사: "
                      + "; ".join(i.text for a in rerun for i in issues_by_agent[a])[:300])

    # 판정 한계를 적어 재작성했는데도 같은 사유로 미달이면 더 반복해도 통과할 수 없으니 멈춘다.
    carried_keys = sorted({issue.key for issue in carried})
    previous = (state.get("feedback") or {}).get("report") or {}
    if carried_keys and set(carried_keys) <= set(previous.get("carried_keys", [])):
        return b.end("end:unverified",
                     "재조사할 수 없는 관점 원인 미달이 판정 한계를 명시한 재작성 후에도 같은 사유로 반복 → 재작성 중단: "
                     + "; ".join(i.text for i in carried)[:240], verified=False)

    # 보고서 서술 결함, 재조사하지 못한 관점 이슈, 새 근거 공백은 보고서 재작성으로 처리한다.
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
