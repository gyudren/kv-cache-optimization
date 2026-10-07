"""품질 평가 노드: 규칙 검사(하드 게이트)와 LLM Judge로 4항목을 판정한다.

항목 통과는 규칙 통과 AND Judge 통과이며, 규칙 실패는 Judge가 뒤집을 수 없다.
Supervisor가 근거 공백(gaps)으로 기록한 관점·기술은 편향·커버리지 규칙에서 인정한다.
"""
from __future__ import annotations
import re
from collections import Counter
from datetime import date
from typing import Any
from ..config import MAX_SINGLE_SOURCE_SHARE, MIN_DISTINCT_SOURCES, PERSPECTIVES
from ..evidence_store import hydrate, source_unit
from ..observability import log_decision
from ..prompts import prompt_template
from ..reporting.sections import (PAPER_CITATION, WEB_CITATION, citeable_evidence, validate_report)
from ..schemas import EvalVerdict
from ..state import deduplicate_evidence, perspective

CRITERIA = ("groundedness", "neutrality", "bias_control", "coverage")
# 항목별 기본 경로. 실제 경로는 원인(target_agents)이 정한다: 관점이면 재조사, report면 재작성.
REWRITE_CRITERIA = ("groundedness", "neutrality")
REINVESTIGATE_CRITERIA = ("bias_control", "coverage")
# 코드가 에이전트 결과를 그대로 옮기는 보고서 부분. Judge가 원인 에이전트를 지목할 때 쓴다.
AGENT_SOURCED_SECTIONS = {"4.1 TRL 표": "tech", "4.2 시장성 판정표": "market",
                          "4.3 이해관계자 판정표": "stakeholder", "4.4 D1-D7 판정표": "domain"}
TECHS = ("mla", "itme")
SECTION_PERSPECTIVE = {"4.1": "tech", "4.2": "market", "4.3": "stakeholder", "4.4": "domain"}

# 중립성 규칙: 단어 하나가 아니라 두 평가 대상 사이의 우열·추천·지시 표현만 잡는다.
_TECH = r"(MLA|ITME|SW|HW|소프트웨어|하드웨어)"
COMPARATIVE = re.compile(rf"{_TECH}\S*\s*보다\s*[^.。|]{{0,20}}?((더\s*)?(우수|우월|뛰어나|낫|효율적|유리|앞서|성숙|적합|빠르))")
ENDORSEMENT = re.compile(r"(추천한다|추천됨|권장한다|권고한다|승자|우승|압도적|최선의\s*선택|최고의\s*(기술|선택)|능가|우위에\s*있|우위를\s*점)")
TECH_DIRECTIVE = re.compile(r"((MLA|ITME)\S*\s*(을|를)?\s*(우선\s*)?(채택|도입|선택)(해야|하라|할\s*것을))")
# 같은 절 안에서 표현 바로 뒤에 오는 부정·유보만 면제한다. 뒤 절의 부정은 앞 절의 우열 판정을 지우지 않는다.
NEARBY_NEGATION = re.compile(r"^[^.。|]{0,25}?(않|아니|없|어렵|판단하지|가리지|정하지)")
CLAUSE_BREAK = re.compile(r"(지만|는데|으나|이나|반면|그러나|,|;)")
# 단위가 붙은 수치만 본다. TRL 등급이나 날짜는 대상이 아니다.
QUANTITY = re.compile(r"\d[\d,.]*\s*(%|배|×|GB|GiB|TB|TiB|MB|ms|μs|us\b|tokens?\b|토큰|tok/s|req/s|x\b)")
EVIDENCE_CITATION = re.compile(r"\[\d+,\s*p\.\d+\]|\[W\d+\]")
# 팀 설계 문서 [D]를 인용할 수 있는 장. [D]는 기술 사실의 근거로 쓰지 않는다.
DESIGN_CHAPTERS = ("SUMMARY", "1", "2")
POSITIVE = {"긍정", "적합"}
CONCERN = {"우려", "제약"}
MIXED = {"혼재", "조건부"}  # 조건부는 긍정·우려 근거를 함께 담은 판정으로 본다
VERDICT_FIELDS = {
    "market": ("market_size_growth_verdict", "adoption_verdict", "ecosystem_verdict", "verdict"),
    "stakeholder": ("competitors_verdict", "developers_adopters_verdict", "investors_verdict", "verdict"),
}


def _body(report: str) -> str:
    return report.split("\n## REFERENCE", 1)[0]


def _chapters(body: str) -> list[tuple[str, str]]:
    """'## ' 장 단위로 나눈다. 장 키는 SUMMARY 또는 장 번호("1"~"7")."""
    out = []
    for block in re.split(r"(?m)^## ", body):
        if not block.strip():
            continue
        heading, _, text = block.partition("\n")
        number = re.match(r"\s*(\d+)\.", heading)
        out.append((number.group(1) if number else heading.strip().split()[0] if heading.strip() else "", text))
    return out


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        if line.startswith("|"):
            out.append(line)  # 표는 행 단위로 본다
        else:
            out.extend(s for s in re.split(r"(?<=[.!?。])\s+", line) if s.strip())
    return out


def _acknowledged(state: dict, name: str, tech: str) -> bool:
    """Supervisor가 이 관점·기술을 근거 공백으로 기록했는가.

    기술을 지정한 공백은 그 기술만, 기술 없는 공백은 관점이 excluded일 때만 두 기술 모두 인정한다.
    """
    gaps = [gap for gap in state.get("gaps", []) if gap["perspective"] == name]
    if any(gap["technology"] == tech for gap in gaps):
        return True
    excluded = (state.get("perspective_status") or {}).get(name) == "excluded"
    return excluded and any(gap["technology"] is None for gap in gaps)


# 고유 출처 수 규칙을 적용하는 관점. domain은 기술별 원문 1편만 쓰므로 빠지고,
# 단일 출처 위험은 agents/domain.py에서 판정을 조건부로 낮춰 드러낸다.
SOURCE_RULE_PERSPECTIVES = ("tech", "market", "stakeholder")


def evidence_shortfalls(state: dict, name: str) -> list[dict]:
    """관점·기술별 고유 출처 수 검사. Supervisor 충분성 검증과 편향 규칙이 같이 쓴다.

    시도마다 누적하면 1개씩만 찾아도 통과하므로 최신 시도의 source_units만 센다.
    source_units가 없는 이전 형식 State에서만 누적 evidence로 센다.
    """
    if name not in SOURCE_RULE_PERSPECTIVES:
        return []
    result = perspective(state, name)
    latest = result.get("source_units")
    evidence = None if latest is not None else deduplicate_evidence(state.get("evidence", []))
    out = []
    for tech in TECHS:
        if latest is not None:
            distinct = set(latest.get(tech, []))
        else:
            distinct = {source_unit(ev) for ev in evidence if ev.get("agent") == name and ev.get("technology") == tech}
        if len(distinct) < MIN_DISTINCT_SOURCES:
            out.append({"perspective": name, "technology": tech, "distinct": len(distinct),
                        "required": MIN_DISTINCT_SOURCES,
                        "reason": f"{name}/{tech}: 고유 출처 {len(distinct)}개(<{MIN_DISTINCT_SOURCES}, 최신 시도 기준)"})
    return out


# 규칙 결과의 item은 {agent, technology, text}이고, agent가 None이면 보고서 서술 문제다.
def _item(agent: str | None, technology: str | None, text: str) -> dict:
    return {"agent": agent, "technology": technology, "text": text}


def _report_items(issues: list[str]) -> list[dict]:
    return [_item(None, None, issue) for issue in issues]


def _label(item: dict) -> str:
    scope = "/".join(x for x in (item["agent"], item["technology"]) if x)
    return f"{scope}: {item['text']}" if scope else item["text"]


def _result(items: list[dict]) -> dict:
    targets = list(dict.fromkeys(item["agent"] or "report" for item in items))
    return {"passed": not items, "issues": [_label(i) for i in items], "items": items, "targets": targets}


def check_groundedness(state: dict) -> dict:
    from ..agents.report import KV_SCALE_FORMULA, KV_SCALE_TABLE  # 코드가 넣는 [D] 설계 표·산식
    report = state.get("report", "")
    validation = validate_report(report, state.get("evidence", []))
    issues = list(validation["issues"])
    static = {line.strip() for line in KV_SCALE_TABLE.splitlines()} | set(_sentences(KV_SCALE_FORMULA))
    for chapter, text in _chapters(_body(report)):
        design_ok = chapter in DESIGN_CHAPTERS
        for sentence in _sentences(text):
            if sentence in static:
                continue
            if not design_ok and "[D]" in sentence and not EVIDENCE_CITATION.search(sentence):
                issues.append(f"설계 문서 [D]만 인용(SUMMARY·1·2장 설계 조건 밖): {sentence[:120]}")
                continue
            if not QUANTITY.search(sentence):
                continue
            cited = EVIDENCE_CITATION.search(sentence) or (design_ok and "[D]" in sentence)
            if not cited:
                issues.append(f"수치 문장에 인용 없음{'([D]는 SUMMARY·1·2장 설계 전제에만 허용)' if '[D]' in sentence else ''}: {sentence[:120]}")
    return {**_result(_report_items(issues)), "pdf_pages": validation.get("pdf_pages")}


def neutrality_issues(text: str) -> list[str]:
    issues = []
    for sentence in _sentences(text):
        for pattern in (COMPARATIVE, ENDORSEMENT, TECH_DIRECTIVE):
            match = pattern.search(sentence)
            tail = CLAUSE_BREAK.split(sentence[match.end():], maxsplit=1)[0] if match else ""
            if match and not NEARBY_NEGATION.search(tail):
                issues.append(f"우열·추천 표현 '{match.group(0)[:30]}': {sentence[:120]}")
                break
    return issues


def check_neutrality(state: dict) -> dict:
    return _result(_report_items(neutrality_issues(_body(state.get("report", "")))))


def _verdicts(state: dict, name: str, tech: str) -> list[str]:
    if name in VERDICT_FIELDS:
        result = perspective(state, name).get("technologies", {}).get(tech, {})
        return [result.get(field) for field in VERDICT_FIELDS[name] if result.get(field)]
    if name == "domain":
        return [item["verdict"] for item in perspective(state, "domain").get("items", [])
                if item.get("technology") == tech and item.get("verdict") != "근거 부족"]
    return []  # TRL은 긍정/우려 판정이 아니라 등급이다


def check_bias(state: dict) -> dict:
    evidence = deduplicate_evidence(state.get("evidence", []))
    found = []
    for name in PERSPECTIVES:
        for tech in TECHS:
            if _acknowledged(state, name, tech):
                continue
            items = [ev for ev in evidence if ev.get("agent") == name and ev.get("technology") == tech]
            problems = []
            short = next((x for x in evidence_shortfalls(state, name) if x["technology"] == tech), None)
            if short:
                problems.append(f"고유 출처 {short['distinct']}개(<{MIN_DISTINCT_SOURCES})")
            web = [ev for ev in items if ev.get("source_type") == "web"]
            if len(web) >= MIN_DISTINCT_SOURCES:
                publisher, count = Counter(ev.get("publisher") or ev.get("url", "") for ev in web).most_common(1)[0]
                if count / len(web) > MAX_SINGLE_SOURCE_SHARE:
                    problems.append(f"단일 발행처 {publisher} 비중 {count / len(web):.0%}(>{MAX_SINGLE_SOURCE_SHARE:.0%})")
            verdicts = set(_verdicts(state, name, tech))
            if verdicts:
                has_positive = bool(verdicts & (POSITIVE | MIXED))
                has_concern = bool(verdicts & (CONCERN | MIXED))
                if not (has_positive and has_concern):
                    side = "긍정" if has_positive else "우려"
                    problems.append(f"판정이 {side} 한쪽뿐(반대 방향 근거 미탐색)")
            if problems:
                found.append(_item(name, tech, ", ".join(problems)))
    return _result(found)


def _section(report: str, number: str) -> str:
    match = re.search(rf"^### {re.escape(number)}[^\n]*\n(.*?)(?=^### |^## |\Z)", report, re.S | re.M)
    return match.group(1) if match else ""


def _has_result(state: dict, name: str, tech: str) -> bool:
    result = perspective(state, name)
    if name == "tech":
        return bool((result.get("trl") or {}).get(tech, "").strip())
    if name == "domain":
        return any(item.get("technology") == tech for item in result.get("items", []))
    summary = result.get("technologies", {}).get(tech, {}).get("summary", "")
    return bool(summary.strip()) and summary.strip() != "근거 부족"


def check_coverage(state: dict) -> dict:
    report = state.get("report", "")
    found = []
    for number, name in SECTION_PERSPECTIVE.items():
        text = _section(report, number)
        narrative = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("|"))
        table = [line for line in text.splitlines() if line.lstrip().startswith("|")]
        problems = []
        if not table:
            problems.append("판정 표 없음")
        if len(narrative.strip()) < 40:
            problems.append("서술 없음")
        for tech, label in (("mla", "MLA"), ("itme", "ITME")):
            if label not in text:
                problems.append(f"{label} 미기재")  # 근거 공백이어도 절에 기술명은 있어야 한다
            if not _has_result(state, name, tech) and not _acknowledged(state, name, tech):
                problems.append(f"{label} 관점 결과 없음")
        if problems:
            found.append(_item(name, None, f"{number} 절 " + ", ".join(problems)))
    return _result(found)


RULES = {"groundedness": check_groundedness, "neutrality": check_neutrality,
         "bias_control": check_bias, "coverage": check_coverage}


def gap_lines(gaps: list[dict]) -> list[str]:
    return [_label({"agent": g["perspective"], "technology": g["technology"], "text": g["detail"]}) for g in gaps]


def rule_checks(state: dict) -> dict[str, dict]:
    return {name: check(state) for name, check in RULES.items()}


def cited_snippets(state: dict) -> list[dict]:
    """보고서가 실제 인용한 근거만 원문과 함께 Judge에 넘긴다.

    전체 카탈로그를 넘기면 길이 제한에 뒤쪽 인용이 잘려 '근거에 없는 인용'으로 오판된다.
    """
    body = _body(state.get("report", ""))
    used = set(PAPER_CITATION.findall(body)) | {f"W{n}" for n in WEB_CITATION.findall(body)}
    out = []
    for ev in citeable_evidence(hydrate(state.get("evidence", []))):
        cite = ev["citation"]
        key = cite.strip("[]")
        paper_key = tuple(x.strip().replace("p.", "") for x in key.split(",")) if "," in key else None
        if (paper_key and paper_key in used) or key in used:
            out.append({"citation": cite, "claim": ev.get("claim", ""), "title": ev.get("title", ""),
                        "publisher": ev.get("publisher", ""), "published_at": ev.get("published_at", ""),
                        "url": ev.get("url", ""), "excerpt": ev.get("excerpt", "")[:1000],
                        "excerpt_truncated": bool(ev.get("excerpt_truncated"))})
    return out


def judge(state: dict, rules: dict[str, dict], llm: Any, snippets: list[dict] | None = None) -> EvalVerdict:
    rule_summary = {name: {"passed": r["passed"], "issues": r["issues"][:10]} for name, r in rules.items()}
    snippets = cited_snippets(state) if snippets is None else snippets
    truncated = sum(1 for s in snippets if s.get("excerpt_truncated"))
    warning = (f"WARNING: {truncated} of {len(snippets)} cited excerpts are TRUNCATED to a short prefix because the full-text "
               "evidence store is unavailable (marked excerpt_truncated=true). Do not treat a claim as supported unless the "
               "truncated text itself supports it; mention unverifiable citations in the groundedness reason.\n"
               if truncated else "")
    return llm.generate_structured(
        prompt_template("quality_evaluator") + "\n" + warning +
        f"Today's date (search/verification date): {date.today().isoformat()}.\n"
        f"Deterministic rule results (cannot be overruled): {rule_summary}\n"
        f"Report parts copied verbatim by code from agent outputs (blame that agent, not report, when the defect is there "
        f"or in prose that only restates that agent's verdict): {AGENT_SOURCED_SECTIONS}\n"
        f"Evidence gaps recorded by the supervisor: {gap_lines(state.get('gaps', []))}\n"
        f"Verified evidence snippets for every citation used in the report: {repr(snippets)}\n"
        f"Report:\n{state.get('report', '')}",
        EvalVerdict,
    )


def combine(rules: dict[str, dict], verdict: EvalVerdict) -> dict:
    """항목별 최종 판정(규칙 AND Judge)과 미달 원인 에이전트를 만든다."""
    criteria = {}
    for name in CRITERIA:
        rule = rules[name]
        llm_part = getattr(verdict, name)
        targets = list(rule["targets"])
        if not llm_part.passed:
            # Judge가 지목한 원인을 따른다. 코드가 옮긴 판정표의 결함은 보고서를 다시 써도 그대로라
            # 해당 관점을 재조사해야 한다.
            if llm_part.target_agent in (*PERSPECTIVES, "report"):
                if llm_part.target_agent not in targets:
                    targets.append(llm_part.target_agent)
            elif not targets:
                targets = ["report"]
        score = max(1, min(5, int(llm_part.score)))
        passed = rule["passed"] and llm_part.passed
        criteria[name] = {
            "passed": passed,
            "score": score if rule["passed"] else min(score, 2),
            "reason": "; ".join(rule["issues"][:5]) if not rule["passed"] else llm_part.reason,
            "target_agents": [] if passed else targets,
            "rule": {"passed": rule["passed"], "issues": rule["issues"], "items": rule.get("items", [])},
            "judge": llm_part.model_dump(),
        }
    return {"passed": all(c["passed"] for c in criteria.values()), "criteria": criteria}


def quality_evaluator_node(state: dict, llm: Any) -> dict:
    rules = rule_checks(state)
    snippets = cited_snippets(state)
    result = combine(rules, judge(state, rules, llm, snippets))
    result["report_attempt"] = state.get("retry_counts", {}).get("report", 0)
    truncated = sum(1 for s in snippets if s.get("excerpt_truncated"))
    # 원문 없이 축약 발췌로만 판정한 건수를 결과에 남긴다.
    result["evidence_store"] = {"cited": len(snippets), "cited_missing_full_text": truncated}
    result["warnings"] = ([f"인용 근거 {truncated}/{len(snippets)}건의 원문이 저장소에 없어 축약 발췌로만 평가됨"]
                          if truncated else [])
    failed = [name for name, c in result["criteria"].items() if not c["passed"]]
    log_decision(state["trace_id"], state.get("step_count", 0), "quality_evaluator",
                 "pass" if result["passed"] else "fail:" + ",".join(failed),
                 "; ".join(f"{n}: {result['criteria'][n]['reason'][:160]}" for n in failed) or "4항목 통과",
                 scores={n: c["score"] for n, c in result["criteria"].items()})
    return {"eval_result": result}
