"""품질 평가 노드 (DEV_PLAN §6, Hybrid = 규칙 검사 하드 게이트 + LLM Judge 내용 게이트).

| 항목 | 규칙 검사(결정적) | LLM Judge | 미달 시 경로 |
|---|---|---|---|
| groundedness | validate_report(목차·인용·REFERENCE·10p) + 수치 문장 인용 필수 | 발췌가 주장을 뒷받침하는가 | 보고서 재작성 |
| neutrality | 우열·추천 표현 탐지(부정·면책 문장은 제외) | 암묵적 우열 판정 | 보고서 재작성 |
| bias_control | 기술·관점별 고유 출처 ≥2, 웹 단일 발행처 비중 상한, 긍정·우려 양방향 근거 | 한쪽 근거 편중 | 원인 관점 재조사 |
| coverage | 4.1~4.4 서술·판정표 존재, 두 기술 모두 기재, 관점 결과 존재 | 4관점 실질 서술 | 원인 관점 재조사 |

규칙 검사 실패는 LLM이 뒤집을 수 없다(항목 통과 = 규칙 통과 AND Judge 통과).
Supervisor가 재조사 상한을 넘겨 근거 공백(gaps)으로 기록한 관점·기술의 편향·커버리지 미달은
"근거 부족을 명시한 상태"로 보고 규칙상 인정한다(공백을 숨기지 않고 드러내는 것이 목표이므로).
"""
from __future__ import annotations
import re
from collections import Counter
from datetime import date
from typing import Any
from ..config import MAX_SINGLE_SOURCE_SHARE, MIN_DISTINCT_SOURCES, PERSPECTIVES
from ..observability import log_decision
from ..prompts import prompt_template
from ..reporting.sections import (PAPER_CITATION, WEB_CITATION, citeable_evidence, validate_report)
from ..schemas import EvalVerdict
from ..state import deduplicate_evidence, perspective

CRITERIA = ("groundedness", "neutrality", "bias_control", "coverage")
# 미달 원인별 경로: 보고서 표현 문제는 재작성, 근거 수집 문제는 원인 관점 재조사
REWRITE_CRITERIA = ("groundedness", "neutrality")
REINVESTIGATE_CRITERIA = ("bias_control", "coverage")
TECHS = ("mla", "itme")
SECTION_PERSPECTIVE = {"4.1": "tech", "4.2": "market", "4.3": "stakeholder", "4.4": "domain"}

# 우열·추천 표현. "추천하지 않는다", "승자 대신" 같은 면책 문장은 NEGATION으로 걸러낸다.
BANNED_PHRASES = re.compile(r"(추천|우월|우위|승자|우승|더\s*낫|더\s*우수|압도적|최선의\s*선택|최고의|1위|채택해야|도입해야)")
NEGATION = re.compile(r"(않|아니|없|대신|금지|배제|지양|말고)")
# 단위가 붙은 수치(성능·용량·비율). TRL 같은 등급 숫자나 날짜는 대상이 아니다.
QUANTITY = re.compile(r"\d[\d,.]*\s*(%|배|×|GB|GiB|TB|TiB|MB|ms|μs|us\b|tokens?\b|토큰|tok/s|req/s|x\b)")
ANY_CITATION = re.compile(r"\[\d+,\s*p\.\d+\]|\[W\d+\]|\[D\]")
POSITIVE = {"긍정", "적합"}
CONCERN = {"우려", "제약"}
MIXED = {"혼재", "조건부"}  # 조건부 = 조건이 붙은 적합 → 긍정·우려 양쪽 근거를 함께 담은 판정
VERDICT_FIELDS = {
    "market": ("market_size_growth_verdict", "adoption_verdict", "ecosystem_verdict", "verdict"),
    "stakeholder": ("competitors_verdict", "developers_adopters_verdict", "investors_verdict", "verdict"),
}


def _body(report: str) -> str:
    return report.split("\n## REFERENCE", 1)[0]


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        if line.startswith("|"):
            out.append(line)  # 표는 행 단위로 본다(행 안에 인용이 있어야 한다)
        else:
            out.extend(s for s in re.split(r"(?<=[.!?。])\s+", line) if s.strip())
    return out


def _acknowledged(gaps: list[str], name: str, tech: str | None = None) -> bool:
    """Supervisor가 이 관점(·기술)을 근거 공백으로 기록했는가."""
    prefixes = (f"{name}:",) + ((f"{name}/{tech}:",) if tech else ())
    return any(gap.startswith(prefixes) for gap in gaps)


# ---- 규칙 검사 -----------------------------------------------------------------------------
def check_groundedness(state: dict) -> dict:
    from ..agents.report import KV_SCALE_FORMULA, KV_SCALE_TABLE  # 코드가 넣는 설계 표·산식(출처 [D])
    report = state.get("report", "")
    validation = validate_report(report, state.get("evidence", []))
    issues = list(validation["issues"])
    static = {line.strip() for line in KV_SCALE_TABLE.splitlines()} | set(_sentences(KV_SCALE_FORMULA))
    for sentence in _sentences(_body(report)):
        if sentence in static or not QUANTITY.search(sentence):
            continue
        if not ANY_CITATION.search(sentence):
            issues.append(f"수치 문장에 인용 없음: {sentence[:120]}")
    return {"passed": not issues, "issues": issues, "targets": ["report"] if issues else [],
            "pdf_pages": validation.get("pdf_pages")}


def check_neutrality(state: dict) -> dict:
    issues = []
    for sentence in _sentences(_body(state.get("report", ""))):
        match = BANNED_PHRASES.search(sentence)
        if match and not NEGATION.search(sentence):
            issues.append(f"우열·추천 표현 '{match.group(1)}': {sentence[:120]}")
    return {"passed": not issues, "issues": issues, "targets": ["report"] if issues else []}


def _source_unit(ev: dict) -> str:
    if ev.get("source_type") == "web":
        return ev.get("url", "")
    return f"{ev.get('doc_id')}:{ev.get('page')}"


def _verdicts(state: dict, name: str, tech: str) -> list[str]:
    if name in VERDICT_FIELDS:
        result = perspective(state, name).get("technologies", {}).get(tech, {})
        return [result.get(field) for field in VERDICT_FIELDS[name] if result.get(field)]
    if name == "domain":
        return [item["verdict"] for item in perspective(state, "domain").get("items", [])
                if item.get("technology") == tech and item.get("verdict") != "근거 부족"]
    return []  # 기술 조사(TRL)는 긍정/우려 판정이 아니라 등급이다


def check_bias(state: dict) -> dict:
    evidence = deduplicate_evidence(state.get("evidence", []))
    gaps = state.get("gaps", [])
    issues, targets = [], []
    for name in PERSPECTIVES:
        for tech in TECHS:
            if _acknowledged(gaps, name, tech):
                continue
            items = [ev for ev in evidence if ev.get("agent") == name and ev.get("technology") == tech]
            problems = []
            distinct = {_source_unit(ev) for ev in items}
            if len(distinct) < MIN_DISTINCT_SOURCES:
                problems.append(f"고유 출처 {len(distinct)}개(<{MIN_DISTINCT_SOURCES})")
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
                issues.append(f"{name}/{tech}: " + ", ".join(problems))
                targets.append(name)
    return {"passed": not issues, "issues": issues, "targets": list(dict.fromkeys(targets))}


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
    gaps = state.get("gaps", [])
    issues, targets = [], []
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
            if _acknowledged(gaps, name, tech):
                continue
            if label not in text:
                problems.append(f"{label} 미기재")
            if not _has_result(state, name, tech):
                problems.append(f"{label} 관점 결과 없음")
        if problems and not _acknowledged(gaps, name):
            issues.append(f"{name}: {number} 절 " + ", ".join(problems))
            targets.append(name)
    return {"passed": not issues, "issues": issues, "targets": targets}


RULES = {"groundedness": check_groundedness, "neutrality": check_neutrality,
         "bias_control": check_bias, "coverage": check_coverage}


def rule_checks(state: dict) -> dict[str, dict]:
    return {name: check(state) for name, check in RULES.items()}


# ---- LLM Judge ------------------------------------------------------------------------------
def cited_snippets(state: dict) -> list[dict]:
    """Judge에게 보고서가 실제 인용한 근거만 원문과 함께 넘긴다.

    전체 카탈로그를 길이 제한으로 자르면 뒤쪽 인용([W…], 뒷페이지)이 잘려 나가
    "제공된 근거에 없는 인용"이라는 오판이 반복된다. 인용된 항목만 넘기면 모두 들어간다.
    """
    body = _body(state.get("report", ""))
    used = set(PAPER_CITATION.findall(body)) | {f"W{n}" for n in WEB_CITATION.findall(body)}
    out = []
    for ev in citeable_evidence(state.get("evidence", [])):
        cite = ev["citation"]
        key = cite.strip("[]")
        paper_key = tuple(x.strip().replace("p.", "") for x in key.split(",")) if "," in key else None
        if (paper_key and paper_key in used) or key in used:
            out.append({"citation": cite, "claim": ev.get("claim", ""), "title": ev.get("title", ""),
                        "publisher": ev.get("publisher", ""), "published_at": ev.get("published_at", ""),
                        "url": ev.get("url", ""), "excerpt": ev.get("excerpt", "")[:1000]})
    return out


def judge(state: dict, rules: dict[str, dict], llm: Any) -> EvalVerdict:
    rule_summary = {name: {"passed": r["passed"], "issues": r["issues"][:10]} for name, r in rules.items()}
    return llm.generate_structured(
        prompt_template("quality_evaluator") + "\n"
        f"Today's date (search/verification date): {date.today().isoformat()}.\n"
        f"Deterministic rule results (cannot be overruled): {rule_summary}\n"
        f"Evidence gaps recorded by the supervisor: {state.get('gaps', [])}\n"
        f"Verified evidence snippets for every citation used in the report: {repr(cited_snippets(state))}\n"
        f"Report:\n{state.get('report', '')}",
        EvalVerdict,
    )


def combine(rules: dict[str, dict], verdict: EvalVerdict) -> dict:
    """항목별 최종 판정. 통과 = 규칙 통과 AND Judge 통과. 미달 원인 에이전트를 함께 기록한다."""
    criteria = {}
    for name in CRITERIA:
        rule = rules[name]
        llm_part = getattr(verdict, name)
        targets = list(rule["targets"])
        if not llm_part.passed and llm_part.target_agent and llm_part.target_agent not in targets:
            if name in REWRITE_CRITERIA or llm_part.target_agent in PERSPECTIVES:
                targets.append(llm_part.target_agent)
        if name in REWRITE_CRITERIA:
            targets = ["report"] if (targets or not llm_part.passed) else []
        elif not targets and not llm_part.passed:
            targets = ["report"]  # 원인 관점을 특정하지 못하면 보고서 서술 보완으로 처리
        score = max(1, min(5, int(llm_part.score)))
        passed = rule["passed"] and llm_part.passed
        criteria[name] = {
            "passed": passed,
            "score": score if rule["passed"] else min(score, 2),
            "reason": "; ".join(rule["issues"][:5]) if not rule["passed"] else llm_part.reason,
            "target_agents": [] if passed else targets,
            "rule": {"passed": rule["passed"], "issues": rule["issues"]},
            "judge": llm_part.model_dump(),
        }
    return {"passed": all(c["passed"] for c in criteria.values()), "criteria": criteria}


def quality_evaluator_node(state: dict, llm: Any) -> dict:
    rules = rule_checks(state)
    result = combine(rules, judge(state, rules, llm))
    result["report_attempt"] = state.get("retry_counts", {}).get("report", 0)
    failed = [name for name, c in result["criteria"].items() if not c["passed"]]
    log_decision(state["trace_id"], state.get("step_count", 0), "quality_evaluator",
                 "pass" if result["passed"] else "fail:" + ",".join(failed),
                 "; ".join(f"{n}: {result['criteria'][n]['reason'][:160]}" for n in failed) or "4항목 통과",
                 scores={n: c["score"] for n, c in result["criteria"].items()})
    return {"eval_result": result}
