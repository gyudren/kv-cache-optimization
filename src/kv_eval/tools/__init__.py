"""외부 검색 도구와 재검색 질의 구성."""


TECH_TOKENS = {"mla": ("mla", "deepseek"), "itme": ("itme", "cxl")}


def mentioned_techs(text: str) -> set[str]:
    lowered = str(text).lower()
    return {tech for tech, tokens in TECH_TOKENS.items() if any(token in lowered for token in tokens)}


def scoped_feedback(feedback: dict | None, tech: str) -> dict:
    """재작업 지시 중 이 기술에 해당하는 부분만 돌려준다(다른 기술 이름이 검색어에 섞이지 않게).

    Supervisor는 `queries_by_tech`(기술별 검색 힌트)를 구조화해서 넘긴다. 이전 형식(rewritten_queries만)은
    다른 기술만 언급한 항목을 걸러 낸다.
    """
    feedback = feedback or {}
    tech = "itme" if tech.startswith("itme") else tech
    missing = [m for m in feedback.get("missing", []) if not mentioned_techs(m) or tech in mentioned_techs(m)]
    if "queries_by_tech" in feedback:
        queries = list(feedback["queries_by_tech"].get(tech, []))
    else:
        queries = [q for q in feedback.get("rewritten_queries", [])
                   if not mentioned_techs(q) or tech in mentioned_techs(q)]
    if not queries:
        return {}
    return {"missing": missing, "rewritten_queries": queries}


def retry_queries(subject: str, feedback: dict, limit: int = 2, tech: str | None = None) -> list[str]:
    """Supervisor가 넘긴 기술별 검색 힌트를 재검색 질의로 바꾼다.

    부족 사유 문장을 그대로 붙이면 검색어가 문장 나열이 되어 결과가 나오지 않으므로, Supervisor가 구조화한
    힌트(`queries_by_tech[tech]`)만 쓰고 기술명(subject)과 묶은 독립 질의로 만든다.
    """
    scoped = scoped_feedback(feedback, tech) if tech else (feedback or {})
    queries = []
    for item in scoped.get("rewritten_queries", [])[:limit]:
        text = " ".join(str(item).replace(":", " ").split())[:110]
        if text:
            queries.append(f"{subject} {text}")
    return queries


def rework_note(feedback: dict) -> str:
    """Supervisor의 재작업 지시를 프롬프트 끝에 붙인다(지시가 없으면 빈 문자열)."""
    if not feedback or not (feedback.get("missing") or feedback.get("queries_by_tech")):
        return ""
    return ("\nSUPERVISOR REWORK REQUEST (this is a re-run; address every item using the supplied sources, "
            "and keep 근거 부족 where sources still do not cover it):\n- missing: "
            + "; ".join(map(str, feedback.get("missing", []))) + "\n- search hints: "
            + "; ".join(f"{tech}: {', '.join(qs)}" for tech, qs in (feedback.get("queries_by_tech") or {}).items() if qs)
            + "\n")
