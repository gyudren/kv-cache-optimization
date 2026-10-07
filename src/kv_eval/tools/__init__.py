"""외부 검색 도구와 재검색 질의 구성."""


def retry_queries(subject: str, feedback: dict, limit: int = 2) -> list[str]:
    """Supervisor가 넘긴 부족 항목(missing)을 재검색 질의로 바꾼다.

    부족 항목 목록을 통째로 문자열에 붙이면 검색어가 문장 나열이 되어 결과가 나오지 않으므로,
    항목별로 짧게 잘라 기술명과 묶은 독립 질의로 만든다.
    """
    queries = []
    for item in (feedback or {}).get("rewritten_queries", [])[:limit]:
        text = " ".join(str(item).replace(":", " ").split())[:110]
        if text:
            queries.append(f"{subject} {text}")
    return queries


def rework_note(feedback: dict) -> str:
    """Supervisor의 재작업 지시를 프롬프트 끝에 붙인다(지시가 없으면 빈 문자열)."""
    if not feedback or not (feedback.get("missing") or feedback.get("rewritten_queries")):
        return ""
    return ("\nSUPERVISOR REWORK REQUEST (this is a re-run; address every item using the supplied sources, "
            "and keep 근거 부족 where sources still do not cover it):\n- missing: "
            + "; ".join(map(str, feedback.get("missing", []))) + "\n- search hints: "
            + "; ".join(map(str, feedback.get("rewritten_queries", []))) + "\n")
