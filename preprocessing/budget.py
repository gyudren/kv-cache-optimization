"""RAG에 넣는 문서의 총 페이지 수가 예산(기본 200p)을 넘지 않는지 검사한다.

참고문헌 구간도 페이지 수에 포함해서 센다.
"""


class PageBudgetExceeded(Exception):
    pass


def enforce_page_budget(document_page_counts: dict[str, int], budget: int = 200) -> int:
    total = sum(document_page_counts.values())
    if total > budget:
        raise PageBudgetExceeded(
            f"총 페이지 수 {total}p가 예산 {budget}p를 초과했습니다: {document_page_counts}"
        )
    return total
