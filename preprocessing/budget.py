"""RAG 적재 문서 페이지 예산(≤200p) 검증.

설계 산출물 B-3): "총 96p로 페이지 제한(200p) 이내 — 코드에서 합계 ≤ 200p를 강제 검증".
참고문헌 구간도 예산 계산에는 포함한다(보수적 집계).
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
