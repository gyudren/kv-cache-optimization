"""참고문헌(References) 구간 색인 제외.

설계 산출물 B-3): 참고문헌 구간은 색인 대상에서 제외하되, 페이지 예산(≤200p) 계산에는
보수적으로 포함한다. 이 모듈은 "제외 대상 페이지"만 판별하며, 예산 계산은
preprocessing.budget 에서 문서 전체 페이지 수 기준으로 별도 수행한다.
"""

import re

from preprocessing.page_text import Page

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references?|bibliography|참고\s*문헌)\s*$",
    re.IGNORECASE,
)


def find_reference_start_page(pages: list[Page], manual_start_page: int | None = None) -> int | None:
    """References 절이 시작하는 페이지 번호를 찾는다.

    manual_start_page가 지정되면(manifest.json의 reference_start_page) 그 값을 그대로 사용하고,
    없으면 각 페이지 첫 몇 줄에서 "References" 류의 제목을 자동 탐지한다.
    이후 페이지는 모두 참고문헌 구간으로 간주한다(참고문헌 뒤에 부록이 있는 경우는
    manifest에서 reference_start_page를 직접 지정해 처리).
    """
    if manual_start_page is not None:
        return manual_start_page

    for page in pages:
        first_lines = [line.strip() for line in page.text.splitlines() if line.strip()][:3]
        if any(REFERENCE_HEADING_RE.match(line) for line in first_lines):
            return page.page_number
    return None


def drop_reference_pages(
    pages: list[Page], reference_start_page: int | None
) -> tuple[list[Page], list[Page]]:
    """색인 대상 본문 페이지와 제외된 참고문헌 페이지를 분리한다."""
    if reference_start_page is None:
        return list(pages), []

    body_pages = [p for p in pages if p.page_number < reference_start_page]
    excluded_pages = [p for p in pages if p.page_number >= reference_start_page]
    return body_pages, excluded_pages
