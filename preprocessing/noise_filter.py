"""참고문헌(References) 구간 색인 제외.

설계 산출물 B-3): 참고문헌 구간은 색인 대상에서 제외하되, 페이지 예산(≤200p) 계산에는
보수적으로 포함한다. 이 모듈은 "제외 대상 페이지"만 판별하며, 예산 계산은
preprocessing.budget 에서 문서 전체 페이지 수 기준으로 별도 수행한다.

References 제목은 페이지 맨 앞이 아니라(2단 레이아웃에서 본문이 끝나자마자 같은 페이지
하단에 등장하는 경우 등) 페이지 중간/끝에 나올 수도 있어, 페이지 전체 줄을 검사하고
제목이 있는 페이지는 그 줄 이전 본문만 남기고 잘라낸다(페이지 전체를 통째로 버리지 않음).
"""

import re

from preprocessing.page_text import Page

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references?|bibliography|참고\s*문헌)\s*[:.]?\s*(\[|$)",
    re.IGNORECASE,
)


def find_reference_start_page(pages: list[Page], manual_start_page: int | None = None) -> int | None:
    """References 절 제목이 있는 페이지 번호를 찾는다.

    manual_start_page가 지정되면(manifest.json의 reference_start_page) 그 값을 그대로 사용하고,
    없으면 각 페이지 전체 줄에서 "References" 류의 제목(그 줄에 다른 내용 없이 제목만 있는 경우)을
    자동 탐지한다. 해당 페이지의 제목 이전 내용은 본문으로 유지하고, 이후 페이지는 모두
    참고문헌 구간으로 간주한다(참고문헌 뒤에 부록이 있는 경우는 manifest에서
    reference_start_page를 직접 지정해 처리).
    """
    if manual_start_page is not None:
        return manual_start_page

    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        if any(REFERENCE_HEADING_RE.match(line) for line in lines):
            return page.page_number
    return None


def _text_before_reference_heading(text: str) -> str | None:
    """References 제목 줄 앞부분만 남긴다(해당 페이지의 실제 본문은 계속 색인 대상).

    제목 줄을 찾지 못하면 None을 반환한다(수동 지정된 경계 페이지 등 - 이 경우 페이지
    전체를 참고문헌 구간으로 간주해 통째로 제외한다).
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if REFERENCE_HEADING_RE.match(line.strip()):
            return "\n".join(lines[:index]).strip()
    return None


def drop_reference_pages(
    pages: list[Page], reference_start_page: int | None
) -> tuple[list[Page], list[Page]]:
    """색인 대상 본문 페이지와 제외된 참고문헌 페이지를 분리한다.

    제목이 있는 경계 페이지는 제목 이전 본문만 남겨 body_pages에 포함시키고,
    excluded_pages에는 원본(제목 포함) 페이지를 감사용으로 기록한다.
    """
    if reference_start_page is None:
        return list(pages), []

    body_pages: list[Page] = []
    excluded_pages: list[Page] = []

    for page in pages:
        if page.page_number < reference_start_page:
            body_pages.append(page)
        elif page.page_number == reference_start_page:
            excluded_pages.append(page)
            remaining_text = _text_before_reference_heading(page.text)
            if remaining_text:
                body_pages.append(Page(doc_id=page.doc_id, page_number=page.page_number, text=remaining_text))
        else:
            excluded_pages.append(page)

    return body_pages, excluded_pages
