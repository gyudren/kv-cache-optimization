"""참고문헌 구간을 색인 대상에서 뺀다. 페이지 예산 계산은 budget.py에서 따로 한다.

References 제목은 2단 레이아웃에서 페이지 중간이나 끝에 나오기도 해서, 제목이 있는
페이지는 제목 앞 본문만 남긴다.
"""

import re

from preprocessing.page_text import Page

REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references?|bibliography|참고\s*문헌)\s*[:.]?\s*(\[|$)",
    re.IGNORECASE,
)


def find_reference_start_page(pages: list[Page], manual_start_page: int | None = None) -> int | None:
    """References 제목 줄이 있는 페이지 번호를 찾는다.

    manual_start_page(manifest.json의 reference_start_page)가 있으면 그 값을 쓴다.
    """
    if manual_start_page is not None:
        return manual_start_page

    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        if any(REFERENCE_HEADING_RE.match(line) for line in lines):
            return page.page_number
    return None


def _text_before_reference_heading(text: str) -> str | None:
    """References 제목 줄 앞 텍스트를 돌려준다.

    제목 줄이 없으면(수동 지정한 경계 페이지 등) None이고, 그 페이지는 통째로 제외된다.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if REFERENCE_HEADING_RE.match(line.strip()):
            return "\n".join(lines[:index]).strip()
    return None


def drop_reference_pages(
    pages: list[Page],
    reference_start_page: int | None,
    reference_end_page: int | None = None,
) -> tuple[list[Page], list[Page]]:
    """색인할 본문 페이지와 제외한 참고문헌 페이지를 나눈다.

    경계 페이지는 제목 앞 본문만 body_pages에 넣고, 원본은 excluded_pages에 남긴다.
    reference_end_page 뒤 페이지는 다시 본문으로 본다. 참고문헌 뒤에 부록이 오는
    논문(예: DeepSeek-V2)을 위해 manifest.json에서 지정한다.
    """
    if reference_start_page is None:
        return list(pages), []

    body_pages: list[Page] = []
    excluded_pages: list[Page] = []

    for page in pages:
        after_references = (
            reference_end_page is not None and page.page_number > reference_end_page
        )
        if page.page_number < reference_start_page or after_references:
            body_pages.append(page)
        elif page.page_number == reference_start_page:
            excluded_pages.append(page)
            remaining_text = _text_before_reference_heading(page.text)
            if remaining_text:
                body_pages.append(Page(doc_id=page.doc_id, page_number=page.page_number, text=remaining_text))
        else:
            excluded_pages.append(page)

    return body_pages, excluded_pages
