"""pypdf 기반 페이지 단위 PDF 파싱."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    doc_id: str
    page_number: int  # 1-indexed, 보고서 인용 [n, p.X]의 X와 동일한 번호 체계
    text: str


def load_pdf_pages(doc_id: str, pdf_path: Path) -> list[Page]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(Page(doc_id=doc_id, page_number=index, text=text))
    return pages
