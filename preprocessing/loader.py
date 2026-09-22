"""pdfplumber 기반 페이지 단위 PDF 파싱.

pypdf의 단순 extract_text() 대신 pdfplumber로 단어 좌표(words), 표(tables),
이미지(images) 정보를 모두 읽어온다. 2단(컬럼) 레이아웃 재정렬, header/footer 제거,
표 캡션/각주 분리는 이 좌표 정보를 기반으로 이후 단계(columns.py, headers_footers.py,
tables.py)에서 처리한다.
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


@dataclass(frozen=True)
class TableBlock:
    bbox: tuple[float, float, float, float]  # (x0, top, x1, bottom)
    rows: list[list[str | None]]


@dataclass(frozen=True)
class ImageBlock:
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class RawPage:
    doc_id: str
    page_number: int  # 1-indexed, 보고서 인용 [n, p.X]의 X와 동일한 번호 체계
    width: float
    height: float
    words: list[Word]
    tables: list[TableBlock]
    images: list[ImageBlock]


def load_pdf_raw_pages(doc_id: str, pdf_path: Path) -> list[RawPage]:
    raw_pages: list[RawPage] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            words = [
                Word(text=w["text"], x0=w["x0"], x1=w["x1"], top=w["top"], bottom=w["bottom"])
                for w in page.extract_words()
            ]

            tables = []
            for table in page.find_tables():
                rows = table.extract()
                if rows:
                    tables.append(TableBlock(bbox=table.bbox, rows=rows))

            images = [
                ImageBlock(bbox=(img["x0"], img["top"], img["x1"], img["bottom"]))
                for img in page.images
            ]

            raw_pages.append(
                RawPage(
                    doc_id=doc_id,
                    page_number=index,
                    width=page.width,
                    height=page.height,
                    words=words,
                    tables=tables,
                    images=images,
                )
            )
    return raw_pages
