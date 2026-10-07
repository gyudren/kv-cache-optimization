"""pdfplumber로 PDF를 페이지 단위로 읽는다.

뒤 단계(컬럼 재정렬, 머리글 제거, 표 분리)에서 쓸 단어 좌표·표·이미지 정보를 함께 읽어 둔다.
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
    page_number: int  # 1부터 시작, 보고서 인용 [n, p.X]의 X
    width: float
    height: float
    words: list[Word]
    tables: list[TableBlock]
    images: list[ImageBlock]
    vector_shape_count: int  # 벡터 도형 수, 다이어그램 탐지용
    rejected_region_bboxes: list[tuple[float, float, float, float]]  # 표 후보에서 기각된 영역, 주로 다이어그램


# pdfplumber 기본값 3pt는 이 논문들의 단어 간격(약 2.2pt)보다 커서 한 줄이 한 단어로 뭉친다.
WORD_X_TOLERANCE = 1.5

# find_tables()는 테두리 있는 다이어그램도 표로 잡으므로 최소한의 표 모양을 확인한다.
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
MIN_TABLE_NON_EMPTY_CELL_RATIO = 0.5
MAX_TABLE_NEWLINES_PER_CELL = 4


def _looks_like_real_table(rows: list[list[str | None]]) -> bool:
    if len(rows) < MIN_TABLE_ROWS:
        return False
    num_cols = max((len(row) for row in rows), default=0)
    if num_cols < MIN_TABLE_COLS:
        return False

    total_cells = 0
    non_empty_cells = 0
    for row in rows:
        for cell in row:
            total_cells += 1
            text = (cell or "").strip()
            if not text:
                continue
            non_empty_cells += 1
            if text.count("\n") > MAX_TABLE_NEWLINES_PER_CELL:
                # 다이어그램 라벨이 한 셀에 뭉친 경우
                return False

    if total_cells == 0:
        return False
    return (non_empty_cells / total_cells) >= MIN_TABLE_NON_EMPTY_CELL_RATIO


def load_pdf_raw_pages(doc_id: str, pdf_path: Path) -> list[RawPage]:
    raw_pages: list[RawPage] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            words = [
                Word(text=w["text"], x0=w["x0"], x1=w["x1"], top=w["top"], bottom=w["bottom"])
                for w in page.extract_words(x_tolerance=WORD_X_TOLERANCE)
            ]

            tables = []
            rejected_region_bboxes = []
            for table in page.find_tables():
                rows = table.extract()
                if rows and _looks_like_real_table(rows):
                    tables.append(TableBlock(bbox=table.bbox, rows=rows))
                else:
                    # 기각된 영역의 라벨이 본문 사이에 끼지 않도록 본문 재구성에서 뺀다.
                    rejected_region_bboxes.append(table.bbox)

            images = [
                ImageBlock(bbox=(img["x0"], img["top"], img["x1"], img["bottom"]))
                for img in page.images
            ]

            vector_shape_count = len(page.rects) + len(page.lines) + len(page.curves)

            raw_pages.append(
                RawPage(
                    doc_id=doc_id,
                    page_number=index,
                    width=page.width,
                    height=page.height,
                    words=words,
                    tables=tables,
                    images=images,
                    vector_shape_count=vector_shape_count,
                    rejected_region_bboxes=rejected_region_bboxes,
                )
            )
    return raw_pages
