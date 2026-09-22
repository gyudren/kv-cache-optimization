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
    vector_shape_count: int  # 사각형/선/곡선 등 벡터 드로잉 요소 수(아키텍처 다이어그램 탐지용)
    rejected_region_bboxes: list[tuple[float, float, float, float]]  # 표로 보였지만 기각된 영역(대부분 다이어그램)


# pdfplumber 기본값(3pt)은 이 논문들의 실제 단어 간 간격(약 2.2pt, LaTeX 양쪽 정렬 조판)보다
# 커서 한 줄 전체가 하나의 "단어"로 뭉쳐버린다(공백이 다 사라짐). 실측 간격보다 작게 낮춰서
# 단어 경계를 올바르게 분리한다.
WORD_X_TOLERANCE = 1.5

# find_tables()는 선(line)만 있으면 표로 인식하므로, 아키텍처/블록 다이어그램처럼 테두리
# 상자가 있는 그림도 "표"로 오탐지한다. 실제 데이터 표처럼 보이는지 최소한으로 검증해
# 다이어그램을 표로 잘못 뽑아 본문에서 통째로 잘라내는 것을 막는다.
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
                # 다이어그램/차트 안의 여러 줄 라벨이 한 셀에 뭉쳐 들어온 경우
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
                    # 표 후보였지만 기각됨(대부분 아키텍처 다이어그램/차트) - 표로 만들지는
                    # 않되, 라벨 텍스트가 본문 문장 사이에 끼어들어 뒤섞이지 않도록
                    # 본문 재구성에서는 제외한다.
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
