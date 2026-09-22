"""표 추출: 캡션·각주 분리, 병합 셀 정규화, 마크다운 변환.

- 표는 캡션까지 하나의 단위로 묶는다(출처 확인용). 각주는 표 본문과 분리해 별도 필드로 보관한다.
- 병합 셀은 pdfplumber에서 빈 문자열/None으로 반환되는데, 그대로 두면 "키가 없는 빈 데이터"가
  되어 검색·해석이 어려워지므로 같은 열의 직전 값으로 채워(forward-fill) 단일 표로 정규화한다.
"""

import re
from dataclasses import dataclass

from preprocessing.columns import group_into_lines, line_text, line_top
from preprocessing.loader import TableBlock, Word

CAPTION_RE = re.compile(r"^\s*(table|표)\s*\d+", re.IGNORECASE)
FOOTNOTE_START_RE = re.compile(r"^\s*(\*|†|주\s*[:：]|note\s*[:：])", re.IGNORECASE)
CAPTION_SEARCH_MARGIN = 40.0  # pt, 표 위/아래로 이 거리 이내에서 캡션·각주를 찾는다


@dataclass(frozen=True)
class NormalizedTable:
    doc_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    caption: str | None
    footnote: str | None
    markdown: str


def _forward_fill_rows(rows: list[list[str | None]]) -> list[list[str]]:
    """세로로 병합된 셀(rowspan)이 None/빈 문자열로 내려오는 것을 같은 열의 직전 값으로 채운다.

    예: 카테고리 라벨이 여러 행에 걸쳐 병합된 표에서, 병합으로 비어 보이는 하위 행의
    셀이 "키가 없는 빈 데이터"가 되지 않도록 바로 위 행(같은 열)의 값을 이어받는다.
    """
    if not rows:
        return []
    num_cols = max(len(row) for row in rows)
    last_values = [""] * num_cols

    filled_rows = []
    for row in rows:
        filled_row = []
        for col_index in range(num_cols):
            cell = row[col_index] if col_index < len(row) else None
            value = (cell or "").strip()
            if value:
                last_values[col_index] = value
            else:
                value = last_values[col_index]
            filled_row.append(value)
        filled_rows.append(filled_row)
    return filled_rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in body:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        else:
            row = row[: len(header)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _find_nearby_line(
    lines_with_top: list[tuple[float, str]], y_reference: float, direction: str, pattern: re.Pattern
) -> str | None:
    candidates = []
    for top, text in lines_with_top:
        distance = (y_reference - top) if direction == "above" else (top - y_reference)
        if 0 <= distance <= CAPTION_SEARCH_MARGIN:
            candidates.append((distance, text))
    candidates.sort(key=lambda item: item[0])
    for _, text in candidates:
        if pattern.match(text.strip()):
            return text.strip()
    return None


def extract_tables_for_page(
    doc_id: str, page_number: int, words: list[Word], tables: list[TableBlock]
) -> list[NormalizedTable]:
    if not tables:
        return []

    all_lines = group_into_lines(words)
    lines_with_top = [(line_top(line), line_text(line)) for line in all_lines]

    normalized_tables = []
    for table in tables:
        _x0, top, _x1, bottom = table.bbox
        caption = _find_nearby_line(lines_with_top, top, "above", CAPTION_RE)
        footnote = _find_nearby_line(lines_with_top, bottom, "below", FOOTNOTE_START_RE)

        filled_rows = _forward_fill_rows(table.rows)
        markdown = _rows_to_markdown(filled_rows)

        normalized_tables.append(
            NormalizedTable(
                doc_id=doc_id,
                page_number=page_number,
                bbox=table.bbox,
                caption=caption,
                footnote=footnote,
                markdown=markdown,
            )
        )
    return normalized_tables


def table_chunk_text(table: NormalizedTable) -> str:
    """표 청크 본문: 캡션 + 마크다운 표 + 각주를 하나의 텍스트로 묶는다."""
    parts = []
    if table.caption:
        parts.append(table.caption)
    parts.append(table.markdown)
    if table.footnote:
        parts.append(table.footnote)
    return "\n\n".join(parts)
