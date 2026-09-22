"""표 추출: 캡션·각주 분리, 병합 셀 정규화, 마크다운 변환.

- 표는 캡션까지 하나의 단위로 묶는다(출처 확인용). 각주는 표 본문과 분리해 별도 필드로 보관한다.
- 병합 셀은 pdfplumber에서 빈 문자열/None으로 반환되는데, 그대로 두면 "키가 없는 빈 데이터"가
  되어 검색·해석이 어려워지므로 같은 열의 직전 값으로 채워(forward-fill) 단일 표로 정규화한다.
"""

import re
from dataclasses import dataclass

from preprocessing.chunker import CHUNK_SIZE, OVERLAP, split_long_text
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


def split_table_into_chunks(
    table: NormalizedTable, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[str]:
    """큰 표를 chunk_size 근처 크기로 나눈다 (설계서 1,200자/겹침 200자 기준).

    표 전체가 chunk_size 이하면 지금처럼 한 청크로 반환한다. 표가 더 크면(행이 많은 표)
    행 단위로 나누되, 매 조각마다 헤더 행(컬럼명 + 구분선)을 반복해서 붙여
    조각 하나만 봐도 어떤 열인지 알 수 있게 한다. 행 하나가 이미 chunk_size보다 큰
    비정상적인 경우에는 헤더/줄 구조를 신뢰할 수 없으므로 글자 단위로 나눈다(split_long_text).
    """
    full_text = table_chunk_text(table)
    if len(full_text) <= chunk_size:
        return [full_text]

    lines = table.markdown.split("\n")
    if len(lines) < 3:  # 헤더 + 구분선 + 본문 1행 미만이면 표 구조를 신뢰할 수 없음
        return split_long_text(full_text, chunk_size=chunk_size, overlap=overlap)

    header_block = "\n".join(lines[:2])  # "| col | ... |" + "| --- | ... |"
    body_lines = lines[2:]

    prefix_parts = [table.caption] if table.caption else []

    def render(rows: list[str]) -> str:
        table_text = header_block + ("\n" + "\n".join(rows) if rows else "")
        return "\n\n".join(prefix_parts + [table_text])

    pieces: list[str] = []
    current_rows: list[str] = []

    for row in body_lines:
        candidate_rows = current_rows + [row]
        if len(render(candidate_rows)) <= chunk_size or not current_rows:
            current_rows = candidate_rows
            continue

        pieces.append(render(current_rows))

        # 겹침: 방금 확정한 조각의 마지막 행들을 overlap 글자 수만큼 다음 조각 앞에 이어붙인다.
        overlap_rows: list[str] = []
        overlap_len = 0
        for prev_row in reversed(current_rows):
            overlap_rows.insert(0, prev_row)
            overlap_len += len(prev_row) + 1
            if overlap_len >= overlap:
                break
        current_rows = overlap_rows + [row]

    if current_rows:
        pieces.append(render(current_rows))

    if table.footnote:
        pieces[-1] = f"{pieces[-1]}\n\n{table.footnote}"

    return pieces
