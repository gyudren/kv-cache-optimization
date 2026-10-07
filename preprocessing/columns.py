"""2단 레이아웃을 감지해 읽기 순서를 다시 맞춘다.

pdfplumber는 y좌표 순으로 단어를 나열해서 2단 문서에서는 좌우 컬럼이 줄 단위로 섞인다.
왼쪽 컬럼을 끝까지 읽은 뒤 오른쪽 컬럼을 읽는 순서로 다시 붙인다.
"""

from preprocessing.headers_footers import BOTTOM_BAND_RATIO, TOP_BAND_RATIO
from preprocessing.loader import Word

LINE_Y_TOLERANCE = 3.0  # pt, 이 안이면 같은 줄
COLUMN_GAP_THRESHOLD = 15.0  # pt, 이보다 넓은 단어 간격은 컬럼 거터로 본다
TWO_COLUMN_ROW_RATIO_THRESHOLD = 0.4  # 거터가 있는 줄 비율이 이 이상이면 2단

MIN_COLUMN_SPLIT_GAP = 15.0  # pt, 실제 거터는 보통 20~40pt
MIN_COLUMN_SIDE_LINE_RATIO = 0.15  # 분리선 양쪽에 각각 있어야 하는 최소 줄 비율

# 이 논문들은 빈 줄 없이 첫 줄 들여쓰기로만 문단을 나눠서, 들여쓴 줄 앞에 빈 줄을 넣는다.
PARAGRAPH_INDENT_THRESHOLD = 6.0  # pt


def _split_line_by_large_gaps(line_words: list[Word]) -> list[list[Word]]:
    """같은 높이에 걸린 좌우 컬럼 단어를 큰 간격 기준으로 다시 나눈다.

    y좌표로만 묶으면 오른쪽 컬럼의 시작 x좌표가 사라져 컬럼 경계를 찾을 수 없다.
    """
    if len(line_words) < 2:
        return [line_words]
    ordered = sorted(line_words, key=lambda w: w.x0)
    groups: list[list[Word]] = [[ordered[0]]]
    for prev_word, next_word in zip(ordered, ordered[1:]):
        if (next_word.x0 - prev_word.x1) >= COLUMN_GAP_THRESHOLD:
            groups.append([])
        groups[-1].append(next_word)
    return groups


def group_into_lines(words: list[Word]) -> list[list[Word]]:
    """단어를 y좌표 기준으로 줄 단위로 묶는다. 같은 높이의 다른 컬럼 단어는 다시 나눈다."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.top, w.x0))
    raw_lines: list[list[Word]] = []
    current: list[Word] = []
    current_top = None
    for word in ordered:
        if current_top is None or abs(word.top - current_top) <= LINE_Y_TOLERANCE:
            current.append(word)
            current_top = word.top if current_top is None else min(current_top, word.top)
        else:
            raw_lines.append(current)
            current = [word]
            current_top = word.top
    if current:
        raw_lines.append(current)

    lines: list[list[Word]] = []
    for raw_line in raw_lines:
        lines.extend(_split_line_by_large_gaps(raw_line))
    return lines


def line_text(line_words: list[Word]) -> str:
    return " ".join(w.text for w in sorted(line_words, key=lambda w: w.x0))


def line_top(line_words: list[Word]) -> float:
    return min(w.top for w in line_words)


def _row_has_center_gutter_gap(row_words: list[Word], page_width: float) -> bool:
    """줄 안에 페이지 중앙을 가로지르는 넓은 빈 간격(컬럼 거터)이 있는지 본다."""
    if len(row_words) < 2:
        return False
    ordered = sorted(row_words, key=lambda w: w.x0)
    center = page_width / 2
    for prev_word, next_word in zip(ordered, ordered[1:]):
        gap = next_word.x0 - prev_word.x1
        if gap >= COLUMN_GAP_THRESHOLD and prev_word.x1 <= center <= next_word.x0:
            return True
    return False


def _gutter_row_ratio(lines: list[list[Word]], page_width: float) -> float:
    if not lines:
        return 0.0
    gutter_rows = sum(1 for row in lines if _row_has_center_gutter_gap(row, page_width))
    return gutter_rows / len(lines)


MIN_WORDS_FOR_COLUMN_GAP_DETECTION = 3  # 페이지 번호 같은 짧은 줄은 거터 탐지에서 뺀다


def _detect_column_split_x(lines: list[list[Word]], page_height: float) -> float | None:
    """줄들의 x 범위를 합쳐, 어느 줄도 걸치지 않는 빈 구간을 컬럼 분리선으로 잡는다.

    시작 x좌표만 보면 분리선을 넘어가는 왼쪽 컬럼의 긴 줄을 놓친다. 짧은 줄과 상·하단
    여백의 running header는 거터를 가리거나 두 컬럼을 이어 버릴 수 있어 탐지에서만 뺀다.
    """
    top_cutoff = page_height * TOP_BAND_RATIO
    bottom_cutoff = page_height * (1 - BOTTOM_BAND_RATIO)
    substantial_lines = [
        line
        for line in lines
        if len(line) >= MIN_WORDS_FOR_COLUMN_GAP_DETECTION and top_cutoff < line_top(line) < bottom_cutoff
    ]
    if len(substantial_lines) < 4:
        return None

    intervals = sorted((_line_x0(line), max(w.x1 for w in line)) for line in substantial_lines)

    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    if len(merged) < 2:
        return None

    best_gap = 0.0
    best_split = None
    for (_, prev_x1), (next_x0, _) in zip(merged, merged[1:]):
        gap = next_x0 - prev_x1
        if gap > best_gap:
            best_gap = gap
            best_split = (prev_x1 + next_x0) / 2

    if best_split is None or best_gap < MIN_COLUMN_SPLIT_GAP:
        return None

    left_count = sum(1 for _, x1 in intervals if x1 <= best_split)
    right_count = sum(1 for x0, _ in intervals if x0 >= best_split)
    min_side = max(1, int(len(intervals) * MIN_COLUMN_SIDE_LINE_RATIO))
    if left_count < min_side or right_count < min_side:
        return None

    return best_split


def _resolve_column_split(lines: list[list[Word]], page_width: float, page_height: float) -> float | None:
    """분리선을 먼저 찾고, 없으면 중앙 거터가 있는 줄 비율로 2단 여부를 정한다."""
    split_x = _detect_column_split_x(lines, page_height)
    if split_x is not None:
        return split_x
    if _gutter_row_ratio(lines, page_width) >= TWO_COLUMN_ROW_RATIO_THRESHOLD:
        return page_width / 2
    return None


def is_two_column(words: list[Word], page_width: float, page_height: float | None = None) -> bool:
    lines = group_into_lines(words)
    resolved_page_height = page_height if page_height is not None else page_width * 1.4142
    return _resolve_column_split(lines, page_width, resolved_page_height) is not None


def _word_in_any_bbox(word: Word, bboxes: list[tuple[float, float, float, float]]) -> bool:
    for x0, top, x1, bottom in bboxes:
        if word.x0 >= x0 - 1 and word.x1 <= x1 + 1 and word.top >= top - 1 and word.bottom <= bottom + 1:
            return True
    return False


def _line_x0(line_words: list[Word]) -> float:
    return min(w.x0 for w in line_words)


def _column_baseline_x0(lines: list[list[Word]]) -> float:
    """들여쓰기 없는 일반 줄들의 좌측 여백(가장 흔한 x0)을 구한다."""
    if not lines:
        return 0.0
    rounded_x0_counts: dict[int, int] = {}
    for line in lines:
        rounded = round(_line_x0(line))
        rounded_x0_counts[rounded] = rounded_x0_counts.get(rounded, 0) + 1
    return float(max(rounded_x0_counts, key=lambda x0: rounded_x0_counts[x0]))


def _lines_to_text(words: list[Word]) -> str:
    lines = group_into_lines(words)
    lines.sort(key=line_top)
    if not lines:
        return ""

    baseline_x0 = _column_baseline_x0(lines)

    parts = [line_text(lines[0])]
    for line in lines[1:]:
        is_new_paragraph = (_line_x0(line) - baseline_x0) >= PARAGRAPH_INDENT_THRESHOLD
        separator = "\n\n" if is_new_paragraph else "\n"
        parts.append(separator)
        parts.append(line_text(line))
    return "".join(parts).strip()


def reconstruct_reading_order_text(
    words: list[Word],
    page_width: float,
    page_height: float | None = None,
    exclude_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> str:
    """왼쪽 컬럼, 오른쪽 컬럼 순으로 읽어 페이지 텍스트를 만든다.

    exclude_bboxes 안의 단어는 뺀다. 표는 tables.py에서 캡션·각주와 함께 따로 처리한다.
    """
    exclude_bboxes = exclude_bboxes or []
    kept_words = [w for w in words if not _word_in_any_bbox(w, exclude_bboxes)]
    if not kept_words:
        return ""

    resolved_page_height = page_height if page_height is not None else page_width * 1.4142
    lines = group_into_lines(kept_words)
    split_x = _resolve_column_split(lines, page_width, resolved_page_height)

    if split_x is not None:
        left_words: list[Word] = []
        right_words: list[Word] = []
        for w in kept_words:
            if w.x1 <= split_x:
                left_words.append(w)
            elif w.x0 > split_x:
                right_words.append(w)
            else:
                # 분리선에 걸친 단어(전체 폭 제목 등)는 더 많이 걸친 쪽에 넣는다
                left_overlap = max(0.0, split_x - w.x0)
                right_overlap = max(0.0, w.x1 - split_x)
                (left_words if left_overlap >= right_overlap else right_words).append(w)

        left_text = _lines_to_text(left_words)
        right_text = _lines_to_text(right_words)
        return f"{left_text}\n\n{right_text}".strip()

    return _lines_to_text(kept_words)
