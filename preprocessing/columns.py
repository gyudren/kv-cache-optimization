"""단(컬럼) 레이아웃 감지 및 읽기 순서 재구성.

논문 PDF는 한 페이지가 좌/우 2단으로 나뉜 경우가 많다. pdfplumber의 기본 추출은
y좌표(top) 순으로 단어를 나열하기 때문에 2단 문서에서는 좌/우 컬럼이 줄 단위로
섞여버린다. 이 모듈은 단어 좌표를 기준으로 2단 여부를 감지하고, 좌측 컬럼을
위→아래로 모두 읽은 뒤 우측 컬럼을 위→아래로 읽는 순서로 재조립한다.
"""

from preprocessing.headers_footers import BOTTOM_BAND_RATIO, TOP_BAND_RATIO
from preprocessing.loader import Word

LINE_Y_TOLERANCE = 3.0  # pt, 이 거리 이내의 단어는 같은 줄로 묶음
COLUMN_GAP_THRESHOLD = 15.0  # pt, 이보다 큰 단어 간 간격은 컬럼 사이 거터로 간주
TWO_COLUMN_ROW_RATIO_THRESHOLD = 0.4  # 거터 간격을 포함한 줄의 비율이 이 값 이상이면 2단으로 판정

MIN_COLUMN_SPLIT_GAP = 15.0  # pt, 실제 컬럼 거터는 보통 20~40pt 수준이라 이를 기준으로 잡음
MIN_COLUMN_SIDE_LINE_RATIO = 0.15  # 분리선 양쪽에 각각 최소 이 비율 이상의 줄이 있어야 함

# 이 논문들은 문단 사이에 빈 줄을 넣지 않고 "첫 줄 들여쓰기"로만 문단을 구분한다.
# 빈 줄 기준으로만 문단을 나누면 페이지 전체가 하나의 거대한 문단이 되어 청킹이
# 사실상 작동하지 않으므로, 컬럼의 일반적인 좌측 여백보다 들여써진 줄을 새 문단의
# 시작으로 보고 명시적으로 빈 줄을 삽입한다.
PARAGRAPH_INDENT_THRESHOLD = 6.0  # pt, 이 값 이상 더 들여써지면 새 문단 시작으로 간주


def _split_line_by_large_gaps(line_words: list[Word]) -> list[list[Word]]:
    """한 줄로 묶인 단어들 중 큰 간격이 있으면 서로 다른 컬럼이 우연히 같은 높이에서
    섞인 것으로 보고 다시 분리한다.

    좌/우 컬럼의 줄 높이가 우연히 일치하면 y좌표만으로는 하나의 줄로 묶여, 우측 컬럼의
    실제 시작 x좌표 정보가 사라져 버린다(컬럼 경계 탐지가 불가능해짐). 정상적인 단어
    간 간격(이 논문들은 약 2~3pt)보다 훨씬 큰 간격을 기준으로 미리 쪼개 이 정보를 보존한다.
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
    """단어를 y좌표 기준으로 같은 줄끼리 묶는다(컬럼 구분 없이).

    같은 높이에서 우연히 겹치는 서로 다른 컬럼의 단어는 큰 간격을 기준으로 다시 분리한다.
    """
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
    """한 줄(row) 안에서 단어 간 간격이 페이지 중앙을 가로지르며 크게 벌어지는지 확인한다.

    2단 레이아웃에서는 좌/우 컬럼의 줄 높이가 우연히 같아 같은 줄로 묶이더라도,
    두 컬럼 사이 거터만큼은 단어가 전혀 없는 빈 구간으로 남는다.
    """
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


MIN_WORDS_FOR_COLUMN_GAP_DETECTION = 3  # 페이지 번호·각주 기호 등 짧은 줄은 거터 탐지에서 제외


def _detect_column_split_x(lines: list[list[Word]]) -> float | None:
    """줄이 실제로 차지하는 가로 범위(x0~x1)를 모아 컬럼 분리선(빈 거터)을 찾는다.

    줄이 "시작하는" x좌표만 보면 왼쪽 컬럼의 긴 줄이 분리선 오른쪽까지 넘어와 있는
    경우를 놓친다(그 줄의 뒷부분 단어가 분리선 너머로 잘못 배정됨). 각 줄이 끝나는
    x좌표까지 함께 봐서, 어느 줄도 걸치지 않는 실제 빈 구간을 거터로 삼는다.

    페이지 번호·각주 기호처럼 단어 수가 아주 적은 줄은 우연히 두 컬럼 사이 애매한
    위치에 있어 실제 거터를 가리거나 엉뚱하게 이어붙일 수 있으므로 탐지에서 제외한다
    (분리선이 정해진 뒤 좌/우 배정에는 모든 단어가 그대로 사용된다).
    """
    substantial_lines = [line for line in lines if len(line) >= MIN_WORDS_FOR_COLUMN_GAP_DETECTION]
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


def _resolve_column_split(lines: list[list[Word]], page_width: float) -> float | None:
    """x0 분포 기반 분리선을 우선 사용하고, 못 찾으면 같은 줄 내 거터 간격 신호로 보완한다."""
    split_x = _detect_column_split_x(lines)
    if split_x is not None:
        return split_x
    if _gutter_row_ratio(lines, page_width) >= TWO_COLUMN_ROW_RATIO_THRESHOLD:
        return page_width / 2
    return None


def is_two_column(words: list[Word], page_width: float) -> bool:
    lines = group_into_lines(words)
    return _resolve_column_split(lines, page_width) is not None


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
    exclude_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> str:
    """단(컬럼) 레이아웃을 좌→우 다음 위→아래 읽기 순서로 재구성한다.

    표 영역(exclude_bboxes)에 속한 단어는 본문 텍스트에서 제외한다 — 표는 별도로
    캡션·각주와 함께 묶어 처리하므로(tables.py) 본문에 중복/파편화된 형태로 섞이면 안 된다.
    """
    exclude_bboxes = exclude_bboxes or []
    kept_words = [w for w in words if not _word_in_any_bbox(w, exclude_bboxes)]
    if not kept_words:
        return ""

    lines = group_into_lines(kept_words)
    split_x = _resolve_column_split(lines, page_width)

    if split_x is not None:
        left_words: list[Word] = []
        right_words: list[Word] = []
        for w in kept_words:
            if w.x1 <= split_x:
                left_words.append(w)
            elif w.x0 > split_x:
                right_words.append(w)
            else:
                # 분리선을 가로지르는 단어(전체 폭 제목 등)는 겹침이 더 큰 쪽으로 배정
                left_overlap = max(0.0, split_x - w.x0)
                right_overlap = max(0.0, w.x1 - split_x)
                (left_words if left_overlap >= right_overlap else right_words).append(w)

        left_text = _lines_to_text(left_words)
        right_text = _lines_to_text(right_words)
        return f"{left_text}\n\n{right_text}".strip()

    return _lines_to_text(kept_words)
