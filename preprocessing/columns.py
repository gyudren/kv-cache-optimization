"""단(컬럼) 레이아웃 감지 및 읽기 순서 재구성.

논문 PDF는 한 페이지가 좌/우 2단으로 나뉜 경우가 많다. pdfplumber의 기본 추출은
y좌표(top) 순으로 단어를 나열하기 때문에 2단 문서에서는 좌/우 컬럼이 줄 단위로
섞여버린다. 이 모듈은 단어 좌표를 기준으로 2단 여부를 감지하고, 좌측 컬럼을
위→아래로 모두 읽은 뒤 우측 컬럼을 위→아래로 읽는 순서로 재조립한다.
"""

from preprocessing.loader import Word

LINE_Y_TOLERANCE = 3.0  # pt, 이 거리 이내의 단어는 같은 줄로 묶음
COLUMN_GAP_THRESHOLD = 15.0  # pt, 이보다 큰 단어 간 간격은 컬럼 사이 거터로 간주
TWO_COLUMN_ROW_RATIO_THRESHOLD = 0.4  # 거터 간격을 포함한 줄의 비율이 이 값 이상이면 2단으로 판정


def group_into_lines(words: list[Word]) -> list[list[Word]]:
    """단어를 y좌표 기준으로 같은 줄끼리 묶는다(컬럼 구분 없이)."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.top, w.x0))
    lines: list[list[Word]] = []
    current: list[Word] = []
    current_top = None
    for word in ordered:
        if current_top is None or abs(word.top - current_top) <= LINE_Y_TOLERANCE:
            current.append(word)
            current_top = word.top if current_top is None else min(current_top, word.top)
        else:
            lines.append(current)
            current = [word]
            current_top = word.top
    if current:
        lines.append(current)
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


def is_two_column(words: list[Word], page_width: float) -> bool:
    rows = group_into_lines(words)
    if not rows:
        return False

    gutter_rows = sum(1 for row in rows if _row_has_center_gutter_gap(row, page_width))
    ratio = gutter_rows / len(rows)
    return ratio >= TWO_COLUMN_ROW_RATIO_THRESHOLD


def _word_in_any_bbox(word: Word, bboxes: list[tuple[float, float, float, float]]) -> bool:
    for x0, top, x1, bottom in bboxes:
        if word.x0 >= x0 - 1 and word.x1 <= x1 + 1 and word.top >= top - 1 and word.bottom <= bottom + 1:
            return True
    return False


def _lines_to_text(words: list[Word]) -> str:
    lines = group_into_lines(words)
    lines.sort(key=line_top)
    return "\n".join(line_text(line) for line in lines).strip()


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

    if is_two_column(kept_words, page_width):
        center = page_width / 2
        left_words: list[Word] = []
        right_words: list[Word] = []
        for w in kept_words:
            if w.x1 <= center:
                left_words.append(w)
            elif w.x0 > center:
                right_words.append(w)
            else:
                # 중앙 거터를 가로지르는 단어(전체 폭 제목 등)는 겹침이 더 큰 쪽으로 배정
                left_overlap = max(0.0, center - w.x0)
                right_overlap = max(0.0, w.x1 - center)
                (left_words if left_overlap >= right_overlap else right_words).append(w)

        left_text = _lines_to_text(left_words)
        right_text = _lines_to_text(right_words)
        return f"{left_text}\n\n{right_text}".strip()

    return _lines_to_text(kept_words)
