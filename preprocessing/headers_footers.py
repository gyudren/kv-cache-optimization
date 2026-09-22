"""반복되는 머리글/바닥글(header/footer) 탐지 및 제거.

같은 문구(페이지 번호 등 숫자만 다른 경우 포함)가 여러 페이지의 상단/하단 여백에서
반복되면 header/footer로 간주해 본문 청크에서 제외한다. 그대로 두면 청크마다 같은
문구가 반복 삽입되어 검색 시 실제 내용이 없는 header/footer 청크가 우선순위로
잡히는 문제가 생길 수 있다. 제거된 줄은 감사(audit)를 위해 그대로 버리지 않고 로그로 남긴다.
"""

import re
from collections import Counter
from dataclasses import dataclass

TOP_BAND_RATIO = 0.08
BOTTOM_BAND_RATIO = 0.08
MIN_REPEAT_RATIO = 0.4  # 전체 페이지 중 이 비율 이상 반복되면 header/footer로 판정

_DIGIT_RE = re.compile(r"\d+")


def _normalize(line: str) -> str:
    """페이지 번호처럼 페이지마다 달라지는 숫자를 지워서 비교한다."""
    return _DIGIT_RE.sub("#", line).strip().lower()


@dataclass(frozen=True)
class PageBandLines:
    page_number: int
    top_lines: list[str]
    bottom_lines: list[str]


def collect_band_lines(
    page_number: int, page_height: float, line_tops_and_texts: list[tuple[float, str]]
) -> PageBandLines:
    top_cutoff = page_height * TOP_BAND_RATIO
    bottom_cutoff = page_height * (1 - BOTTOM_BAND_RATIO)
    top_lines = [text for top, text in line_tops_and_texts if top <= top_cutoff and text.strip()]
    bottom_lines = [text for top, text in line_tops_and_texts if top >= bottom_cutoff and text.strip()]
    return PageBandLines(page_number=page_number, top_lines=top_lines, bottom_lines=bottom_lines)


def detect_boilerplate_lines(band_lines_per_page: list[PageBandLines]) -> set[str]:
    """문서 전체에서 반복되는 정규화된 header/footer 줄 집합을 반환한다."""
    total_pages = len(band_lines_per_page)
    if total_pages == 0:
        return set()

    counter: Counter[str] = Counter()
    for band in band_lines_per_page:
        seen_this_page = {_normalize(line) for line in band.top_lines + band.bottom_lines}
        counter.update(seen_this_page)

    threshold = max(2, int(total_pages * MIN_REPEAT_RATIO))
    return {normalized for normalized, count in counter.items() if count >= threshold}


def strip_boilerplate_lines(text: str, boilerplate: set[str]) -> tuple[str, list[str]]:
    """본문 텍스트에서 boilerplate로 판정된 줄을 제거한다.

    반환값: (제거 후 텍스트, 제거된 원본 줄 목록 — summary에 기록해 감사 가능하게 함)
    """
    if not boilerplate:
        return text, []

    kept_lines = []
    removed_lines = []
    for line in text.splitlines():
        if line.strip() and _normalize(line) in boilerplate:
            removed_lines.append(line)
        else:
            kept_lines.append(line)
    return "\n".join(kept_lines), removed_lines
