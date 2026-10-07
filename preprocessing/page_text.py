"""컬럼 재정렬과 머리글/바닥글 제거를 마친 페이지 텍스트."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    doc_id: str
    page_number: int  # 1부터 시작, 보고서 인용 [n, p.X]의 X
    text: str
