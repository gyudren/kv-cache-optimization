"""페이지 단위로 재조립이 끝난 순수 텍스트 컨테이너.

loader.RawPage(좌표·표·이미지 포함)를 columns/headers_footers 단계로 가공한 뒤
남는, 참고문헌 판별과 청킹에서 사용할 최종 페이지 텍스트 표현이다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    doc_id: str
    page_number: int  # 1-indexed, 보고서 인용 [n, p.X]의 X와 동일한 번호 체계
    text: str
