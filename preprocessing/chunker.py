"""문단 경계를 보존하는 청킹 (1,200자 / 겹침 200자).

설계 산출물 B-3)의 "1,200자 / 겹침 200자" 기준을 적용한다.
문단(빈 줄로 구분되는 블록)을 중간에서 자르지 않는 것을 최우선으로 하고,
그 다음으로 목표 청크 길이(chunk_size)와 겹침(overlap)을 맞춘다.
따라서 단일 문단이 chunk_size보다 길면 해당 청크는 목표 길이를 초과할 수 있다.

페이지 경계에서 문장이 끊기는 경우(표 헤더 이월, 문장이 페이지를 넘어가는 경우 등)를
대비해, 문장부호로 끝나지 않는 페이지 마지막 문단은 다음 페이지 첫 문단과 이어붙인다.
이어붙이지 않은 채 넘어가는 애매한 경계는 review_flags로 남겨 사람이 확인하게 한다.
"""

import re
from dataclasses import dataclass

from preprocessing.page_text import Page

CHUNK_SIZE = 1200
OVERLAP = 200

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?。][\"'\)\]]*$")
_TABLE_LIKE_RE = re.compile(r"(?:\d[\d.,]*\s{2,}){2,}\d")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int


def split_into_paragraphs(pages: list[Page]) -> list[tuple[str, int]]:
    """페이지 순서를 유지하며 (문단 텍스트, 페이지 번호) 목록을 만든다."""
    paragraphs = []
    for page in pages:
        for raw_paragraph in _PARAGRAPH_SPLIT_RE.split(page.text):
            paragraph = raw_paragraph.strip()
            if paragraph:
                paragraphs.append((paragraph, page.page_number))
    return paragraphs


def _ends_mid_sentence(text: str) -> bool:
    return not bool(_SENTENCE_END_RE.search(text.strip()))


def _looks_like_table_row(text: str) -> bool:
    return bool(_TABLE_LIKE_RE.search(text))


def stitch_cross_page_paragraphs(
    paragraphs: list[tuple[str, int]],
) -> tuple[list[tuple[str, int, int]], list[dict]]:
    """페이지 경계에서 끊긴 문장을 이어 붙이고, 의심스러운 경계는 review_flags에 남긴다."""
    if not paragraphs:
        return [], []

    stitched: list[tuple[str, int, int]] = []
    review_flags: list[dict] = []

    text, start_page = paragraphs[0]
    end_page = start_page

    for next_text, next_page in paragraphs[1:]:
        crosses_page = next_page != end_page
        mid_sentence = _ends_mid_sentence(text)
        suspicious = crosses_page and (
            mid_sentence or _looks_like_table_row(text) or _looks_like_table_row(next_text)
        )

        if suspicious:
            review_flags.append(
                {
                    "page_boundary": [end_page, next_page],
                    "reason": "sentence_incomplete" if mid_sentence else "table_like_text",
                    "tail_preview": text[-80:],
                    "head_preview": next_text[:80],
                }
            )

        if crosses_page and mid_sentence:
            text = f"{text} {next_text}"
            end_page = next_page
        else:
            stitched.append((text, start_page, end_page))
            text, start_page = next_text, next_page
            end_page = start_page

    stitched.append((text, start_page, end_page))
    return stitched, review_flags


def chunk_document(
    doc_id: str,
    pages: list[Page],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> tuple[list[Chunk], list[dict]]:
    if chunk_size < 1:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap은 0 이상이고 chunk_size보다 작아야 합니다.")

    paragraphs = split_into_paragraphs(pages)
    stitched_paragraphs, review_flags = stitch_cross_page_paragraphs(paragraphs)
    for flag in review_flags:
        flag["doc_id"] = doc_id

    chunks: list[Chunk] = []
    current_text = ""
    current_start_page: int | None = None
    current_end_page: int | None = None

    def flush() -> None:
        if not current_text:
            return
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=len(chunks),
                text=current_text.strip(),
                start_page=current_start_page,
                end_page=current_end_page,
            )
        )

    for paragraph, start_page, end_page in stitched_paragraphs:
        if len(paragraph) > chunk_size:
            review_flags.append(
                {
                    "doc_id": doc_id,
                    "page_boundary": [start_page, end_page],
                    "reason": "paragraph_exceeds_chunk_size",
                    "char_count": len(paragraph),
                    "preview": paragraph[:160],
                }
            )

        candidate = f"{current_text}\n\n{paragraph}" if current_text else paragraph

        # 현재 청크가 비어 있으면 긴 단일 문단도 통째로 담아 문단 경계를 보존한다.
        if len(candidate) <= chunk_size or not current_text:
            current_text = candidate
            current_start_page = (
                start_page
                if current_start_page is None
                else min(current_start_page, start_page)
            )
            current_end_page = (
                end_page
                if current_end_page is None
                else max(current_end_page, end_page)
            )
            continue

        # 다음 문단을 넣으면 목표 크기를 넘으므로 현재 청크를 확정한다.
        # 검색 문맥 연결을 위해 끝부분 overlap자와 다음 문단을 새 청크에 함께 넣는다.
        carried_end_page = current_end_page
        flush()
        overlap_text = current_text[-overlap:] if overlap > 0 else ""
        if overlap_text:
            current_text = f"{overlap_text}\n\n{paragraph}".strip()
            current_start_page = carried_end_page
        else:
            current_text = paragraph
            current_start_page = start_page
        current_end_page = end_page

    flush()

    return chunks, review_flags
