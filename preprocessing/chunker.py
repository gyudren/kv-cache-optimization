"""문단 경계를 지키면서 1,200자 단위(겹침 200자)로 청킹한다.

문장부호로 끝나지 않는 페이지 마지막 문단은 다음 페이지 첫 문단과 잇고,
판단이 애매한 페이지 경계는 review_flags로 남긴다.
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


def split_long_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """빈 줄로 나뉘지 않는 긴 덩어리(수식 섞인 문단, 큰 표 등)를 chunk_size 단위로 자른다.

    가능하면 공백·개행에서 자르고, 근처에 경계가 없으면 글자 단위로 자른다.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    n = len(text)
    # 공백·개행을 찾을 때 목표 지점에서 거슬러 볼 최대 길이
    boundary_search_window = min(80, chunk_size // 4)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            search_from = max(start, end - boundary_search_window)
            space_at = text.rfind(" ", search_from, end)
            newline_at = text.rfind("\n", search_from, end)
            boundary = max(space_at, newline_at)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= n:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end

    return pieces


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
    paragraphs = split_into_paragraphs(pages)
    stitched_paragraphs, review_flags = stitch_cross_page_paragraphs(paragraphs)
    for flag in review_flags:
        flag["doc_id"] = doc_id

    # 긴 문단을 미리 잘라 두면 아래 누적 루프에서 한 문단이 chunk_size를 넘는 일이 없다.
    split_paragraphs: list[tuple[str, int, int]] = []
    for paragraph, start_page, end_page in stitched_paragraphs:
        for piece in split_long_text(paragraph, chunk_size=chunk_size, overlap=overlap):
            split_paragraphs.append((piece, start_page, end_page))
    stitched_paragraphs = split_paragraphs

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
        candidate = f"{current_text}\n\n{paragraph}" if current_text else paragraph

        # 문단은 이미 chunk_size 이하라서 새 청크의 첫 문단은 그대로 담는다.
        if len(candidate) <= chunk_size or not current_text:
            current_text = candidate
            current_start_page = start_page if current_start_page is None else min(current_start_page, start_page)
            current_end_page = end_page if current_end_page is None else max(current_end_page, end_page)
            continue

        # 넘치면 지금까지를 청크로 확정하고 끝부분 overlap만큼 이어받는다.
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
