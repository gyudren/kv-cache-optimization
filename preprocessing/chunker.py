"""문단 경계를 보존하는 청킹 (1,200자 / 겹침 200자).

설계 산출물 B-3): "1,200자 / 겹침 200자 → 333개 청크 (수식·표 문단 경계 보존)".
문단(빈 줄로 구분되는 블록) 중간에서 자르지 않는 것을 최우선으로 하고,
그 다음으로 목표 청크 길이(chunk_size)와 겹침(overlap)을 맞춘다.

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


def split_long_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """chunk_size보다 긴 텍스트 덩어리를 chunk_size 단위(겹침 overlap)로 강제 분할한다.

    빈 줄로 나뉘지 않는 긴 문단(수식이 섞인 본문, 표처럼 보이는 텍스트 등)이나
    큰 표 마크다운처럼, 자연스러운 문단 경계가 없어서 그대로 두면 한 청크가
    수천~수만 자까지 커지는 경우에 사용한다. 가능하면 단어/줄 경계(공백, 개행)에서
    자르고, 근처에 마땅한 경계가 없으면 어쩔 수 없이 글자 단위로 자른다.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    n = len(text)
    # 자연스러운 경계(공백/개행)를 찾을 때 목표 지점에서 뒤로 얼마나 물러나 볼지
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
        # 다음 조각은 overlap 만큼 뒤에서부터 시작 (겹침 유지)
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

    # chunk_size보다 긴 문단(수식·표처럼 빈 줄로 안 나뉘는 덩어리, 페이지 경계 이어붙이기로
    # 커진 문단 등)은 이 시점에 미리 chunk_size 단위로 쪼개 둔다. 이렇게 해두면 아래
    # 누적 루프에 들어오는 문단은 항상 chunk_size 이하이므로, "문단 하나가 통째로
    # chunk_size를 훨씬 넘는 청크가 되는" 문제 없이 설계서대로 1,200자/겹침 200자를 지킬 수 있다.
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

        # 문단 경계 보존: 위에서 이미 모든 문단을 chunk_size 이하로 쪼개뒀으므로,
        # current_text가 비어 있을 때(=새 청크 시작) 들어오는 문단은 항상 그대로 담아도 안전하다.
        if len(candidate) <= chunk_size or not current_text:
            current_text = candidate
            current_start_page = start_page if current_start_page is None else min(current_start_page, start_page)
            current_end_page = end_page if current_end_page is None else max(current_end_page, end_page)
            continue

        # 넣으면 chunk_size를 넘으므로 지금까지 쌓인 내용을 청크로 확정하고,
        # 겹침(overlap) 분량만 이어받아 다음 청크를 시작한다.
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
