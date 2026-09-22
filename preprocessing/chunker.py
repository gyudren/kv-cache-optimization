"""문단 경계를 보존하는 청킹 (1,200자 / 겹침 200자).

설계 산출물 B-3): "1,200자 / 겹침 200자 → 333개 청크 (수식·표 문단 경계 보존)".
문단(빈 줄로 구분되는 블록) 중간에서 자르지 않는 것을 최우선으로 하고,
그 다음으로 목표 청크 길이(chunk_size)와 겹침(overlap)을 맞춘다.
"""

import re
from dataclasses import dataclass

from preprocessing.loader import Page

CHUNK_SIZE = 1200
OVERLAP = 200

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


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


def chunk_document(
    doc_id: str,
    pages: list[Page],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    paragraphs = split_into_paragraphs(pages)

    chunks: list[Chunk] = []
    current_text = ""
    current_pages: set[int] = set()

    def flush() -> None:
        if not current_text:
            return
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_index=len(chunks),
                text=current_text.strip(),
                start_page=min(current_pages),
                end_page=max(current_pages),
            )
        )

    for paragraph, page_number in paragraphs:
        candidate = f"{current_text}\n\n{paragraph}" if current_text else paragraph

        # 문단 경계 보존: 현재 청크가 비어 있으면(=문단 하나가 chunk_size보다 커도)
        # 그 문단을 통째로 담아 수식·표 등이 중간에 잘리지 않게 한다.
        if len(candidate) <= chunk_size or not current_text:
            current_text = candidate
            current_pages.add(page_number)
            continue

        # 넣으면 chunk_size를 넘으므로 지금까지 쌓인 내용을 청크로 확정하고,
        # 겹침(overlap) 분량만 이어받아 다음 청크를 시작한다.
        flush()
        overlap_text = current_text[-overlap:] if overlap > 0 else ""
        current_text = f"{overlap_text}\n\n{paragraph}".strip() if overlap_text else paragraph
        current_pages = {page_number}

    flush()
    return chunks
