"""Research PDF loading, source page provenance and bounded indexing."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal
from ..config import DECLARED_PAGES, MAX_PAGES, CHUNK_SIZE, CHUNK_OVERLAP

@dataclass(frozen=True)
class PaperSpec:
    doc_id: str
    path: str
    technology: Literal["mla", "itme", "baseline"]
    declared_pages: int
    citation_number: int


def paper_manifest(directory: Path) -> list[PaperSpec]:
    return [
        PaperSpec("deepseek_v2", str(directory / "deepseek_v2.pdf"), "mla", 52, 1),
        PaperSpec("itme", str(directory / "itme.pdf"), "itme", 13, 2),
        PaperSpec("infinigen", str(directory / "infinigen.pdf"), "baseline", 18, 3),
        PaperSpec("cxl_pnm", str(directory / "cxl_pnm.pdf"), "baseline", 13, 4),
    ]


def bibliography_starts(text: str) -> bool:
    """Only treat a page as bibliography when it begins with a reference heading.

    Footnotes or paragraphs merely mentioning references must not be discarded.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return any(bool(re.match(r"^(?:\d+[.\s]+)?(?:references|bibliography)\s*$", line, re.I)) for line in lines[:5])


def load_papers(manifest: list[PaperSpec]) -> list[dict]:
    if len(manifest) != 4 or {p.doc_id for p in manifest} != set(DECLARED_PAGES):
        raise ValueError("Exactly the four approved research papers must be supplied")
    from pypdf import PdfReader
    pages: list[dict] = []
    total = 0
    for paper in manifest:
        path = Path(paper.path)
        if not path.is_file():
            raise FileNotFoundError(f"Required original paper missing: {path}")
        try:
            pdf = PdfReader(str(path))
            size = len(pdf.pages)
        except Exception as exc:
            raise ValueError(f"PDF cannot be parsed: {paper.doc_id}: {exc}") from exc
        total += size
        if total > MAX_PAGES:
            raise ValueError(f"Research corpus exceeds {MAX_PAGES} pages: {total}")
        excludes = False
        for index, pdf_page in enumerate(pdf.pages, 1):
            try:
                extracted = pdf_page.extract_text() or ""
            except Exception as exc:
                raise ValueError(f"Cannot extract {paper.doc_id} page {index}: {exc}") from exc
            if bibliography_starts(extracted):
                excludes = True
            pages.append({
                "doc_id": paper.doc_id, "page": index, "text": extracted,
                "technology": paper.technology, "citation_number": paper.citation_number,
                "excluded": excludes, "actual_pages": size, "declared_pages": paper.declared_pages,
            })
    if not any(not page["excluded"] and page["text"].strip() for page in pages):
        raise ValueError("No extractable research pages after bibliography filtering")
    return pages


def chunk_pages(pages: list[dict], size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size and overlap")
    chunks: list[dict] = []
    for page in pages:
        if page.get("excluded"):
            continue
        text = page["text"].strip()
        if not text:
            continue
        begin = 0
        while begin < len(text):
            end = min(len(text), begin + size)
            # Prefer paragraph/sentence break, but never discard material.
            if end < len(text):
                window = text[begin:end]
                break_pos = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
                if break_pos >= size // 2:
                    end = begin + break_pos + (2 if window[break_pos:break_pos+2] in ("\n\n", ". ") else 1)
            if end <= begin:
                raise AssertionError("chunking failed to advance")
            chunks.append({
                "chunk_id": f'{page["doc_id"]}:p{page["page"]}:o{begin}',
                "doc_id": page["doc_id"], "page": page["page"],
                "text": text[begin:end], "technology": page["technology"],
                "citation_number": page["citation_number"],
            })
            if end == len(text):
                break
            begin = max(begin + 1, end - overlap)
    return chunks


def corpus_stats(pages: list[dict], chunks: list[dict]) -> dict:
    actual = {p["doc_id"]: p["actual_pages"] for p in pages}
    declared = {p["doc_id"]: p["declared_pages"] for p in pages}
    excluded = sum(p["excluded"] for p in pages)
    return {"actual_pages": actual, "declared_pages": declared, "total_pages": sum(actual.values()),
            "excluded_pages": excluded, "indexed_pages": len(pages)-excluded,
            "chunks": len(chunks), "design_expected_chunks": 333,
            "warnings": (["Actual PDF page counts differ from design: review paper versions"] if actual != declared else [])
                        + (["Bibliography pages differ from design's 11: verify exclusions"] if excluded != 11 else [])
                        + (["Measured chunks differ from design's stated 333: review parsing"] if len(chunks) != 333 else [])}
