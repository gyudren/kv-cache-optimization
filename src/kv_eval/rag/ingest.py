"""Research PDF loading, source page provenance and bounded indexing.

NOTE(데이터 전처리 담당): 아래 ``load_papers``/``chunk_pages``는 ``data/papers/*.pdf``에서
직접 raw PDF를 파싱하는 경로인데, 그 폴더에는 실제 PDF가 없다(``data/papers/FILES_REQUIRED.txt``
참고 — placeholder만 있음). 원문 PDF는 ``data/raw/``에 있고, 이미 ``preprocessing/`` 패키지가
2단 레이아웃 재정렬·표 처리·참고문헌 제외·청킹까지 끝내 ``data/processed/chunks.jsonl``로
만들어 두었다. 이 파일 아래쪽의 ``load_prepared_chunks``/``preprocessed_corpus_stats``가
그 결과를 읽는 실제 동작 경로이고, index.py는 기본적으로 이쪽을 사용한다. 위쪽 함수들은
기존 테스트 호환을 위해 그대로 남겨둔다.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
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


# ---------------------------------------------------------------------------
# 실제 사용 경로: 데이터 전처리 담당(preprocessing/) 산출물을 그대로 읽는다.
# ---------------------------------------------------------------------------

PREPROCESSED_ROOT = Path(__file__).resolve().parents[3]
CHUNKS_PATH = PREPROCESSED_ROOT / "data" / "processed" / "chunks.jsonl"
SUMMARY_PATH = PREPROCESSED_ROOT / "data" / "processed" / "summary.json"

# preprocessing/manifest.json의 doc_id ↔ 이 패키지가 기대하는 technology 구분
DOC_TECHNOLOGY = {
    "deepseek_v2_mla": "mla",
    "itme": "itme",
    "infinigen": "baseline",
    "cxl_pnm": "baseline",
}

_CITATION_NUMBER_RE = re.compile(r"^\[(\d+),")


def _citation_number(citation: str) -> int:
    match = _CITATION_NUMBER_RE.match(citation)
    if not match:
        raise ValueError(f"citation 형식이 올바르지 않습니다: {citation!r}")
    return int(match.group(1))


def load_prepared_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """전처리 산출물을 읽어 RAG 색인이 바로 쓸 수 있는 청크 목록으로 변환한다.

    raw PDF를 여기서 다시 파싱하지 않는다 — 2단 레이아웃 재정렬, 표 캡션/각주 분리,
    참고문헌 제외, 1,200자/겹침 200자 청킹은 이미 preprocessing/pipeline.py가 끝냈다.
    반환 필드는 위쪽 chunk_pages()의 출력과 같은 모양(chunk_id, doc_id, page, text,
    technology, citation_number)이라 index.py의 나머지 로직은 그대로 재사용된다.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 `python -m preprocessing.pipeline` 을 실행해 "
            f"RAG 적재 문서 4편을 전처리하세요."
        )

    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            doc_id = raw["doc_id"]
            if doc_id not in DOC_TECHNOLOGY:
                raise ValueError(f"승인되지 않은 문서가 청크에 포함되어 있습니다: {doc_id}")
            chunks.append(
                {
                    "chunk_id": raw["chunk_id"],
                    "doc_id": doc_id,
                    "page": raw["start_page"],
                    "text": raw["text"],
                    "technology": DOC_TECHNOLOGY[doc_id],
                    "citation_number": _citation_number(raw["citation"]),
                }
            )
    if not chunks:
        raise ValueError(f"{path} 에서 읽은 청크가 없습니다.")
    return chunks


# DECLARED_PAGES(config.py)는 위쪽 raw-PDF 경로(paper_manifest 등)의 옛 doc_id("deepseek_v2")
# 기준이라 여기서는 재사용하지 않고, 실제 전처리 산출물 doc_id 기준으로 따로 선언한다.
PREPROCESSED_DECLARED_PAGES = {"deepseek_v2_mla": 52, "itme": 13, "infinigen": 18, "cxl_pnm": 13}


def preprocessed_corpus_stats(path: Path = SUMMARY_PATH) -> dict:
    """전처리 단계가 이미 검증한 페이지 예산·통계를 그대로 가져온다.

    (참고문헌 제외, 200p 예산 검증은 preprocessing/pipeline.py에서 이미 강제됨)
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 `python -m preprocessing.pipeline` 을 실행하세요.")
    with path.open(encoding="utf-8") as f:
        summary = json.load(f)

    per_doc = {d["doc_id"]: d for d in summary["documents"]}
    actual_pages = {doc_id: d["total_pages"] for doc_id, d in per_doc.items()}
    return {
        "actual_pages": actual_pages,
        "declared_pages": PREPROCESSED_DECLARED_PAGES,
        "total_pages": summary["total_pages"],
        "page_budget": summary["page_budget"],
        "indexed_pages": sum(d["indexed_pages"] for d in per_doc.values()),
        "chunks": summary["total_chunks"],
        "warnings": (
            ["Actual PDF page counts differ from design: review paper versions"]
            if actual_pages != PREPROCESSED_DECLARED_PAGES
            else []
        ),
    }
