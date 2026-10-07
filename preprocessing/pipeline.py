"""PDF 전처리 파이프라인. 결과는 data/processed/chunks.jsonl과 summary.json에 쓴다.

파싱, 머리글/바닥글 제거, 표 분리, 참고문헌 제외, 청킹, 페이지 예산 검사 순으로 처리한다.
차트 위 텍스트나 표 헤더 이월처럼 자동 판단이 위험한 항목은 review_flags와
pages_with_charts에 표시만 한다.

    python -m preprocessing.pipeline
"""

import json
from pathlib import Path

from preprocessing.budget import enforce_page_budget
from preprocessing.chunker import CHUNK_SIZE, OVERLAP, chunk_document
from preprocessing.columns import group_into_lines, line_text, line_top, reconstruct_reading_order_text
from preprocessing.headers_footers import collect_band_lines, detect_boilerplate_lines, strip_boilerplate_lines
from preprocessing.loader import RawPage, load_pdf_raw_pages
from preprocessing.noise_filter import drop_reference_pages, find_reference_start_page
from preprocessing.page_text import Page
from preprocessing.tables import NormalizedTable, extract_tables_for_page, split_table_into_chunks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.json"

CHART_AREA_RATIO_THRESHOLD = 0.15  # 페이지 면적 대비 이 비율 이상인 이미지는 차트로 본다
VECTOR_DIAGRAM_SHAPE_THRESHOLD = 20  # 벡터 도형이 이만큼 있으면 다이어그램으로 본다


def format_citation(doc_index: int, start_page: int, end_page: int) -> str:
    """보고서 인용 형식 [n, p.X] (여러 페이지에 걸치면 p.X-Y)."""
    if start_page == end_page:
        return f"[{doc_index}, p.{start_page}]"
    return f"[{doc_index}, p.{start_page}-{end_page}]"


def _band_lines_for_page(raw_page: RawPage) -> list[tuple[float, str]]:
    lines = group_into_lines(raw_page.words)
    return [(line_top(line), line_text(line)) for line in lines]


def _pages_with_large_images(raw_pages: list[RawPage]) -> list[int]:
    flagged = []
    for raw_page in raw_pages:
        page_area = raw_page.width * raw_page.height
        if page_area <= 0:
            continue
        for image in raw_page.images:
            x0, top, x1, bottom = image.bbox
            image_area = max(0.0, x1 - x0) * max(0.0, bottom - top)
            if image_area / page_area >= CHART_AREA_RATIO_THRESHOLD:
                flagged.append(raw_page.page_number)
                break
    return flagged


def _pages_with_vector_diagrams(raw_pages: list[RawPage]) -> list[int]:
    """벡터 도형으로 그린 다이어그램·차트가 있는 페이지를 찾는다.

    표나 이미지로 잡히지 않아 라벨이 본문 사이에 섞일 수 있으므로 사람이 확인하게 표시한다.
    """
    return [
        raw_page.page_number
        for raw_page in raw_pages
        if raw_page.vector_shape_count >= VECTOR_DIAGRAM_SHAPE_THRESHOLD
    ]


def _process_document(doc: dict, doc_index: int) -> tuple[list[dict], dict, int]:
    pdf_path = RAW_DIR / doc["filename"]
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"원문 PDF를 찾을 수 없습니다: {pdf_path}\n"
            f"data/raw/ 아래에 파일을 배치한 뒤 다시 실행하세요 "
            f"(data/manifest.json의 filename과 일치해야 합니다)."
        )

    raw_pages = load_pdf_raw_pages(doc["doc_id"], pdf_path)
    total_pages = len(raw_pages)

    band_lines_per_page = [
        collect_band_lines(rp.page_number, rp.height, _band_lines_for_page(rp)) for rp in raw_pages
    ]
    boilerplate = detect_boilerplate_lines(band_lines_per_page)

    text_pages: list[Page] = []
    all_tables: list[NormalizedTable] = []
    removed_boilerplate_lines = 0

    for raw_page in raw_pages:
        exclude_bboxes = [t.bbox for t in raw_page.tables] + raw_page.rejected_region_bboxes
        body_text = reconstruct_reading_order_text(
            raw_page.words, raw_page.width, raw_page.height, exclude_bboxes=exclude_bboxes
        )
        clean_text, removed_lines = strip_boilerplate_lines(body_text, boilerplate)
        removed_boilerplate_lines += len(removed_lines)
        text_pages.append(Page(doc_id=doc["doc_id"], page_number=raw_page.page_number, text=clean_text))

        all_tables.extend(
            extract_tables_for_page(doc["doc_id"], raw_page.page_number, raw_page.words, raw_page.tables)
        )

    pages_with_charts = _pages_with_large_images(raw_pages)
    pages_with_vector_diagrams = _pages_with_vector_diagrams(raw_pages)

    reference_start_page = find_reference_start_page(
        text_pages, manual_start_page=doc.get("reference_start_page")
    )
    reference_end_page = doc.get("reference_end_page")
    body_pages, excluded_pages = drop_reference_pages(
        text_pages, reference_start_page, reference_end_page
    )
    if reference_start_page is not None:
        indexed_page_numbers = {page.page_number for page in body_pages}
        all_tables = [t for t in all_tables if t.page_number in indexed_page_numbers]

    text_chunks, review_flags = chunk_document(doc["doc_id"], body_pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    doc_chunks: list[dict] = []
    for chunk in text_chunks:
        doc_chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_text_{chunk.chunk_index:04d}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "camp": doc["camp"],  # SW / HW, 기술별 필터용
                "role": doc["role"],  # primary / baseline
                "content_type": "text",
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "citation": format_citation(doc_index, chunk.start_page, chunk.end_page),
                "text": chunk.text,
                "char_count": len(chunk.text),
            }
        )

    table_chunk_index = 0
    for table in all_tables:
        # 큰 표는 여러 조각으로 나뉘므로 순번은 표 단위가 아니라 표 청크 전체에 매긴다.
        for text in split_table_into_chunks(table, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
            doc_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_table_{table_chunk_index:04d}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "camp": doc["camp"],
                    "role": doc["role"],
                    "content_type": "table",
                    "start_page": table.page_number,
                    "end_page": table.page_number,
                    "citation": format_citation(doc_index, table.page_number, table.page_number),
                    "text": text,
                    "char_count": len(text),
                    "has_caption": table.caption is not None,
                    "has_footnote": table.footnote is not None,
                }
            )
            table_chunk_index += 1

    doc_summary = {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "camp": doc["camp"],
        "role": doc["role"],
        "total_pages": total_pages,
        "reference_start_page": reference_start_page,
        "reference_end_page": reference_end_page,
        "excluded_reference_pages": len(excluded_pages),
        "indexed_pages": len(body_pages),
        "text_chunk_count": len(text_chunks),
        "table_count": len(all_tables),
        "removed_boilerplate_line_count": removed_boilerplate_lines,
        "pages_with_charts": pages_with_charts,
        "pages_with_vector_diagrams": pages_with_vector_diagrams,
        "page_boundary_review_flags": review_flags,
    }

    return doc_chunks, doc_summary, total_pages


def run(chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    page_budget = manifest.get("page_budget", 200)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    document_page_counts: dict[str, int] = {}
    doc_summaries: list[dict] = []

    for doc_index, doc in enumerate(documents, start=1):
        doc_chunks, doc_summary, total_pages = _process_document(doc, doc_index)
        all_chunks.extend(doc_chunks)
        doc_summaries.append(doc_summary)
        document_page_counts[doc["doc_id"]] = total_pages

    total_pages = enforce_page_budget(document_page_counts, budget=page_budget)

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    needs_manual_review = any(
        doc["pages_with_charts"] or doc["pages_with_vector_diagrams"] or doc["page_boundary_review_flags"]
        for doc in doc_summaries
    )

    result = {
        "total_pages": total_pages,
        "page_budget": page_budget,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(all_chunks),
        "needs_manual_review": needs_manual_review,
        "documents": doc_summaries,
    }
    summary_path = PROCESSED_DIR / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"총 {len(documents)}개 문서, {total_pages}p (예산 {page_budget}p), "
        f"{len(all_chunks)}개 청크 생성 완료"
    )
    print(f"- 청크: {chunks_path}")
    print(f"- 요약: {summary_path}")
    if needs_manual_review:
        print("주의: 차트/다이어그램/표 경계 의심 항목이 있습니다. summary.json의 pages_with_charts / "
              "pages_with_vector_diagrams / page_boundary_review_flags를 확인하세요.")

    return result


if __name__ == "__main__":
    run()
