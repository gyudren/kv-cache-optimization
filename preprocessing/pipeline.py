"""데이터 전처리 파이프라인.

파싱(2단 레이아웃 재정렬) → header/footer 제거 → 표 분리(캡션·각주·병합셀 정규화)
→ 참고문헌 구간 제외 → 청킹(페이지 경계 문장 이어붙이기 포함) → 페이지 예산 검증
→ 메타데이터 부착 순으로 처리한다.

역할 범위(데이터 전처리)만 다룬다. 임베딩 생성, FAISS/BM25 색인, 에이전트/프롬프트 로직은
다른 담당자의 영역이므로 이 모듈에서 다루지 않는다.

자동으로 판단하기 위험한 항목(차트/이미지 위 텍스트, 표 헤더 이월 의심 등)은 임의로
넘겨짚지 않고 review_flags / pages_with_charts 로 표시만 하여 사람이 확인하도록 한다.

출력:
    data/processed/chunks.jsonl  - 다음 단계(vector db 구성)에서 바로 임베딩할 수 있는 청크 목록
    data/processed/summary.json - 문서별 통계, 페이지 예산 검증 결과, 수동 확인이 필요한 항목 목록

실행:
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
from preprocessing.tables import NormalizedTable, extract_tables_for_page, table_chunk_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.json"

CHART_AREA_RATIO_THRESHOLD = 0.15  # 페이지 면적의 이 비율을 넘는 이미지는 차트로 간주해 검토 대상 표시
VECTOR_DIAGRAM_SHAPE_THRESHOLD = 20  # 이 개수 이상의 사각형/선/곡선이 있으면 벡터 다이어그램으로 간주


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
    """래스터 이미지가 아니라 사각형/선/곡선으로 직접 그린 아키텍처 다이어그램·차트를 감지한다.

    이런 다이어그램은 표로도, 이미지로도 잡히지 않고 라벨 텍스트만 본문 흐름 중간에 끼어들어가
    주변 문장과 뒤섞일 수 있어(성능 실험 결과) 별도로 표시해 사람이 확인하게 한다.
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
    body_pages, excluded_pages = drop_reference_pages(text_pages, reference_start_page)
    if reference_start_page is not None:
        all_tables = [t for t in all_tables if t.page_number < reference_start_page]

    text_chunks, review_flags = chunk_document(doc["doc_id"], body_pages, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

    doc_chunks: list[dict] = []
    for chunk in text_chunks:
        doc_chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_text_{chunk.chunk_index:04d}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "camp": doc["camp"],  # SW / HW - 기술별 문서 필터용
                "role": doc["role"],  # primary / baseline
                "content_type": "text",
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "citation": format_citation(doc_index, chunk.start_page, chunk.end_page),
                "text": chunk.text,
                "char_count": len(chunk.text),
            }
        )

    for table_index, table in enumerate(all_tables):
        text = table_chunk_text(table)
        doc_chunks.append(
            {
                "chunk_id": f"{doc['doc_id']}_table_{table_index:04d}",
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

    doc_summary = {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "camp": doc["camp"],
        "role": doc["role"],
        "total_pages": total_pages,
        "reference_start_page": reference_start_page,
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
