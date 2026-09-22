"""데이터 전처리 파이프라인: 파싱 → 참고문헌 제외 → 청킹 → 예산 검증 → 메타데이터 부착.

역할 범위(데이터 전처리)만 다룬다. 임베딩 생성, FAISS/BM25 색인, 에이전트/프롬프트 로직은
다른 담당자의 영역이므로 이 모듈에서 다루지 않는다.

출력:
    data/processed/chunks.jsonl  - 다음 단계(vector db 구성)에서 바로 임베딩할 수 있는 청크 목록
    data/processed/summary.json - 문서별 페이지/청크 통계 및 페이지 예산 검증 결과

실행:
    python -m preprocessing.pipeline
"""

import json
from pathlib import Path

from preprocessing.budget import enforce_page_budget
from preprocessing.chunker import CHUNK_SIZE, OVERLAP, chunk_document
from preprocessing.loader import load_pdf_pages
from preprocessing.noise_filter import drop_reference_pages, find_reference_start_page

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def format_citation(doc_index: int, start_page: int, end_page: int) -> str:
    """보고서 인용 형식 [n, p.X] (여러 페이지에 걸치면 p.X-Y)."""
    if start_page == end_page:
        return f"[{doc_index}, p.{start_page}]"
    return f"[{doc_index}, p.{start_page}-{end_page}]"


def run(chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    page_budget = manifest.get("page_budget", 200)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    document_page_counts: dict[str, int] = {}
    doc_summaries: list[dict] = []

    for doc_index, doc in enumerate(documents, start=1):
        pdf_path = RAW_DIR / doc["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(
                f"원문 PDF를 찾을 수 없습니다: {pdf_path}\n"
                f"data/raw/ 아래에 파일을 배치한 뒤 다시 실행하세요 "
                f"(data/manifest.json의 filename과 일치해야 합니다)."
            )

        pages = load_pdf_pages(doc["doc_id"], pdf_path)
        document_page_counts[doc["doc_id"]] = len(pages)

        reference_start_page = find_reference_start_page(
            pages, manual_start_page=doc.get("reference_start_page")
        )
        body_pages, excluded_pages = drop_reference_pages(pages, reference_start_page)

        chunks = chunk_document(doc["doc_id"], body_pages, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            all_chunks.append(
                {
                    "chunk_id": f"{doc['doc_id']}_{chunk.chunk_index:04d}",
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "camp": doc["camp"],  # SW / HW - 기술별 문서 필터용
                    "role": doc["role"],  # primary / baseline
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "citation": format_citation(doc_index, chunk.start_page, chunk.end_page),
                    "text": chunk.text,
                    "char_count": len(chunk.text),
                }
            )

        doc_summaries.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "camp": doc["camp"],
                "role": doc["role"],
                "total_pages": len(pages),
                "reference_start_page": reference_start_page,
                "excluded_reference_pages": len(excluded_pages),
                "indexed_pages": len(body_pages),
                "chunk_count": len(chunks),
            }
        )

    total_pages = enforce_page_budget(document_page_counts, budget=page_budget)

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    result = {
        "total_pages": total_pages,
        "page_budget": page_budget,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_chunks": len(all_chunks),
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

    return result


if __name__ == "__main__":
    run()
