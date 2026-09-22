"""전처리 파이프라인 산출물(data/processed/chunks.jsonl)을 색인 입력으로 적재한다.

설계 산출물 B-3)의 파싱·노이즈 제거·청킹(1,200자/겹침 200자)·페이지 예산(≤200p) 검증은
preprocessing 패키지(`python -m preprocessing.pipeline`)가 담당하고, 이 모듈은 그 결과를
읽어 검색 색인이 쓰는 형태로 변환하는 역할만 한다. 같은 PDF를 여기서 다시 파싱하면
2단 레이아웃 재정렬·표 분리·header/footer 제거·참고문헌 제외가 모두 사라지므로,
원문 PDF를 직접 읽지 않는다.

chunks.jsonl 한 줄(청크)의 필드 중 색인에 쓰는 값:
    chunk_id, doc_id, content_type(text/table), start_page, text
manifest(data/manifest.json)에서 문서별 진영/역할을 읽어 기술 필터용 technology와
보고서 인용 번호 [n, p.X]의 n(citation_number)을 문서 순서대로 부여한다.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from ..config import CLEAN_FORMULA_NOISE, MAX_PAGES, MIN_INDEXED_CHARS

Technology = Literal["mla", "itme", "baseline"]


@dataclass(frozen=True)
class PaperSpec:
    doc_id: str
    technology: Technology
    citation_number: int
    title: str
    declared_pages: int


def _technology_of(camp: str, role: str) -> Technology:
    """선정 기술 2건(SW=MLA, HW=ITME)과 비교용 베이스라인을 구분한다(설계 A-4)."""
    if role != "primary":
        return "baseline"
    return "mla" if camp == "SW" else "itme"


def paper_manifest(manifest_path: Path) -> list[PaperSpec]:
    """data/manifest.json 순서대로 문서 메타데이터를 만든다.

    인용 번호는 manifest의 문서 순서(1부터)이며, 전처리 단계가 chunks.jsonl의
    citation 문자열을 만들 때 쓴 번호와 동일하다.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [
        PaperSpec(
            doc_id=doc["doc_id"],
            technology=_technology_of(doc["camp"], doc["role"]),
            citation_number=index,
            title=doc["title"],
            declared_pages=doc.get("expected_pages", 0),
        )
        for index, doc in enumerate(manifest["documents"], start=1)
    ]


# 수식이 깨져 나온 줄 판별용: 영문 3자 이상 단어가 하나도 없는 줄은 본문 문장이 아니다.
_REAL_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def clean_formula_noise(text: str) -> str:
    """임베딩 전에 깨진 수식 줄을 제거한다.

    PDF의 수식은 렌더링되지 않고 글리프만 좌표 순서대로 추출되기 때문에 "𝐡𝐡𝑡𝑡",
    "𝐿𝐿", "4 …" 처럼 단어가 없는 줄이 생긴다. 이런 줄은 문장이 아니라서 임베딩에
    의미 없는 토큰만 더해 검색 관련성을 떨어뜨리므로 색인 전에 걷어낸다.
    실제 문장에는 3자 이상 영단어가 거의 항상 있어 오탐 위험은 낮다.
    """
    kept = [line for line in text.split("\n") if not line.strip() or _REAL_WORD_RE.search(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def load_processed_chunks(chunks_path: Path, manifest: list[PaperSpec]) -> list[dict]:
    """chunks.jsonl을 읽어 색인용 청크 목록으로 변환한다.

    표 청크는 숫자·짧은 라벨 위주라 수식 노이즈 제거 기준을 적용하면 정상 데이터가
    지워질 수 있으므로 본문(text) 청크에만 정제를 적용한다.
    """
    path = Path(chunks_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"전처리 산출물이 없습니다: {path}\n"
            f"원문 PDF를 data/raw/에 배치한 뒤 `python -m preprocessing.pipeline`을 먼저 실행하세요."
        )
    specs = {spec.doc_id: spec for spec in manifest}

    chunks: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        spec = specs.get(record["doc_id"])
        if spec is None:
            raise ValueError(
                f"manifest에 없는 문서의 청크입니다: {record['doc_id']} "
                f"(manifest 문서: {sorted(specs)})"
            )
        text = record["text"]
        if CLEAN_FORMULA_NOISE and record["content_type"] == "text":
            text = clean_formula_noise(text)
        if len(text) < MIN_INDEXED_CHARS:
            continue
        chunks.append({
            "chunk_id": record["chunk_id"],
            "doc_id": record["doc_id"],
            # 청크가 페이지를 걸치면 시작 페이지로 인용한다(보고서 인용 형식 [n, p.X] 유지).
            "page": record["start_page"],
            "text": text,
            "technology": spec.technology,
            "citation_number": spec.citation_number,
            "content_type": record["content_type"],
        })

    if not chunks:
        raise ValueError(f"색인 가능한 청크가 없습니다: {path}")
    indexed_docs = {c["doc_id"] for c in chunks}
    if indexed_docs != set(specs):
        raise ValueError(f"색인 대상 문서 누락: {sorted(set(specs) - indexed_docs)}")
    return chunks


def corpus_stats(chunks: list[dict], manifest: list[PaperSpec], summary_path: Path) -> dict:
    """페이지 예산(≤200p)과 설계서 대비 편차를 기록한다(설계 B-3 ④).

    페이지 수는 전처리 단계가 원문 PDF에서 실제로 센 값(summary.json)을 사용하고,
    설계서에 적힌 값과 다르면 숨기지 않고 warnings로 남긴다.
    """
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    actual = {doc["doc_id"]: doc["total_pages"] for doc in summary["documents"]}
    declared = {spec.doc_id: spec.declared_pages for spec in manifest}
    total_pages = summary["total_pages"]
    if total_pages > MAX_PAGES:
        raise ValueError(f"문서 풀이 페이지 예산을 초과했습니다: {total_pages}p > {MAX_PAGES}p")

    per_doc_chunks: dict[str, int] = {}
    for chunk in chunks:
        per_doc_chunks[chunk["doc_id"]] = per_doc_chunks.get(chunk["doc_id"], 0) + 1
    warnings = []
    if actual != declared:
        warnings.append("실제 PDF 페이지 수가 설계서 기재값과 다릅니다: 문서 버전을 확인하세요")
    if len(chunks) != 333:
        warnings.append(f"측정된 청크 수({len(chunks)})가 설계서 기재값 333과 다릅니다")
    if summary.get("needs_manual_review"):
        warnings.append("전처리 summary.json에 사람이 확인해야 할 항목이 있습니다")
    return {
        "actual_pages": actual,
        "declared_pages": declared,
        "total_pages": total_pages,
        "page_budget": MAX_PAGES,
        "indexed_pages": sum(doc["indexed_pages"] for doc in summary["documents"]),
        "excluded_reference_pages": sum(doc["excluded_reference_pages"] for doc in summary["documents"]),
        "chunks": len(chunks),
        "chunks_per_document": per_doc_chunks,
        "design_expected_chunks": 333,
        "warnings": warnings,
    }
