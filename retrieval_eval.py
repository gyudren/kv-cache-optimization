"""ChromaDB 검색 품질을 LLM 생성과 분리해 평가한다.

실행 예시:
    python -m retrieval_eval
    python -m retrieval_eval --top-k 5 --show-results
    EMBEDDING_DEVICE=mps python -m retrieval_eval
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from statistics import mean
from typing import Any

import chromadb

from vectordb import (
    COLLECTION_NAME,
    PERSIST_DIR,
    Qwen3EmbeddingFunction,
    get_collection_names,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = BASE_DIR / "evaluation" / "retrieval_cases.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "outputs" / "retrieval_eval.json"


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    category: str
    language: str
    query: str
    expected_doc_ids: tuple[str, ...]
    expected_pages: tuple[int, ...]
    expected_content_types: tuple[str, ...]
    required_terms: tuple[str, ...]


def load_suite(path: Path) -> tuple[list[RetrievalCase], dict[str, float]]:
    """평가 JSON을 읽고 필수 필드를 검증한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if not raw_cases:
        raise ValueError(f"평가 케이스가 없습니다: {path}")

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        case_id = raw["id"]
        if case_id in seen_ids:
            raise ValueError(f"중복 평가 ID: {case_id}")
        seen_ids.add(case_id)

        expected_doc_ids = tuple(raw.get("expected_doc_ids", []))
        expected_pages = tuple(raw.get("expected_pages", []))
        if not expected_doc_ids:
            raise ValueError(f"{case_id}: expected_doc_ids가 필요합니다.")
        if not expected_pages:
            raise ValueError(f"{case_id}: expected_pages가 필요합니다.")

        cases.append(
            RetrievalCase(
                case_id=case_id,
                category=raw["category"],
                language=raw.get("language", "ko"),
                query=raw["query"],
                expected_doc_ids=expected_doc_ids,
                expected_pages=expected_pages,
                expected_content_types=tuple(raw.get("expected_content_types", [])),
                required_terms=tuple(raw.get("required_terms", [])),
            )
        )

    return cases, payload.get("acceptance", {})


def page_matches(metadata: dict[str, Any], expected_pages: tuple[int, ...]) -> bool:
    """검색 청크의 페이지 범위가 기대 페이지 중 하나와 겹치는지 확인한다."""
    start_page = metadata.get("start_page")
    end_page = metadata.get("end_page")
    if not isinstance(start_page, int) or not isinstance(end_page, int):
        return False
    return any(start_page <= page <= end_page for page in expected_pages)


def is_relevant(metadata: dict[str, Any], case: RetrievalCase) -> bool:
    return (
        metadata.get("doc_id") in case.expected_doc_ids
        and page_matches(metadata, case.expected_pages)
    )


def numeric_ratio(text: str) -> float:
    compact = re.sub(r"\s", "", text)
    if not compact:
        return 0.0
    return sum(character.isdigit() for character in compact) / len(compact)


def is_table_like(document: str, metadata: dict[str, Any]) -> bool:
    """명시적 표 또는 숫자 배열 형태로 본문에 섞인 표를 휴리스틱으로 탐지한다."""
    if metadata.get("content_type") == "table":
        return True
    if numeric_ratio(document) >= 0.18:
        return True

    numeric_lines = sum(
        1
        for line in document.splitlines()
        if len(re.findall(r"\d+(?:\.\d+)?", line)) >= 3
    )
    return bool(re.search(r"\bTable\s+\d+", document, re.IGNORECASE)) and numeric_lines >= 2


def make_snippet(text: str, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


def evaluate_case(collection: Any, case: RetrievalCase, top_k: int) -> dict[str, Any]:
    response = collection.query(
        query_texts=[case.query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for index, result_id in enumerate(ids):
        document = documents[index] or ""
        metadata = metadatas[index] or {}
        relevant = is_relevant(metadata, case)
        results.append(
            {
                "rank": index + 1,
                "chunk_id": result_id,
                "distance": distances[index],
                "doc_id": metadata.get("doc_id"),
                "content_type": metadata.get("content_type"),
                "start_page": metadata.get("start_page"),
                "end_page": metadata.get("end_page"),
                "citation": metadata.get("citation"),
                "relevant": relevant,
                "table_like": is_table_like(document, metadata),
                "numeric_ratio": round(numeric_ratio(document), 4),
                "snippet": make_snippet(document),
                "_document": document,
            }
        )

    relevant_results = [result for result in results if result["relevant"]]
    first_relevant_rank = relevant_results[0]["rank"] if relevant_results else None
    relevant_evidence = "\n".join(
        result["_document"].casefold() for result in relevant_results
    )
    matched_terms = [
        term for term in case.required_terms if term.casefold() in relevant_evidence
    ]
    term_coverage = (
        len(matched_terms) / len(case.required_terms)
        if case.required_terms
        else None
    )

    expected_types = set(case.expected_content_types)
    first_relevant_type_match = None
    if relevant_results and expected_types:
        first_relevant_type_match = relevant_results[0]["content_type"] in expected_types

    result_count = len(results)
    metrics = {
        "first_relevant_rank": first_relevant_rank,
        "hit_at_1": first_relevant_rank == 1,
        "hit_at_3": first_relevant_rank is not None and first_relevant_rank <= 3,
        "hit_at_5": first_relevant_rank is not None and first_relevant_rank <= 5,
        "reciprocal_rank": 1 / first_relevant_rank if first_relevant_rank else 0.0,
        "document_contamination_rate": (
            sum(result["doc_id"] not in case.expected_doc_ids for result in results)
            / result_count
            if result_count
            else 0.0
        ),
        "table_noise_rate": (
            sum(result["table_like"] for result in results) / result_count
            if result_count
            else 0.0
        ),
        "required_term_coverage": term_coverage,
        "matched_required_terms": matched_terms,
        "first_relevant_type_match": first_relevant_type_match,
    }

    for result in results:
        result.pop("_document", None)

    return {
        "id": case.case_id,
        "category": case.category,
        "language": case.language,
        "query": case.query,
        "expected": {
            "doc_ids": list(case.expected_doc_ids),
            "pages": list(case.expected_pages),
            "content_types": list(case.expected_content_types),
            "required_terms": list(case.required_terms),
        },
        "metrics": metrics,
        "results": results,
    }


def aggregate_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """전체 및 카테고리별 평균 지표를 계산한다."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["overall"] = case_results
    for result in case_results:
        grouped[result["category"]].append(result)

    summary: dict[str, Any] = {}
    for group_name, results in grouped.items():
        metrics = [result["metrics"] for result in results]
        term_coverages = [
            metric["required_term_coverage"]
            for metric in metrics
            if metric["required_term_coverage"] is not None
        ]
        type_matches = [
            metric["first_relevant_type_match"]
            for metric in metrics
            if metric["first_relevant_type_match"] is not None
        ]
        summary[group_name] = {
            "case_count": len(results),
            "hit_at_1": mean(metric["hit_at_1"] for metric in metrics),
            "hit_at_3": mean(metric["hit_at_3"] for metric in metrics),
            "hit_at_5": mean(metric["hit_at_5"] for metric in metrics),
            "mrr": mean(metric["reciprocal_rank"] for metric in metrics),
            "document_contamination_rate": mean(
                metric["document_contamination_rate"] for metric in metrics
            ),
            "table_noise_rate": mean(metric["table_noise_rate"] for metric in metrics),
            "required_term_coverage": mean(term_coverages) if term_coverages else None,
            "first_relevant_type_match_rate": mean(type_matches) if type_matches else None,
        }
    return summary


def evaluate_acceptance(
    summary: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    overall = summary["overall"]
    concept = summary.get("concept", {})
    checks = {
        "hit_at_5": {
            "actual": overall["hit_at_5"],
            "operator": ">=",
            "threshold": thresholds.get("hit_at_5_min", 0.8),
        },
        "mrr": {
            "actual": overall["mrr"],
            "operator": ">=",
            "threshold": thresholds.get("mrr_min", 0.6),
        },
        "concept_table_noise_rate": {
            "actual": concept.get("table_noise_rate", 0.0),
            "operator": "<=",
            "threshold": thresholds.get("concept_table_noise_rate_max", 0.2),
        },
        "required_term_coverage": {
            "actual": overall["required_term_coverage"],
            "operator": ">=",
            "threshold": thresholds.get("required_term_coverage_min", 0.8),
        },
    }

    for check in checks.values():
        actual = check["actual"]
        threshold = check["threshold"]
        check["passed"] = (
            actual >= threshold if check["operator"] == ">=" else actual <= threshold
        )

    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
    }


def print_summary(report: dict[str, Any], show_results: bool) -> None:
    print(f"\n검색 평가: {report['acceptance']['passed'] and 'PASS' or 'FAIL'}")
    print(
        "전체 "
        f"Hit@1={report['summary']['overall']['hit_at_1']:.3f} | "
        f"Hit@3={report['summary']['overall']['hit_at_3']:.3f} | "
        f"Hit@5={report['summary']['overall']['hit_at_5']:.3f} | "
        f"MRR={report['summary']['overall']['mrr']:.3f}"
    )

    for result in report["cases"]:
        metrics = result["metrics"]
        status = "PASS" if metrics["hit_at_5"] else "FAIL"
        print(
            f"[{status}] {result['id']} | rank={metrics['first_relevant_rank']} | "
            f"contamination={metrics['document_contamination_rate']:.2f} | "
            f"table_noise={metrics['table_noise_rate']:.2f}"
        )
        if show_results:
            for hit in result["results"]:
                marker = "*" if hit["relevant"] else " "
                print(
                    f"  {marker}#{hit['rank']} {hit['chunk_id']} "
                    f"{hit['citation']} distance={hit['distance']:.4f} | {hit['snippet']}"
                )

    print(f"결과 파일: {report['output_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ChromaDB 검색 품질 평가")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--show-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 5:
        raise ValueError("Hit@5 평가를 위해 --top-k는 5 이상이어야 합니다.")
    if args.device:
        os.environ["EMBEDDING_DEVICE"] = args.device
    # 색인 생성 당시 내려받은 동일 모델을 사용하며, 평가 중 네트워크 확인 재시도를 막는다.
    os.environ.setdefault("EMBEDDING_LOCAL_FILES_ONLY", "1")

    cases, thresholds = load_suite(args.cases)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    if COLLECTION_NAME not in get_collection_names(client):
        raise RuntimeError(
            f"컬렉션 '{COLLECTION_NAME}'이 없습니다. 먼저 python -m vectordb를 실행하세요."
        )

    embedding_function = Qwen3EmbeddingFunction()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    if collection.count() == 0:
        raise RuntimeError(f"컬렉션 '{COLLECTION_NAME}'이 비어 있습니다.")

    actual_top_k = min(args.top_k, collection.count())
    case_results = [
        evaluate_case(collection, case, actual_top_k)
        for case in cases
    ]
    summary = aggregate_cases(case_results)
    acceptance = evaluate_acceptance(summary, thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION_NAME,
        "collection_count": collection.count(),
        "top_k": actual_top_k,
        "device": embedding_function.device,
        "cases_path": str(args.cases),
        "summary": summary,
        "acceptance": acceptance,
        "cases": case_results,
        "output_path": str(args.output),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(report, args.show_results)


if __name__ == "__main__":
    main()
