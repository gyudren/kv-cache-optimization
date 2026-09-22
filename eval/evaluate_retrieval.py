"""운영 검색기(FAISS + BM25 → RRF)의 검색 품질을 LLM 생성과 분리해 평가한다.

app.py가 실제로 쓰는 것과 동일한 경로(전처리 청크 → Qwen3 임베딩 → FAISS/BM25 → RRF)를
그대로 사용하므로, 여기서 나온 수치가 곧 파이프라인의 검색 성능이다.

평가 케이스: eval/retrieval_cases.json
- concept/table/formula : 기대 페이지·필수 용어까지 확인하는 정밀 케이스
- coverage              : 문서당 20개씩, 정답 문서가 상위에 오는지 확인하는 커버리지 케이스

지표
- Hit@K : 상위 K개 안에 정답 문서(expected_doc_ids)의 청크가 있으면 1
- MRR   : 정답 문서가 처음 등장한 순위의 역수 평균
- page_hit           : 기대 페이지(expected_pages)가 상위 K개에 포함된 비율
- required_term_cov  : 필수 용어(required_terms)가 검색된 본문에 등장한 비율
합격 기준(acceptance)을 모두 만족해야 passed=True 가 된다.

Usage:
    python -m eval.evaluate_retrieval [--top-k 5] [--show-hits]
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # 소스 체크아웃 실행 지원
from kv_eval.config import RETRIEVAL_K
from kv_eval.rag.index import build_index
from kv_eval.rag.ingest import load_processed_chunks, paper_manifest
from kv_eval.rag.retrieve import HybridRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "eval" / "retrieval_cases.json"
RESULTS_PATH = BASE_DIR / "outputs" / "retrieval_eval.json"
PRECISE_CATEGORIES = ("concept", "table", "formula")


def _technology_filter(expected_doc_ids: list[str]) -> str:
    """케이스의 정답 문서에 맞는 기술 필터를 고른다(운영과 동일한 필터 경로)."""
    if expected_doc_ids == ["deepseek_v2_mla"]:
        return "mla"
    if expected_doc_ids == ["itme"]:
        return "itme"
    return "itme_baseline"


def evaluate(top_k: int = RETRIEVAL_K, show_hits: bool = False) -> dict:
    spec = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases, acceptance = spec["cases"], spec["acceptance"]
    print(f"평가 케이스 {len(cases)}개 로드", flush=True)

    manifest = paper_manifest(BASE_DIR / "data" / "manifest.json")
    chunks = load_processed_chunks(BASE_DIR / "data" / "processed" / "chunks.jsonl", manifest)
    print(f"청크 {len(chunks)}개 색인 중(FAISS + BM25)...", flush=True)
    retriever = HybridRetriever(build_index(chunks))
    print("색인 완료", flush=True)

    results: list[dict] = []
    per_category: dict[str, list[dict]] = defaultdict(list)

    for case in cases:
        hits = retriever.hybrid_search(case["query"], _technology_filter(case["expected_doc_ids"]), top_k)
        expected_docs = set(case["expected_doc_ids"])
        rank = next((i + 1 for i, h in enumerate(hits) if h.doc_id in expected_docs), None)

        expected_pages = set(case.get("expected_pages", []))
        page_hit = (bool(expected_pages & {h.page for h in hits if h.doc_id in expected_docs})
                    if expected_pages else None)
        terms = case.get("required_terms", [])
        retrieved_text = " ".join(h.text for h in hits if h.doc_id in expected_docs).lower()
        term_cov = (sum(t.lower() in retrieved_text for t in terms) / len(terms)) if terms else None
        # 정밀 케이스에서 정답 문서가 아닌 청크가 1위를 차지하면 노이즈로 센다.
        noise = bool(hits) and hits[0].doc_id not in expected_docs

        record = {
            "id": case["id"], "category": case["category"], "query": case["query"],
            "expected_doc_ids": case["expected_doc_ids"], "rank_of_correct_doc": rank,
            "page_hit": page_hit, "required_term_coverage": term_cov, "top1_is_noise": noise,
            "hits": [{"rank": i + 1, "chunk_id": h.chunk_id, "doc_id": h.doc_id,
                      "citation": f"[{h.citation_number}, p.{h.page}]", "score": round(h.score, 5),
                      "snippet": h.text[:200].replace("\n", " ")} for i, h in enumerate(hits)],
            "human_feedback": None,  # 사람이 검토 후 "ok"/"bad"/메모를 직접 채우는 칸
        }
        results.append(record)
        per_category[case["category"]].append(record)

    def _rate(items: list[dict], predicate) -> float:
        return (sum(bool(predicate(r)) for r in items) / len(items)) if items else 0.0

    precise = [r for r in results if r["category"] in PRECISE_CATEGORIES]
    page_checked = [r for r in results if r["page_hit"] is not None]
    term_checked = [r for r in results if r["required_term_coverage"] is not None]

    metrics = {
        "n_cases": len(results),
        "hit@1": _rate(results, lambda r: r["rank_of_correct_doc"] == 1),
        "hit@3": _rate(results, lambda r: r["rank_of_correct_doc"] and r["rank_of_correct_doc"] <= 3),
        f"hit@{top_k}": _rate(results, lambda r: r["rank_of_correct_doc"] is not None),
        "mrr": sum(1.0 / r["rank_of_correct_doc"] if r["rank_of_correct_doc"] else 0.0
                   for r in results) / len(results),
        "page_hit_rate": _rate(page_checked, lambda r: r["page_hit"]),
        "required_term_coverage": (sum(r["required_term_coverage"] for r in term_checked) / len(term_checked)
                                   if term_checked else 0.0),
        "concept_table_noise_rate": _rate(precise, lambda r: r["top1_is_noise"]),
        "per_category": {name: {"n": len(items),
                                "hit@1": _rate(items, lambda r: r["rank_of_correct_doc"] == 1),
                                f"hit@{top_k}": _rate(items, lambda r: r["rank_of_correct_doc"] is not None)}
                         for name, items in sorted(per_category.items())},
    }

    checks = {
        f"hit@{top_k} ≥ {acceptance['hit_at_5_min']}": metrics[f"hit@{top_k}"] >= acceptance["hit_at_5_min"],
        f"MRR ≥ {acceptance['mrr_min']}": metrics["mrr"] >= acceptance["mrr_min"],
        f"노이즈율 ≤ {acceptance['concept_table_noise_rate_max']}":
            metrics["concept_table_noise_rate"] <= acceptance["concept_table_noise_rate_max"],
        f"필수 용어 커버리지 ≥ {acceptance['required_term_coverage_min']}":
            metrics["required_term_coverage"] >= acceptance["required_term_coverage_min"],
    }
    report = {"top_k": top_k, "acceptance": acceptance, "checks": checks,
              "passed": all(checks.values()), "metrics": metrics, "cases": results}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report, show_hits)
    return report


def _print(report: dict, show_hits: bool) -> None:
    m = report["metrics"]
    top_k = report["top_k"]
    print("\n=== 카테고리별 ===")
    print(f"{'category':<12}{'n':>4}{'Hit@1':>8}{f'Hit@{top_k}':>8}")
    for name, s in m["per_category"].items():
        print(f"{name:<12}{s['n']:>4}{s['hit@1']:>8.2f}{s[f'hit@{top_k}']:>8.2f}")

    print("\n=== 전체 ===")
    print(f"케이스 {m['n_cases']}개 | Hit@1={m['hit@1']:.2f} Hit@3={m['hit@3']:.2f} "
          f"Hit@{top_k}={m[f'hit@{top_k}']:.2f} MRR={m['mrr']:.2f}")
    print(f"기대 페이지 적중={m['page_hit_rate']:.2f} | 필수 용어 커버리지={m['required_term_coverage']:.2f} "
          f"| 정밀 케이스 노이즈율={m['concept_table_noise_rate']:.2f}")

    print("\n=== 합격 기준 ===")
    for name, ok in report["checks"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"→ 종합: {'PASS' if report['passed'] else 'FAIL'}")

    misses = [r for r in report["cases"] if r["rank_of_correct_doc"] is None]
    if misses:
        print(f"\n=== 검색 실패(top-{top_k} 밖) {len(misses)}건 ===")
        for r in misses:
            print(f"- [{r['id']}] {r['query']}")

    if show_hits:
        print("\n=== 케이스별 hit 문서 (사람 검토용) ===")
        for r in report["cases"]:
            mark = "OK" if r["rank_of_correct_doc"] == 1 else "MISS"
            print(f"\n[{mark}] {r['id']}: {r['query']} (정답: {', '.join(r['expected_doc_ids'])})")
            for hit in r["hits"]:
                flag = "*" if hit["doc_id"] in r["expected_doc_ids"] else " "
                print(f"  {flag}{hit['rank']}위 {hit['doc_id']:<18}{hit['citation']:<14}{hit['snippet'][:90]}")
    print(f"\n결과 저장: {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="운영 검색기(FAISS+BM25+RRF) 검색 품질 평가")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_K)
    parser.add_argument("--show-hits", action="store_true", help="케이스별 검색 결과를 모두 출력")
    args = parser.parse_args()
    evaluate(top_k=args.top_k, show_hits=args.show_hits)
