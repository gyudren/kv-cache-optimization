"""eval/queries.json의 질의로 ChromaDB 검색을 실제로 돌려서 검색 품질을 평가한다.

queries.json은 [{"query": "...", "expected_doc_id": "..."}, ...] 형태로,
각 질의가 원래 어느 문서 내용에 대한 질문인지 정답(ground truth)을 함께 들고 있다.
이 정답을 기준으로, 질의를 검색했을 때 상위 결과에 그 doc_id의 청크가 실제로
나오는지를 Hit@1 / Hit@3 / Hit@5 / MRR로 계산한다.

- Hit@K: 상위 K개 결과 안에 정답 문서(doc_id)의 청크가 하나라도 있으면 1, 없으면 0
- MRR(Mean Reciprocal Rank): 정답 문서가 처음 등장한 순위의 역수 평균 (1등이면 1.0, 3등이면 0.33)

Usage:
    python -m eval.evaluate_retrieval
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import chromadb

from vectordb import COLLECTION_NAME, PERSIST_DIR, Qwen3EmbeddingFunction, log

BASE_DIR = Path(__file__).resolve().parent.parent
QUERIES_PATH = BASE_DIR / "eval" / "queries.json"
RESULTS_PATH = BASE_DIR / "eval" / "retrieval_eval_results.json"
TOP_K = 5


def load_queries(path: Path = QUERIES_PATH) -> List[Dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(top_k: int = TOP_K) -> dict:
    queries = load_queries()
    log(f"질의 {len(queries)}개 로드 완료 (query + 정답 expected_doc_id 쌍)")

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=Qwen3EmbeddingFunction(),
    )
    log(f"컬렉션 '{COLLECTION_NAME}' 로드 완료 (총 {collection.count()}개 청크)")

    per_query_results = []
    per_doc_stats = defaultdict(lambda: {"hit@1": 0, "hit@3": 0, "hit@5": 0, "rr_sum": 0.0, "n": 0})

    for item in queries:
        question = item["query"]
        doc_id = item["expected_doc_id"]

        result = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )
        retrieved_metas = result["metadatas"][0]
        retrieved_doc_ids = [m["doc_id"] for m in retrieved_metas]
        retrieved_chunk_ids = result["ids"][0]
        distances = result["distances"][0]
        retrieved_texts = result["documents"][0]

        # 정답 doc_id가 상위 결과에서 몇 번째로 처음 나오는지(1-based). 없으면 None.
        rank = next((i + 1 for i, d in enumerate(retrieved_doc_ids) if d == doc_id), None)
        reciprocal_rank = 1.0 / rank if rank else 0.0

        stats = per_doc_stats[doc_id]
        stats["n"] += 1
        stats["hit@1"] += int(rank == 1)
        stats["hit@3"] += int(rank is not None and rank <= 3)
        stats["hit@5"] += int(rank is not None and rank <= 5)
        stats["rr_sum"] += reciprocal_rank

        # 사람이 직접 눈으로 보고 "이 청크가 실제로 질문에 답이 되는 내용인가"를
        # 판단할 수 있도록, hit된 문서 각각의 청크ID/인용/본문 스니펫을 남겨둔다.
        hits = [
            {
                "rank": i + 1,
                "chunk_id": retrieved_chunk_ids[i],
                "doc_id": retrieved_doc_ids[i],
                "citation": retrieved_metas[i]["citation"],
                "content_type": retrieved_metas[i]["content_type"],
                "distance": distances[i],
                "snippet": retrieved_texts[i][:200].replace("\n", " "),
            }
            for i in range(len(retrieved_chunk_ids))
        ]

        per_query_results.append(
            {
                "expected_doc_id": doc_id,
                "query": question,
                "rank_of_correct_doc": rank,
                "top1_chunk_id": retrieved_chunk_ids[0],
                "top1_doc_id": retrieved_doc_ids[0],
                "top1_distance": distances[0],
                "retrieved_doc_ids": retrieved_doc_ids,
                "hits": hits,
                "human_feedback": None,  # 사람이 검토 후 "ok" / "bad" / 메모 등을 직접 채워 넣는 칸
            }
        )

    # 문서별 요약 지표 계산
    per_doc_summary = {}
    for doc_id, stats in per_doc_stats.items():
        n = stats["n"]
        per_doc_summary[doc_id] = {
            "n_queries": n,
            "hit@1": stats["hit@1"] / n,
            "hit@3": stats["hit@3"] / n,
            "hit@5": stats["hit@5"] / n,
            "mrr": stats["rr_sum"] / n,
        }

    n_total = len(per_query_results)
    overall = {
        "n_queries": n_total,
        "hit@1": sum(r["rank_of_correct_doc"] == 1 for r in per_query_results) / n_total,
        "hit@3": sum(r["rank_of_correct_doc"] is not None and r["rank_of_correct_doc"] <= 3 for r in per_query_results) / n_total,
        "hit@5": sum(r["rank_of_correct_doc"] is not None and r["rank_of_correct_doc"] <= 5 for r in per_query_results) / n_total,
        "mrr": sum((1.0 / r["rank_of_correct_doc"]) if r["rank_of_correct_doc"] else 0.0 for r in per_query_results) / n_total,
    }

    report = {"overall": overall, "per_document": per_doc_summary, "queries": per_query_results}
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"결과 저장: {RESULTS_PATH}")

    _print_hits_for_human_review(report)
    _print_report(report)
    return report


def _print_hits_for_human_review(report: dict) -> None:
    """질의마다 실제로 hit된(검색된) 문서를 사람이 눈으로 검토할 수 있게 출력한다.

    Hit@K 같은 자동 지표는 "정답 문서에서 나온 청크인가"만 보므로, 그 청크 내용이
    실제로 질문에 답이 되는지는 사람이 직접 읽어봐야 판단할 수 있다. 이 함수는 채점이
    아니라 검토용 출력이고, 판단 결과는 retrieval_eval_results.json의 각 질의
    "human_feedback" 필드에 직접 적어 넣으면 된다(기본값 None).
    """
    print("\n=== 질의별 hit 문서 (사람 검토용) ===")
    for r in report["queries"]:
        mark = "OK" if r["rank_of_correct_doc"] == 1 else "MISS"
        print(f"\n[{mark}] 질의: {r['query']}  (정답 문서: {r['expected_doc_id']})")
        for hit in r["hits"]:
            flag = "*" if hit["doc_id"] == r["expected_doc_id"] else " "
            print(
                f"  {flag}{hit['rank']}위 {hit['doc_id']:<16}{hit['chunk_id']:<28}"
                f"{hit['citation']:<14}dist={hit['distance']:.3f}"
            )
            print(f"      {hit['snippet']}")


def _print_report(report: dict) -> None:
    print("\n=== 문서별 검색 성능 ===")
    print(f"{'doc_id':<20}{'n':>4}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}")
    for doc_id, s in report["per_document"].items():
        print(f"{doc_id:<20}{s['n_queries']:>4}{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['hit@5']:>8.2f}{s['mrr']:>8.2f}")

    o = report["overall"]
    print("\n=== 전체 ===")
    print(f"질의 수: {o['n_queries']}")
    print(f"Hit@1={o['hit@1']:.2f}  Hit@3={o['hit@3']:.2f}  Hit@5={o['hit@5']:.2f}  MRR={o['mrr']:.2f}")

    # 정답 문서가 top_k 안에도 안 들어온(완전히 놓친) 질의 목록
    misses = [r for r in report["queries"] if r["rank_of_correct_doc"] is None]
    if misses:
        print(f"\n=== 검색 실패(top-{TOP_K} 밖) 질의 {len(misses)}건 ===")
        for m in misses:
            print(f"- [{m['expected_doc_id']}] {m['query']}  (1위로 대신 나온 문서: {m['top1_doc_id']})")


if __name__ == "__main__":
    evaluate()
