import json
import pytest
from pathlib import Path
from kv_eval.rag.ingest import (PaperSpec, clean_formula_noise, corpus_stats,
                                load_processed_chunks, paper_manifest)
from kv_eval.rag.retrieve import rrf_fuse, permitted_technologies, RetrievedChunk
from kv_eval.rag.workflow import RAGWorkflow
from kv_eval.schemas import QueryPlan, Relevance, Rewrite, RAGResponse

MANIFEST = {"page_budget": 200, "documents": [
    {"doc_id": "deepseek_v2_mla", "title": "DeepSeek-V2 (MLA)", "camp": "SW", "role": "primary", "expected_pages": 52},
    {"doc_id": "itme", "title": "ITME", "camp": "HW", "role": "primary", "expected_pages": 13},
    {"doc_id": "infinigen", "title": "InfiniGen", "camp": "HW", "role": "baseline", "expected_pages": 18},
    {"doc_id": "cxl_pnm", "title": "CXL-PNM", "camp": "HW", "role": "baseline", "expected_pages": 13},
]}


def _corpus(tmp_path: Path) -> tuple[Path, list[PaperSpec]]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    records = [
        {"chunk_id": f"{doc['doc_id']}_text_0000", "doc_id": doc["doc_id"], "content_type": "text",
         "start_page": 3, "end_page": 3, "text": "Reported memory expansion result for evaluation."}
        for doc in MANIFEST["documents"]
    ]
    chunks_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return chunks_path, paper_manifest(manifest_path)


def test_filter_isolation():
    assert permitted_technologies("mla") == {"mla"}
    assert permitted_technologies("itme") == {"itme"}
    assert permitted_technologies("itme_baseline") == {"itme", "baseline"}
    with pytest.raises(ValueError):
        permitted_technologies("all")


def test_manifest_assigns_technology_and_citation_number(tmp_path: Path):
    _, manifest = _corpus(tmp_path)
    assert [(s.doc_id, s.technology, s.citation_number) for s in manifest] == [
        ("deepseek_v2_mla", "mla", 1), ("itme", "itme", 2),
        ("infinigen", "baseline", 3), ("cxl_pnm", "baseline", 4)]


def test_rrf_fusion():
    ranked = rrf_fuse(["a", "b"], ["b", "a"])
    assert set(k for k, _ in ranked) == {"a", "b"}


def test_formula_noise_lines_removed_but_sentences_kept():
    cleaned = clean_formula_noise("Standard MHA first produces keys.\n\U0001D407\U0001D407\U0001D461\n4 …\nDuring inference.")
    assert cleaned == "Standard MHA first produces keys.\nDuring inference."


def test_processed_chunks_loaded_with_provenance(tmp_path: Path):
    chunks_path, manifest = _corpus(tmp_path)
    chunks = load_processed_chunks(chunks_path, manifest)
    assert len(chunks) == 4
    assert {c["technology"] for c in chunks} == {"mla", "itme", "baseline"}
    assert all(c["page"] == 3 for c in chunks)
    mla = next(c for c in chunks if c["doc_id"] == "deepseek_v2_mla")
    assert mla["citation_number"] == 1


def test_missing_preprocessing_output_rejected(tmp_path: Path):
    _, manifest = _corpus(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_processed_chunks(tmp_path / "absent.jsonl", manifest)


def test_page_budget_enforced(tmp_path: Path):
    chunks_path, manifest = _corpus(tmp_path)
    chunks = load_processed_chunks(chunks_path, manifest)
    summary_path = tmp_path / "summary.json"
    documents = [{"doc_id": d["doc_id"], "total_pages": d["expected_pages"],
                  "indexed_pages": d["expected_pages"], "excluded_reference_pages": 0}
                 for d in MANIFEST["documents"]]
    summary_path.write_text(json.dumps(
        {"total_pages": 96, "needs_manual_review": False, "documents": documents}), encoding="utf-8")
    stats = corpus_stats(chunks, manifest, summary_path)
    assert stats["total_pages"] == 96 and stats["page_budget"] == 200
    assert stats["chunks_per_document"]["itme"] == 1

    summary_path.write_text(json.dumps(
        {"total_pages": 201, "needs_manual_review": False, "documents": documents}), encoding="utf-8")
    with pytest.raises(ValueError):
        corpus_stats(chunks, manifest, summary_path)


class FakeRetriever:
    def __init__(self, count):
        self.count = count
        self.queries = []

    def hybrid_search(self, query, filter_, k):
        self.queries.append(query)
        return [RetrievedChunk("itme", 2, "verified excerpt", "itme", 0.03, 2, f"id{i}")
                for i in range(self.count)]


class FakeLLM:
    def generate_structured(self, prompt, schema):
        if schema is QueryPlan:
            return QueryPlan(english_query="ITME memory")
        if schema is Relevance:
            return Relevance(relevant_ids=["id0", "id1"])
        if schema is Rewrite:
            return Rewrite(english_query="ITME revised")
        if schema is RAGResponse:
            return RAGResponse(answer="Reported data", cited_ids=["id0"])
        raise AssertionError(schema)


def test_rag_rewrite_max_two_and_missing():
    retriever = FakeRetriever(1)
    ans = RAGWorkflow(retriever, FakeLLM()).rag_answer("질문", "itme")
    assert ans["search_attempts"] == 3
    assert ans["answer"] == "근거 부족"
    assert not ans["sufficient"]
    assert len(retriever.queries) == 3


def test_rag_valid_cited_ids():
    ans = RAGWorkflow(FakeRetriever(2), FakeLLM()).rag_answer("질문", "itme")
    assert ans["search_attempts"] == 1
    assert ans["sufficient"]
    assert len(ans["evidence"]) == 1
    assert "[2, p.2]" in ans["answer"]
