import pytest
from pathlib import Path
from kv_eval.rag.ingest import PaperSpec, load_papers, chunk_pages, bibliography_starts, corpus_stats
from kv_eval.rag.retrieve import rrf_fuse, permitted_docs, RetrievedChunk
from kv_eval.rag.workflow import RAGWorkflow
from kv_eval.schemas import QueryPlan, Relevance, Rewrite, RAGResponse


def test_filter_isolation():
    # doc_id는 데이터 전처리 산출물(data/processed/chunks.jsonl)의 실제 값을 따른다.
    assert permitted_docs("mla") == {"deepseek_v2_mla"}
    assert permitted_docs("itme") == {"itme"}
    assert permitted_docs("itme_baseline") == {"itme", "infinigen", "cxl_pnm"}
    with pytest.raises(ValueError):
        permitted_docs("all")


def test_rrf_fusion_and_chunking():
    ranked = rrf_fuse(["a", "b"], ["b", "a"])
    assert set(k for k, _ in ranked) == {"a", "b"}
    text = "\n".join("long scientific sentence. " * 30 for _ in range(2))
    chunks = chunk_pages([{"doc_id": "itme", "page": 3, "text": text, "technology": "itme", "citation_number": 2}])
    assert len(chunks) >= 2
    assert all(ch["page"] == 3 for ch in chunks)
    assert all(len(ch["text"]) <= 1200 for ch in chunks)
    assert bibliography_starts("References\nSome paper")
    assert not bibliography_starts("This paper references recent work\nMore body")


def test_missing_file_rejected(tmp_path: Path):
    manifest = [PaperSpec(name, str(tmp_path/f"{name}.pdf"), tech, 1, i)
                for i, (name, tech) in enumerate((("deepseek_v2", "mla"), ("itme", "itme"),
                                                   ("infinigen", "baseline"), ("cxl_pnm", "baseline")), 1)]
    with pytest.raises(FileNotFoundError):
        load_papers(manifest)


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
