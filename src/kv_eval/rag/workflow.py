"""Inner Agentic RAG workflow. Not a top-level graph agent."""
from __future__ import annotations
from typing import Any
from .retrieve import HybridRetriever, RetrievedChunk
from ..schemas import QueryPlan, Relevance, Rewrite, RAGResponse
from ..config import MIN_RELEVANT, RAG_REWRITES, RETRIEVAL_K

class RAGWorkflow:
    def __init__(self, retriever: HybridRetriever, llm: Any):
        self.retriever = retriever
        self.llm = llm

    def rag_answer(self, question: str, technology_filter: str, feedback: dict | None = None) -> dict:
        feedback = feedback or {}
        plan = self.llm.generate_structured(
            f"Translate/plan this Korean research question to precise English technical search terms, without changing its scope.\nQuestion: {question}\nMissing: {feedback.get('missing', [])}", QueryPlan)
        query = plan.english_query
        attempts = 0
        relevant: list[RetrievedChunk] = []
        for attempt in range(RAG_REWRITES + 1):
            attempts += 1
            retrieved = self.retriever.hybrid_search(query, technology_filter, RETRIEVAL_K)
            if retrieved:
                candidates = "\n".join(f"ID: {c.chunk_id} | [{c.citation_number}, p.{c.page}] {c.text[:1100]}" for c in retrieved)
                judgement = self.llm.generate_structured(
                    f"Select IDs of passages directly relevant to this exact question; do not invent IDs.\nQuestion: {question}\n{candidates}", Relevance)
                allowed = set(judgement.relevant_ids)
                relevant = [c for c in retrieved if c.chunk_id in allowed]
            else:
                relevant = []
            if len(relevant) >= MIN_RELEVANT:
                break
            if attempt < RAG_REWRITES:
                rewritten = self.llm.generate_structured(
                    f"Rewrite this English academic-search query to find missing evidence (avoid same phrasing). Question: {question}\nPrevious: {query}\nMissing: {feedback.get('missing', [])}\nRelevant snippets: {len(relevant)}", Rewrite)
                query = rewritten.english_query
        if len(relevant) < MIN_RELEVANT:
            return {"answer": "근거 부족", "evidence": [], "sufficient": False,
                    "missing": [f"{question}: 관련 청크 {len(relevant)}개(<{MIN_RELEVANT})"], "search_attempts": attempts}
        available = {c.chunk_id: c for c in relevant}
        context = "\n".join(f"ID: {c.chunk_id} | [{c.citation_number}, p.{c.page}] {c.text}" for c in relevant)
        generated = self.llm.generate_structured(
            f"Answer the question in Korean using ONLY the cited passages. Cite [n, p.X] from source metadata, include exact source IDs in cited_ids. Missing facts must go to missing. No unsupported quantitative results.\nQuestion: {question}\n{context}", RAGResponse)
        valid = [available[cid] for cid in generated.cited_ids if cid in available]
        if not valid:
            return {"answer": "근거 부족", "evidence": [], "sufficient": False,
                    "missing": [f"{question}: 응답이 검증 가능한 문서를 인용하지 않음"], "search_attempts": attempts}
        evidence = [{"source_id": c.chunk_id, "claim": question, "excerpt": c.text,
                     "doc_id": c.doc_id, "page": c.page, "technology": c.technology,
                     "citation_number": c.citation_number, "source_type": "paper"} for c in valid]
        # Deterministic citation suffix ensures answer is traceable even if model omitted citations.
        citations = " ".join(f"[{c.citation_number}, p.{c.page}]" for c in valid)
        return {"answer": f"{generated.answer}\n근거: {citations}", "evidence": evidence,
                "sufficient": not bool(generated.missing), "missing": generated.missing,
                "search_attempts": attempts}


def answer_with_cache(rag: Any, cache: dict, question: str, technology_filter: str, feedback: dict | None) -> dict:
    """재시도 시 이미 근거가 충분했던 질문은 이전 답변을 재사용한다.

    Master 게이트가 일부 항목만 부족하다고 판정해도 모든 질문을 다시 검색·답변하면
    LLM 호출의 절반 이상이 같은 답을 반복하는 데 쓰인다(설계 D-2: 부족 항목만 재검색).
    sufficient=False 였던 질문만 부족 항목 피드백과 함께 다시 실행한다.
    """
    previous = cache.get(question)
    if previous and previous.get("sufficient"):
        return previous
    return rag.rag_answer(question, technology_filter, feedback)
