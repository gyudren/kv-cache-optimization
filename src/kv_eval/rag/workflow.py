"""Agentic RAG workflow used inside research agents."""
from __future__ import annotations
import json
import re
from hashlib import sha256
from typing import Any
from ..tools import scoped_feedback
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
            f"Translate/plan this Korean research question to precise English technical search terms, without changing its scope.\nQuestion: {question}\nMissing: {feedback.get('missing', [])}\nSearch hints from the supervisor (cover these if relevant): {feedback.get('rewritten_queries', [])}", QueryPlan)
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
        # Append citations in code in case the model omitted them.
        citations = " ".join(f"[{c.citation_number}, p.{c.page}]" for c in valid)
        return {"answer": f"{generated.answer}\n근거: {citations}", "evidence": evidence,
                "sufficient": not bool(generated.missing), "missing": generated.missing,
                "search_attempts": attempts}


def feedback_fingerprint(feedback: dict | None) -> str:
    """재작업 지시(missing·재검색 질의)의 지문."""
    items = [*(feedback or {}).get("missing", []), *(feedback or {}).get("rewritten_queries", [])]
    if not items:
        return ""
    return sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def question_targeted(question: str, technology_filter: str, feedback: dict | None) -> bool:
    """재작업 지시가 이 질문을 겨냥하는가. 지시에 D-코드가 있으면 그 차원의 질문만 해당한다."""
    scoped = scoped_feedback(feedback, technology_filter)
    if not scoped:
        return False
    dims = {d for item in scoped.get("missing", []) for d in re.findall(r"\bD([1-7])\b", str(item))}
    dim = re.search(r"\bD([1-7])\b", question)
    return not dims or bool(dim and dim.group(1) in dims)


def answer_with_cache(rag: Any, cache: dict, question: str, technology_filter: str, feedback: dict | None) -> dict:
    """재시도 시 근거가 충분했던 질문은 이전 답변을 재사용한다.

    재작업 지시가 이 질문을 겨냥하면 같은 지시로 검색한 답일 때만 재사용한다.
    """
    scoped = scoped_feedback(feedback, technology_filter)
    fingerprint = feedback_fingerprint(scoped)
    targeted = bool(fingerprint) and question_targeted(question, technology_filter, feedback)
    previous = cache.get(question)
    if previous and previous.get("sufficient") and (not targeted or previous.get("feedback_fp") == fingerprint):
        return previous
    answer = rag.rag_answer(question, technology_filter, scoped)
    return {**answer, "feedback_fp": fingerprint if targeted else ""}
