"""재작업 지시가 실제 재검색·프롬프트에 반영되는지(D-04), 스레드 컨텍스트가 전파되는지(D-02) 검증한다."""
from __future__ import annotations

from fakes import FakeLLM, FakeRAG
from kv_eval.rag.workflow import answer_with_cache, question_targeted


def test_tech_rework_re_searches_with_feedback(run_graph):
    rag, llm = FakeRAG(), FakeLLM(insufficient={"tech": 1})
    run_graph(llm, rag=rag)
    reworked = [c for c in rag.log if c["feedback"].get("missing")]
    assert reworked, "재작업 RAG 호출에 feedback이 실려야 한다"
    # 필수 결함 "tech/mla: TRL 판정·근거 미기재"(사유) + 세부 항목 "MLA 상용 채택 직접 근거"(MLA 검색 힌트)
    # → MLA 질문만 재검색(ITME 질문은 캐시 재사용)
    assert all(c["question"].startswith("[mla]") for c in reworked)
    assert any("TRL 판정·근거 미기재" in m for m in reworked[0]["feedback"]["missing"])
    assert "SUPERVISOR REWORK REQUEST" in llm.prompts["tech"][1]
    assert "SUPERVISOR REWORK REQUEST" not in llm.prompts["tech"][0]


def test_eval_driven_domain_rework_bypasses_cache_and_uses_hints(run_graph):
    rag, llm = FakeRAG(), FakeLLM(judge_fail={"bias_control": (1, "domain")})
    state = run_graph(llm, rag=rag)
    domain_calls = [c for c in rag.log if " D" in c["question"] and not c["question"].startswith("[")]
    first, rework = domain_calls[:14], domain_calls[14:]
    assert len(first) == 14 and len(rework) == 14  # Judge만 미달(기술 미특정) → 14개 전부 재검색
    assert all(c["feedback"].get("rewritten_queries") for c in rework)
    # 사유 문장이 아니라 힌트 질의로 검색한다(D-10)
    queries = state["feedback"]["domain"]["rewritten_queries"]
    assert queries and not any("bias_control" in q for q in queries)
    assert "SUPERVISOR REWORK REQUEST" in llm.prompts["domain"][-1]


def test_cache_key_includes_feedback_fingerprint():
    rag = FakeRAG()
    cache: dict = {}
    first = answer_with_cache(rag, cache, "itme D5 운영 안정성: q", "itme", None)
    cache["itme D5 운영 안정성: q"] = first
    assert answer_with_cache(rag, cache, "itme D5 운영 안정성: q", "itme", None) is first  # 지시 없음 → 재사용
    other = {"missing": ["mla/D1: 근거 부족"], "rewritten_queries": []}
    assert not question_targeted("itme D5 운영 안정성: q", "itme", other)
    assert answer_with_cache(rag, cache, "itme D5 운영 안정성: q", "itme", other) is first  # 다른 대상 → 재사용
    target = {"missing": ["domain/itme: D5 판정 한쪽뿐"], "rewritten_queries": ["limitations"]}
    again = answer_with_cache(rag, cache, "itme D5 운영 안정성: q", "itme", target)
    assert again is not first and again["feedback_fp"] and rag.calls == 2
    cache["itme D5 운영 안정성: q"] = again
    assert answer_with_cache(rag, cache, "itme D5 운영 안정성: q", "itme", target) is again  # 같은 지시 → 재사용


def test_worker_threads_inherit_graph_run_context(run_graph):
    rag = FakeRAG()
    state = run_graph(FakeLLM(), rag=rag)
    assert rag.log and all(c["config"] is not None for c in rag.log)
    assert {c["config"]["metadata"]["trace_id"] for c in rag.log} == {state["trace_id"]}
