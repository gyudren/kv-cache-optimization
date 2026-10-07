"""D-16·D-17: Supervisor 충분성 검사는 최신 시도 출처만 세고, 재검색 질의는 부족한 기술에만 구조화된 힌트로 붙는다."""
from __future__ import annotations
import re

from fakes import FakeLLM, FakeWeb
from kv_eval.agents.market import MARKET_QUERIES
from kv_eval.config import RETRY_LIMITS
from kv_eval.observability import read_decisions
from kv_eval.supervisor.policy import SHORTFALL_HINT

BASE = {q for qs in MARKET_QUERIES.values() for q, _ in qs}
MLA, ITME = "DeepSeek-V2 MLA", "ITME CXL hybrid memory"


def _retry_queries(web):
    return [q for q in web.queries if q not in BASE and q.startswith((MLA, ITME))]


def test_one_source_per_attempt_never_accumulates_to_sufficient(run_graph):
    web, llm = FakeWeb(), FakeLLM(cite_limit={"market": {"mla": 1, "itme": 1}})
    state = run_graph(llm, web=web)
    # 매 시도 출처 1개 → 누적하면 2회차에 통과하지만, 최신 시도 기준이라 한도까지 재조사 후 제외된다
    assert llm.runs["market"] == 1 + RETRY_LIMITS["market"]
    assert state["perspective_status"]["market"] == "excluded"
    assert any(g.startswith("market:") and "고유 출처 1개" in g for g in state["gaps"])
    assert state["perspectives"]["market"]["source_units"]["mla"] and len(state["perspectives"]["market"]["source_units"]["mla"]) == 1


def test_shortfall_reasons_are_not_search_queries(run_graph):
    web, llm = FakeWeb(), FakeLLM(cite_limit={"market": {"mla": 1, "itme": 1}})
    run_graph(llm, web=web)
    queries = _retry_queries(web)
    assert queries, "재조사 질의가 있어야 한다"
    for q in queries:
        assert "고유 출처" not in q and "<2" not in q and "market/" not in q  # 사유 문장이 검색어가 되지 않는다
        assert SHORTFALL_HINT in q
        if q.startswith(MLA):
            assert not re.search(r"itme|cxl", q[len(MLA):], re.I)
        else:
            assert not re.search(r"mla|deepseek", q[len(ITME):], re.I)


def test_only_short_technology_gets_rework_queries(run_graph):
    web, llm = FakeWeb(), FakeLLM(cite_limit={"market": {"itme": 1}})
    state = run_graph(llm, web=web)
    queries = _retry_queries(web)
    assert queries and all(q.startswith(ITME) for q in queries)  # MLA는 충분하므로 재검색 질의 없음
    feedback = state["feedback"]["market"]
    assert set(feedback["queries_by_tech"]) == {"itme"}
    assert any("market/itme" in m for m in feedback["missing"])
    assert "dispatch:market" in [d["decision"] for d in read_decisions(state["trace_id"])]
