from kv_eval.agents import market, stakeholder
from kv_eval.schemas import MarketAssessment, StakeholderAssessment
from kv_eval.state import initial_state

class WebStub:
    def __init__(self):
        self.market_calls = 0
        self.stakeholder_calls = 0
    def search_market(self, query, topic="news"):
        self.market_calls += 1
        self.queries = getattr(self, "queries", []) + [query]
        return [{"title": "Industry comment", "url": f"https://example.org/market/{self.market_calls}", "publisher": "Org", "published_at": "", "excerpt": "Verifiable statement"}]
    def search_stakeholder(self, query):
        self.stakeholder_calls += 1
        return [{"title": "Named developer", "url": f"https://example.org/stakeholder/{self.stakeholder_calls}", "publisher": "Org", "published_at": "", "excerpt": "Verifiable statement"}]

class LLMStub:
    def generate_structured(self, prompt, schema):
        import re
        cites = re.findall(r"(web:[\w:]+):", prompt)
        if schema is MarketAssessment:
            return MarketAssessment(summary="market", market_size_growth="growth", adoption="adoption", ecosystem="ecosystem",
                                    sufficient=True, cited_ids=cites[:1])
        if schema is StakeholderAssessment:
            return StakeholderAssessment(summary="stakeholders", competitors="reactions", developers_adopters="reviews",
                                         investors="comment", sufficient=True, cited_ids=cites[:1])
        raise AssertionError(schema)


def test_market_uses_web_only():
    web = WebStub()
    result = market.market_node(initial_state("query"), web, LLMStub())
    assert web.market_calls == 10  # 기술별 M1~M3 검색어 5개
    assert "market_result" in result
    assert result["market_result"]["sufficient"]
    assert all(x["source_type"] == "web" for x in result["evidence"])


def test_stakeholder_uses_web_only():
    web = WebStub()
    result = stakeholder.stakeholder_node(initial_state("query"), web, LLMStub())
    assert web.stakeholder_calls == 8  # 기술별 S1~S3 검색어 4개
    assert result["stakeholder_result"]["sufficient"]
    assert all(x["source_type"] == "web" for x in result["evidence"])


def test_retry_queries_are_short_standalone_searches():
    from kv_eval.tools import retry_queries
    feedback = {"rewritten_queries": ["MLA 자체의 제품 출시: 공식 발표 필요 " * 10, "vLLM 직접 지원 문서", "third"]}
    queries = retry_queries("DeepSeek-V2 MLA", feedback)
    assert len(queries) == 2
    assert all("[" not in q and len(q) < 140 for q in queries)


def test_market_retry_does_not_embed_python_list_in_query():
    state = initial_state("query")
    state["review_feedback"] = {"market": {"rewritten_queries": ["M1 시장조사 자료 필요"]}}
    web = WebStub()
    market.market_node(state, web, LLMStub())
    assert any(q.endswith("M1 시장조사 자료 필요") for q in web.queries)
    assert not any("['" in q for q in web.queries)


def test_domain_items_normalize_llm_dimension_names_and_citations():
    from kv_eval.agents.domain import normalize_items
    evidence = [{"source_id": "deepseek_v2_mla_text_0016", "citation_number": 1, "page": 16}]
    items = normalize_items([{"dimension": "MLA D1 워크로드 수용", "technology": "mla", "verdict": "적합",
                              "explanation": "", "cited_ids": ["[1, p.16]", "[9, p.9]"]}], evidence)
    assert items[0]["dimension"] == "D1 워크로드 수용 능력"
    assert items[0]["cited_ids"] == ["deepseek_v2_mla_text_0016", "[9, p.9]"]  # 없는 인용은 검증에서 걸리도록 유지
