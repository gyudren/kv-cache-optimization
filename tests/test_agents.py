from kv_eval.agents import market, stakeholder
from kv_eval.schemas import MarketAssessment, StakeholderAssessment
from kv_eval.state import initial_state

class WebStub:
    def __init__(self):
        self.market_calls = 0
        self.stakeholder_calls = 0
    def search_market(self, query):
        self.market_calls += 1
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
    assert web.market_calls == 6
    assert "market_result" in result
    assert result["market_result"]["sufficient"]
    assert all(x["source_type"] == "web" for x in result["evidence"])


def test_stakeholder_uses_web_only():
    web = WebStub()
    result = stakeholder.stakeholder_node(initial_state("query"), web, LLMStub())
    assert web.stakeholder_calls == 6
    assert result["stakeholder_result"]["sufficient"]
    assert all(x["source_type"] == "web" for x in result["evidence"])
