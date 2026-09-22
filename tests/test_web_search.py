import pytest
from kv_eval.tools.web_search import (MARKET_EXCLUDED_DOMAINS, STAKEHOLDER_EXCLUDED_DOMAINS,
                                      WebSearch, speaker_hint)


class FakeTavily:
    """Tavily 응답을 흉내 내는 주입용 클라이언트(네트워크 호출 없음)."""

    def __init__(self, hits):
        self.hits = hits
        self.payloads = []

    def search(self, payload):
        self.payloads.append(payload)
        return self.hits


HITS = [
    {"url": "https://www.reuters.com/a", "title": "CXL memory demand", "content": "시장 전망 본문",
     "score": 0.9, "published_date": "2026-01-05"},
    {"url": "https://blog.example.com/b", "title": "developer notes", "content": "도입 후기 본문", "score": 0.6},
    {"url": "https://spam.example.com/c", "title": "low relevance", "content": "무관", "score": 0.1},
    {"url": "https://empty.example.com/d", "title": "no body", "content": "   ", "score": 0.8},
]


def test_missing_key_uses_fallback_without_fabricated_scores():
    results = WebSearch("", fallback=lambda *a, **kw: [HITS[0]]).search_market("query")
    assert results[0]["provider"] == "duckduckgo"
    assert results[0]["score"] is None
    assert results[0]["excerpt"] == HITS[0]["content"]


def test_market_search_normalizes_and_filters():
    fake = FakeTavily(HITS)
    results = WebSearch("key", client=fake).search_market("DeepSeek-V2 MLA market size")
    # 관련도 미달(0.1)과 본문 없는 결과는 근거로 쓰지 않는다.
    assert [r["url"] for r in results] == ["https://www.reuters.com/a", "https://blog.example.com/b"]
    first = results[0]
    assert first["excerpt"] == "시장 전망 본문"
    assert first["publisher"] == "reuters.com"
    assert first["published_at"] == "2026-01-05"
    assert first["speaker"] == "언론"
    # 게시일이 없으면 빈 값으로 두고 REFERENCE 단계에서 "게시일 미확인"으로 표기된다.
    assert results[1]["published_at"] == ""
    assert fake.payloads[0]["exclude_domains"] == MARKET_EXCLUDED_DOMAINS


def test_stakeholder_search_excludes_profiles_and_papers():
    fake = FakeTavily(HITS)
    WebSearch("key", client=fake).search_stakeholder("ITME 개발자 평가")
    assert fake.payloads[0]["exclude_domains"] == STAKEHOLDER_EXCLUDED_DOMAINS
    assert "arxiv.org" in fake.payloads[0]["exclude_domains"]


def test_speaker_hint_classifies_known_domains():
    assert speaker_hint("https://news.ycombinator.com/item?id=1") == "개발자 커뮤니티"
    assert speaker_hint("https://www.skhynix.com/news") == "기업 공식 발표"
    assert speaker_hint("https://unknown.example.org/post") == "기타"


def test_rate_limit_switches_to_fallback_and_skips_further_tavily_calls():
    import requests
    class Limited:
        calls = 0
        def search(self, payload):
            self.calls += 1
            response = requests.Response()
            response.status_code = 429
            raise requests.HTTPError(response=response)
    client = Limited()
    web = WebSearch("key", client=client, fallback=lambda *a, **kw: [HITS[0]])
    with pytest.warns(RuntimeWarning, match="Tavily"):
        assert web.search_market("first")[0]["provider"] == "duckduckgo"
    assert web.search_market("second")
    assert client.calls == 1


def test_fallback_failure_is_missing_evidence():
    def broken(*a, **kw):
        raise RuntimeError("unavailable")
    with pytest.warns(RuntimeWarning, match="대체 검색 실패"):
        assert WebSearch("", fallback=broken).search_market("query") == []


def test_fallback_filters_excluded_hosts_but_not_url_path():
    hits = [
        {"url": "https://www.reddit.com/a", "content": "excluded"},
        {"url": "https://example.org/reddit.com", "content": "valid"},
        {"url": "https://example.org/empty", "content": " "},
    ]
    web = WebSearch("", fallback=lambda *a, **kw: hits)
    assert [r["url"] for r in web.search_market("query")] == ["https://example.org/reddit.com"]
