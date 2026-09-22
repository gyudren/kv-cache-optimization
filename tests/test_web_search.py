import pytest
from kv_eval.tools.web_search import (DEFAULT_EXCLUDED_DOMAINS, STAKEHOLDER_EXCLUDED_DOMAINS,
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


def test_requires_credentials():
    with pytest.raises(ValueError):
        WebSearch("")


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
    assert fake.payloads[0]["exclude_domains"] == DEFAULT_EXCLUDED_DOMAINS


def test_stakeholder_search_excludes_profiles_and_papers():
    fake = FakeTavily(HITS)
    WebSearch("key", client=fake).search_stakeholder("ITME 개발자 평가")
    assert fake.payloads[0]["exclude_domains"] == STAKEHOLDER_EXCLUDED_DOMAINS
    assert "arxiv.org" in fake.payloads[0]["exclude_domains"]


def test_speaker_hint_classifies_known_domains():
    assert speaker_hint("https://news.ycombinator.com/item?id=1") == "개발자 커뮤니티"
    assert speaker_hint("https://www.skhynix.com/news") == "기업 공식 발표"
    assert speaker_hint("https://unknown.example.org/post") == "기타"
