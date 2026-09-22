"""외부 검색 도구 (Tavily). 시장 평가·이해관계자 평가 Agent 공용.

RAG를 쓰지 않는 두 Agent(B-2에서 RAG 여부 X)가 함께 쓴다.
- web_search : 공용 저수준 호출부 (Tavily, 키 없으면 DuckDuckGo로 자동 대체)
- WebClient : agents/market.py·agents/stakeholder.py가 기대하는
  ``web.search_market(query)`` / ``web.search_stakeholder(query)`` 인터페이스 어댑터
판정(긍정/우려/혼재)은 Agent의 LLM 몫이며 여기서는 근거만 모은다.
"""

import os
from urllib.parse import urlparse

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# 어느 관점에서도 입장 근거가 되지 못하는 영상·강의 사이트
DEFAULT_EXCLUDED_DOMAINS = ["youtube.com", "udemy.com", "coursera.org"]

# 이해관계자 전용 추가 제외: 프로필 페이지는 입장 표명이 아니고, 논문은 RAG 담당
STAKEHOLDER_EXCLUDED_DOMAINS = DEFAULT_EXCLUDED_DOMAINS + ["linkedin.com", "arxiv.org"]

# 시장 평가 전용 추가 제외: 커뮤니티 잡담은 시장 규모/채택 근거로 약함
MARKET_EXCLUDED_DOMAINS = DEFAULT_EXCLUDED_DOMAINS + ["reddit.com"]

# 발언 주체 구분용 도메인 힌트. 최종 귀속은 Agent가 본문을 보고 판단한다.
SPEAKER_HINTS = {
    "언론": ("reuters.", "bloomberg.", "cnbc.", "zdnet.", "theelec.", "hankyung.",
             "mk.co.kr", "etnews.", "chosun.", "yna.co.kr", "techcrunch."),
    "개발자 커뮤니티": ("reddit.com", "news.ycombinator.com", "stackoverflow.com",
                  "github.com", "huggingface.co"),
    "기업·기술 블로그": ("medium.com", "blog.", "/blog", "substack.com"),
    "기업 공식 발표": ("nvidia.com", "skhynix.com", "samsung.com", "deepseek.com",
                 "micron.com", "intel.com", "amd.com"),
    "투자·애널리스트": ("seekingalpha.com", "morningstar.", "fool.com", "marketwatch."),
}


def speaker_hint(url: str) -> str:
    lowered = url.lower()
    for speaker, patterns in SPEAKER_HINTS.items():
        if any(p in lowered for p in patterns):
            return speaker
    return "기타"


def _publisher_from_url(url: str) -> str:
    netloc = urlparse(url).netloc
    return netloc or "발행 주체 미확인"


def _ddg_search_fallback(
    query: str, *, max_results: int, exclude_domains: list[str]
) -> list[dict]:
    """TAVILY_API_KEY가 없을 때 쓰는 무료 대체 검색(DuckDuckGo, 키 불필요).

    README에 명시적으로 밝히는 도구 대체 — 재현성을 위해 기본값으로 동작하되,
    TAVILY_API_KEY가 있으면 항상 Tavily를 우선 사용한다.
    """
    from ddgs import DDGS

    excluded = tuple(exclude_domains)
    with DDGS() as ddgs:
        raw_hits = list(ddgs.text(query, max_results=max_results * 2))

    hits = []
    for rank, hit in enumerate(raw_hits):
        url = hit.get("href", "")
        if any(domain in url.lower() for domain in excluded):
            continue
        hits.append(
            {
                "title": hit.get("title", ""),
                "url": url,
                "content": hit.get("body", ""),
                # DDG는 점수를 주지 않으므로 순위 기반으로 합리적인 값을 근사한다.
                "score": max(0.3, 0.9 - rank * 0.07),
            }
        )
        if len(hits) >= max_results:
            break
    return hits


def web_search(
    query: str,
    *,
    max_results: int = 4,
    search_depth: str = "basic",
    topic: str = "general",
    exclude_domains: list[str] | None = None,
    min_score: float | None = None,
    include_answer: bool = False,
) -> list[dict]:
    """웹 검색 호출. TAVILY_API_KEY가 있으면 Tavily, 없으면 DuckDuckGo로 대체한다.

    min_score를 주면 그 미만은 걸러낸다.
    """
    exclude = exclude_domains or DEFAULT_EXCLUDED_DOMAINS
    api_key = os.environ.get("TAVILY_API_KEY")

    try:
        if api_key:
            response = requests.post(
                TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                    "topic": topic,
                    "exclude_domains": exclude,
                    "include_answer": include_answer,
                },
                timeout=30,
            )
            response.raise_for_status()
            hits = response.json().get("results", [])
        else:
            hits = _ddg_search_fallback(query, max_results=max_results, exclude_domains=exclude)
    except Exception as exc:
        # 검색 엔진의 일시적 오류로 전체 실행이 죽지 않게 한다. 검색 실패는 evidence
        # 부족으로 이어져 sufficient=false / "근거 부족"으로 자연스럽게 처리된다.
        print(f"[web_search] '{query}' 검색 실패, 이 질의는 건너뜀: {exc}")
        hits = []

    if min_score is None:
        return hits
    return [h for h in hits if (h.get("score") or 0) >= min_score]


class WebClient:
    """market/stakeholder Agent가 기대하는 얇은 검색 어댑터.

    agents/market.py·agents/stakeholder.py는 ``web.search_market(query)`` /
    ``web.search_stakeholder(query)`` 형태로 질의 문자열 하나를 넘기고
    ``{url, title, excerpt, publisher, published_at}`` 딕셔너리 리스트를 기대한다.
    실제 검색은 위 ``web_search()``(Tavily, 키 없으면 DuckDuckGo)를 그대로 쓰고
    필드만 이 계약에 맞게 다시 포장한다.
    """

    def __init__(self, max_results: int = 4, min_score: float | None = None):
        self.max_results = max_results
        self.min_score = min_score

    def _search(self, query: str, *, exclude_domains: list[str]) -> list[dict]:
        hits = web_search(
            query,
            max_results=self.max_results,
            exclude_domains=exclude_domains,
            min_score=self.min_score,
        )
        return [
            {
                "url": h.get("url", ""),
                "title": h.get("title", ""),
                "excerpt": h.get("content", ""),
                "publisher": _publisher_from_url(h.get("url", "")),
                "published_at": h.get("published_date") or h.get("published_at"),
            }
            for h in hits
            if h.get("url")
        ]

    def search_market(self, query: str) -> list[dict]:
        return self._search(query, exclude_domains=MARKET_EXCLUDED_DOMAINS)

    def search_stakeholder(self, query: str) -> list[dict]:
        return self._search(query, exclude_domains=STAKEHOLDER_EXCLUDED_DOMAINS)
