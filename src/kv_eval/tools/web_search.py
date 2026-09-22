"""외부 검색 도구 (Tavily). 시장 평가·이해관계자 평가 Agent 공용.

설계 B-2에서 두 Agent는 RAG 여부 X로, 논문 풀 대신 웹 검색만 근거로 쓴다
(논문 4편에는 시장 규모·채택 현황·이해관계자 발언 근거가 없기 때문).
이 모듈은 근거 수집까지만 담당하고, 긍정/우려/혼재 판정은 Agent의 LLM이 한다.

반환 항목은 Agent가 evidence로 바로 옮길 수 있도록 정규화한다:
    url, title, excerpt, publisher, published_at, speaker
"""
from __future__ import annotations
from urllib.parse import urlparse

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# 어느 관점에서도 입장 근거가 되지 못하는 영상·강의 사이트
DEFAULT_EXCLUDED_DOMAINS = ["youtube.com", "udemy.com", "coursera.org"]

# 이해관계자 전용 추가 제외: 프로필 페이지는 입장 표명이 아니고, 논문은 RAG 담당
STAKEHOLDER_EXCLUDED_DOMAINS = DEFAULT_EXCLUDED_DOMAINS + ["linkedin.com", "arxiv.org"]

MIN_SCORE = 0.4  # Tavily 관련도 점수가 이보다 낮은 결과는 근거로 쓰지 않는다
MAX_RESULTS = 4

# 발언 주체 구분용 도메인 힌트(설계 C-3: 발언 주체를 함께 기록).
# 최종 귀속은 Agent가 본문을 보고 판단하며, 여기서는 후보만 붙인다.
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
        if any(pattern in lowered for pattern in patterns):
            return speaker
    return "기타"


def _publisher(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _normalize(hit: dict) -> dict:
    """Tavily 응답 1건을 Agent가 쓰는 근거 형식으로 바꾼다."""
    url = hit.get("url", "")
    return {
        "url": url,
        "title": hit.get("title", ""),
        "excerpt": hit.get("content", ""),
        "publisher": _publisher(url),
        # 게시일이 없는 결과가 많다. 없으면 빈 값으로 두고 REFERENCE에서 "게시일 미확인"으로 표기된다.
        "published_at": hit.get("published_date", "") or "",
        "speaker": speaker_hint(url),
        "score": hit.get("score"),
    }


class WebSearch:
    """시장·이해관계자 Agent가 쓰는 Tavily 검색 클라이언트."""

    def __init__(self, api_key: str, client: object | None = None):
        if not api_key and client is None:
            raise ValueError("TAVILY_API_KEY is required")
        self.api_key = api_key
        self._client = client  # 테스트용 주입 지점. 실제 실행에서는 requests를 쓴다.

    def _post(self, payload: dict) -> list[dict]:
        if self._client is not None:
            return self._client.search(payload)
        import requests

        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("results", [])

    def _search(self, query: str, *, topic: str, exclude_domains: list[str],
                max_results: int = MAX_RESULTS, min_score: float = MIN_SCORE) -> list[dict]:
        hits = self._post({
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "topic": topic,
            "exclude_domains": exclude_domains,
            "include_answer": False,
        })
        results = [_normalize(hit) for hit in hits
                   if (hit.get("score") or 0) >= min_score and hit.get("url")]
        # 근거로 쓸 수 없는 빈 본문은 버린다(인용해도 검증이 불가능하므로).
        return [item for item in results if item["excerpt"].strip()]

    def search_market(self, query: str, topic: str = "news") -> list[dict]:
        """C-2 시장 규모·성장성, 상용화·채택, 생태계 지지 근거 수집.

        생태계 지지(공식 문서·릴리스 노트)는 뉴스가 아니므로 topic="general"로 찾는다.
        """
        return self._search(query, topic=topic, exclude_domains=DEFAULT_EXCLUDED_DOMAINS)

    def search_stakeholder(self, query: str) -> list[dict]:
        """C-3 경쟁사·개발자/도입기업·투자업계의 '발언' 근거 수집."""
        return self._search(query, topic="general", exclude_domains=STAKEHOLDER_EXCLUDED_DOMAINS)
