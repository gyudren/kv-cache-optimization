"""Tavily 웹 검색 도구.

근거 수집까지만 하고 긍정/우려/혼재 판정은 각 Agent의 LLM이 한다.
"""
from __future__ import annotations
import json
import os
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# 영상·강의 사이트와 SNS는 검증 가능한 발췌가 남지 않아 뺀다.
DEFAULT_EXCLUDED_DOMAINS = ["youtube.com", "udemy.com", "coursera.org",
                            "instagram.com", "facebook.com", "tiktok.com", "x.com", "twitter.com", "pinterest.com"]

# 이해관계자 검색에서는 프로필 페이지와 논문(RAG 담당)도 뺀다.
STAKEHOLDER_EXCLUDED_DOMAINS = DEFAULT_EXCLUDED_DOMAINS + ["linkedin.com", "arxiv.org"]

MIN_SCORE = 0.4  # Tavily 관련도 하한
MAX_RESULTS = 4

# 개인 의견 글. 개발자 반응의 보조 근거로만 쓴다.
COMMUNITY_SPEAKER = "개인·커뮤니티 글"

# URL로 붙이는 발언 주체 후보. 최종 판단은 Agent가 본문을 보고 한다.
# 처음 맞는 유형을 쓰므로 커뮤니티 글을 먼저 본다(discussion.fool.com은 투자 매체가 아니라 토론 게시판).
SPEAKER_HINTS = {
    COMMUNITY_SPEAKER: ("reddit.com", "news.ycombinator.com", "stackoverflow.com", "discussion.", "forum.",
                        "forums.", "community.", ".github.io", "medium.com", "substack.com", "tistory.com",
                        "velog.io", "brunch.co.kr", "dev.to", "hashnode."),
    "언론": ("reuters.", "bloomberg.", "cnbc.", "zdnet.", "theelec.", "hankyung.",
             "mk.co.kr", "etnews.", "chosun.", "yna.co.kr", "techcrunch."),
    "개발자 공식 저장소": ("github.com", "huggingface.co"),
    "기업·기술 블로그": ("blog.", "/blog"),
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
        # 게시일이 없는 결과가 많다. 빈 값은 REFERENCE에서 "게시일 미확인"으로 표기된다.
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
        self._client = client  # 테스트용 주입 지점

    def _post(self, payload: dict) -> list[dict]:
        if self._client is not None:
            return self._client.search(payload)
        import requests

        # 같은 질의는 디스크 캐시를 읽어 크레딧을 아끼고 같은 근거로 재현한다. WEB_CACHE_DIR를 비우면 캐시를 끈다.
        cache_file = None
        cache_dir = os.getenv("WEB_CACHE_DIR", "data/cache/tavily")
        if cache_dir:
            key = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
            cache_file = Path(cache_dir) / f"{key}.json"
            if cache_file.is_file():
                return json.loads(cache_file.read_text(encoding="utf-8")).get("results", [])

        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30,
        )
        if response.status_code in (429, 432, 433):
            detail = response.text[:200]
            raise RuntimeError(f"Tavily 사용 한도 초과(HTTP {response.status_code}): {detail} "
                               "— Tavily 대시보드에서 한도를 확인하거나 다른 TAVILY_API_KEY를 .env에 넣으세요")
        response.raise_for_status()
        results = response.json().get("results", [])
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"query": payload.get("query"), "results": results},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
        return results

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
        # 본문이 빈 결과는 인용해도 검증할 수 없어 버린다.
        return [item for item in results if item["excerpt"].strip()]

    def search_market(self, query: str, topic: str = "news") -> list[dict]:
        """시장 규모·채택·생태계 근거를 검색한다.

        공식 문서·릴리스 노트는 뉴스가 아니므로 topic="general"로 찾는다.
        """
        return self._search(query, topic=topic, exclude_domains=DEFAULT_EXCLUDED_DOMAINS)

    def search_stakeholder(self, query: str) -> list[dict]:
        """경쟁사·개발자·도입 기업·투자 업계의 발언 근거를 검색한다."""
        return self._search(query, topic="general", exclude_domains=STAKEHOLDER_EXCLUDED_DOMAINS)
