"""외부 검색 도구 (Tavily). 시장 평가·이해관계자 평가 Agent 공용.

RAG를 쓰지 않는 두 Agent(B-2에서 RAG 여부 X)가 함께 쓴다.
- web_search / to_evidence : 공용 호출부. D-1 evidence 필수 필드를 한곳에서 강제한다.
- search_stakeholder_views : C-3 관점 3종의 근거 수집 (이해관계자 전용)
판정(긍정/우려/혼재)은 Agent의 LLM 몫이며 여기서는 근거만 모은다.
"""

import os
from typing import Iterable

import requests

from kv_eval import config
from kv_eval.state import source_id, to_evidence

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# 어느 관점에서도 입장 근거가 되지 못하는 영상·강의 사이트
DEFAULT_EXCLUDED_DOMAINS = ["youtube.com", "udemy.com", "coursera.org"]

# 이해관계자 전용 추가 제외: 프로필 페이지는 입장 표명이 아니고, 논문은 RAG 담당
STAKEHOLDER_EXCLUDED_DOMAINS = DEFAULT_EXCLUDED_DOMAINS + ["linkedin.com", "arxiv.org"]

STAKEHOLDER_AGENT = "stakeholder"
STAKEHOLDER_NODE = "stakeholder_eval"

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
    """Tavily search 호출. min_score를 주면 그 미만은 걸러낸다."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 환경변수가 설정되지 않았습니다.")

    response = requests.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "exclude_domains": exclude_domains or DEFAULT_EXCLUDED_DOMAINS,
            "include_answer": include_answer,
        },
        timeout=30,
    )
    response.raise_for_status()
    hits = response.json().get("results", [])

    if min_score is None:
        return hits
    return [h for h in hits if (h.get("score") or 0) >= min_score]


def _build_queries(
    tech_key: str, category_key: str, rewritten_queries: dict[str, list[str]] | None
) -> list[str]:
    tech = config.TECHNOLOGIES[tech_key]
    category = config.STAKEHOLDER_CATEGORIES[category_key]
    queries = []
    for level in category["levels"]:
        queries.append(f"{tech[f'{level}_ko']} {category['ko_terms']}")
        queries.append(f"{tech[f'{level}_en']} {category['en_terms']}")
    # review_feedback의 재작성 질의는 Master가 만든 완성형이므로 그대로 쓴다
    queries.extend((rewritten_queries or {}).get(category_key, []))
    return queries


def search_stakeholder_views(
    tech_key: str,
    categories: Iterable[str] | None = None,
    attempt: int = 1,
    rewritten_queries: dict[str, list[str]] | None = None,
    max_results: int = 4,
    search_depth: str = "basic",
    min_score: float = config.MIN_SCORE,
    min_evidence_per_category: int = config.MIN_EVIDENCE_PER_CATEGORY,
) -> dict:
    """선정 기술 1건에 대해 C-3 관점별 근거를 수집한다.

    재실행 시 Master는 미충족 관점을 categories로,
    review_feedback["stakeholder"]["queries"]를 rewritten_queries로 넘긴다.
    """
    if tech_key not in config.TECHNOLOGIES:
        raise ValueError(f"알 수 없는 기술 키: {tech_key} (가능: {list(config.TECHNOLOGIES)})")

    target_categories = list(categories or config.STAKEHOLDER_CATEGORIES)
    evidence: list[dict] = []
    seen: set[str] = set()

    for category_key in target_categories:
        category = config.STAKEHOLDER_CATEGORIES[category_key]
        for query in _build_queries(tech_key, category_key, rewritten_queries):
            hits = web_search(
                query,
                max_results=max_results,
                search_depth=search_depth,
                topic=category["topic"],
                exclude_domains=STAKEHOLDER_EXCLUDED_DOMAINS,
                min_score=min_score,
            )
            for hit in hits:
                url = hit.get("url", "")
                if source_id(url) in seen:
                    continue
                seen.add(source_id(url))
                evidence.append(
                    to_evidence(
                        hit,
                        agent=STAKEHOLDER_AGENT,
                        attempt=attempt,
                        technology=config.TECHNOLOGIES[tech_key]["label"],
                        category=category_key,
                        category_label=category["label"],
                        expected_sources=category["sources"],
                        speaker_hint=speaker_hint(url),
                        query=query,
                    )
                )

    counts = {c: sum(1 for e in evidence if e["category"] == c) for c in target_categories}
    missing = [c for c, n in counts.items() if n < min_evidence_per_category]

    return {
        "technology": config.TECHNOLOGIES[tech_key]["label"],
        "attempt": attempt,
        "evidence": evidence,
        "evidence_by_category": counts,
        "sufficient": not missing,
        "missing": missing,
        "logs": [
            {
                "node": STAKEHOLDER_NODE,
                "attempt": attempt,
                "evidence_count": len(evidence),
                "gate": "sufficient" if not missing else "insufficient",
                "missing": missing,
            }
        ],
    }
