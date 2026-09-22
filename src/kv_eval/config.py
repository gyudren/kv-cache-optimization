"""실행 환경과 설계 고정값.

A-1 도메인, A-4 선정 기술, C-3 이해관계자 평가 기준, D-1 단계·재시도 한도를 한곳에 모은다.
"""

import os

# ── A-1 평가 도메인 ──────────────────────────────────────────────
DOMAIN = "데이터센터·클라우드 장문맥 LLM 서빙"

# ── A-4 선정 기술 ────────────────────────────────────────────────
# 이해관계자는 기술명이 아니라 그 기술을 담은 주체·제품을 두고 발언하므로
# 검색 질의 레벨을 기술(tech)과 주체·제품(entity)으로 나눈다.
TECHNOLOGIES = {
    "MLA": {
        "label": "DeepSeek-V2 MLA",
        "camp": "SW",
        "tech_ko": "DeepSeek-V2 MLA 멀티헤드 잠재 어텐션 KV 캐시",
        "tech_en": "DeepSeek-V2 Multi-head Latent Attention MLA KV cache",
        "entity_ko": "딥시크 DeepSeek 모델 추론 비용",
        "entity_en": "DeepSeek AI model inference efficiency cost",
    },
    "ITME": {
        "label": "ITME (CXL-Hybrid 계층 메모리 확장)",
        "camp": "HW",
        "tech_ko": "CXL 하이브리드 메모리 KV 캐시 계층 확장",
        "tech_en": "CXL hybrid memory tiering LLM KV cache offloading",
        "entity_ko": "SK하이닉스 CXL 메모리 확장 AI 서버",
        "entity_en": "SK hynix CXL memory expansion AI server",
    },
}

# ── C-3 이해관계자 관점: 평가 대상·근거원·판정 기준 ──────────────
STAKEHOLDER_CATEGORIES = {
    "competitor": {
        "label": "경쟁 진영",
        "question": "경쟁사는 이 기술을 어떻게 보는가",
        "sources": ["경쟁사 발표", "기술 비교 기사"],
        "criteria": "- 경쟁사가 병행·협력 가능성을 언급하면 긍정\n"
        "- 한계를 지적하거나 대체 기술을 내세우면 우려",
        "ko_terms": "경쟁사 입장 채택하지 않은 이유 대체 기술 내세워 한계 지적",
        "en_terms": "why competitors did not adopt criticized limitation rival vendor said alternative instead",
        "topic": "general",
        "levels": ["entity"],
    },
    "adopter_developer": {
        "label": "도입 기업·개발자",
        "question": "실제 사용자는 어떻게 평가하는가",
        "sources": ["개발자 커뮤니티", "기업 기술 블로그"],
        "criteria": "- 사용 효과(메모리 절감, 속도 향상) 후기가 있으면 긍정\n"
        "- 품질 저하·도입 복잡도가 지적되면 우려",
        "ko_terms": "도입 후기 운영 경험 실제 적용해보니 문제점 불편",
        "en_terms": "engineers report production experience lessons learned pain points complained serving in practice",
        "topic": "general",
        "levels": ["tech", "entity"],
    },
    "investor": {
        "label": "투자 업계",
        "question": "투자자/전문가는 어떻게 보는가",
        "sources": ["애널리스트 보고서", "경제 언론", "주가 반응 기사"],
        "criteria": "- 성장 기대 및 투자 확대 평가가 있으면 긍정\n"
        "- 과대평가, 수요 감소 우려가 있으면 우려",
        "ko_terms": "애널리스트 전망 투자 확대 과대평가 우려 수요 감소 주가 반응",
        "en_terms": "analysts say investors outlook overhyped concerns demand forecast stock reaction",
        "topic": "news",
        "levels": ["entity"],
    },
}

# 관점당 이 건수 미만이면 미충족으로 보고 missing에 넣는다
MIN_EVIDENCE_PER_CATEGORY = 2

# 관측된 노이즈(프로필·북마크 페이지)는 0.27 이하, 유효 근거는 0.30 이상에 분포
MIN_SCORE = 0.28

# ── D-1 실행 단계와 재시도 한도 ──────────────────────────────────
PHASES = ["init", "tech_research", "parallel_eval", "synthesis", "report", "done"]
PARALLEL_AGENTS = ["market", "stakeholder", "domain"]
RETRY_LIMITS = {
    "tech": 2,
    "market": 2,
    "stakeholder": 2,
    "domain": 2,
    "synthesis": 1,
    "report": 2,
}

# ── 실행 환경 ────────────────────────────────────────────────────
# README의 LLM 버전이 미확정이라 환경변수로 받는다
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PAPERS_DIR = os.path.join(_ROOT, "data", "papers")
OUTPUT_DIR = os.path.join(_ROOT, "outputs")
