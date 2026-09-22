"""Internal Pydantic output schemas, not extra Graph agents."""
from pydantic import BaseModel, Field
from typing import Literal

class QueryPlan(BaseModel):
    english_query: str

class Relevance(BaseModel):
    relevant_ids: list[str]

class Rewrite(BaseModel):
    english_query: str

class RAGResponse(BaseModel):
    answer: str
    cited_ids: list[str]
    missing: list[str] = Field(default_factory=list)

class EvidenceAssessment(BaseModel):
    sufficient: bool
    missing: list[str] = Field(default_factory=list)
    reason: str = ""

class PerTechnologyText(BaseModel):
    """기술별 값. OpenAI 구조화 출력(strict)은 자유형 dict를 허용하지 않아 키를 명시한다."""
    mla: str
    itme: str

class TechnologyAssessment(BaseModel):
    summary: str
    trl: PerTechnologyText        # mla and itme independently
    trl_basis: PerTechnologyText  # distinct evidence and limitations per technology
    sufficient: bool
    missing: list[str] = Field(default_factory=list)

class MarketAssessment(BaseModel):
    summary: str
    market_size_growth: str
    adoption: str
    ecosystem: str
    market_size_growth_verdict: Literal["긍정", "우려", "혼재"] | None = None
    adoption_verdict: Literal["긍정", "우려", "혼재"] | None = None
    ecosystem_verdict: Literal["긍정", "우려", "혼재"] | None = None
    verdict: Literal["긍정", "우려", "혼재"] | None = None
    cited_ids: list[str] = Field(default_factory=list)
    sufficient: bool
    missing: list[str] = Field(default_factory=list)

class StakeholderAssessment(BaseModel):
    summary: str
    competitors: str
    developers_adopters: str
    investors: str
    competitors_verdict: Literal["긍정", "우려", "혼재"] | None = None
    developers_adopters_verdict: Literal["긍정", "우려", "혼재"] | None = None
    investors_verdict: Literal["긍정", "우려", "혼재"] | None = None
    verdict: Literal["긍정", "우려", "혼재"] | None = None
    cited_ids: list[str] = Field(default_factory=list)
    sufficient: bool
    missing: list[str] = Field(default_factory=list)

class DomainItem(BaseModel):
    dimension: str
    technology: Literal["mla", "itme"]
    verdict: Literal["적합", "조건부", "제약", "근거 부족"]
    explanation: str
    cited_ids: list[str] = Field(default_factory=list)

class DomainAssessment(BaseModel):
    items: list[DomainItem]
    summary: str
    sufficient: bool
    missing: list[str] = Field(default_factory=list)

class PerTechnologyItems(BaseModel):
    """기술별 목록(상충 사례 등). strict 스키마 제약으로 키를 명시한다."""
    mla: list[str] = Field(default_factory=list)
    itme: list[str] = Field(default_factory=list)

class SynthesisAssessment(BaseModel):
    summary: str
    perspective_matrix: str
    agreements: list[str]
    conflicts: PerTechnologyItems
    evidence_gaps: list[str]
    implications: str
    limitations: str
    needs_source_agents: list[Literal["tech", "market", "stakeholder", "domain"]] = Field(default_factory=list)
    needs_revision: bool = False

class ReportAssessment(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)

class ReportParts(BaseModel):
    summary: str
    background: str
    selection: str
    technology_overview: str
    perspective_trl: str
    perspective_market: str
    perspective_stakeholder: str
    perspective_domain: str
    synthesis: str
    implications: str
    limitations: str
