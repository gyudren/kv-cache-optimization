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

class TechnologyAssessment(BaseModel):
    summary: str
    trl: dict[str, str]        # mla and itme independently
    trl_basis: dict[str, str]  # distinct evidence and limitations per technology
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

class SynthesisAssessment(BaseModel):
    summary: str
    perspective_matrix: str
    agreements: list[str]
    conflicts: dict[str, list[str]]
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
    perspectives: str
    synthesis: str
    implications: str
    limitations: str
