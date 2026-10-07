"""API 키 없이 그래프 흐름을 검증하기 위한 Fake LLM·Web·RAG.

FakeLLM은 스키마 종류로 어느 에이전트의 호출인지 구분하고, 시나리오 옵션에 따라
근거 부족·한쪽 판정·예외·금지 표현·Judge 미달을 정해진 실행 회차에만 만들어 낸다.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from hashlib import sha256

from kv_eval.schemas import (CriterionVerdict, DomainAssessment, DomainItem, EvalVerdict, MarketAssessment,
                             PerTechnologyItems, PerTechnologyText, ReportParts, StakeholderAssessment,
                             SynthesisAssessment, TechnologyAssessment)

DOCS = {"mla": ("deepseek_v2_mla", 1, "mla"), "itme": ("itme", 2, "itme"), "itme_baseline": ("infinigen", 3, "baseline")}
DIMENSIONS = ["D1 워크로드 수용 능력", "D2 서비스 성능", "D3 메모리 자원 효율", "D4 품질·정보 보존",
              "D5 운영 안정성", "D6 도입·확장 용이성", "D7 비용 효율성"]
SCHEMA_AGENT = {TechnologyAssessment: "tech", MarketAssessment: "market", StakeholderAssessment: "stakeholder",
                DomainAssessment: "domain", SynthesisAssessment: "synthesis", ReportParts: "report",
                EvalVerdict: "judge"}


class SimulatedCrash(BaseException):
    """프로세스 강제 종료를 흉내 낸다(에이전트 래퍼가 잡지 않는 BaseException)."""


class FakeRAG:
    """no_evidence: {질문 일부 문자열: N}  그 문자열을 포함한 질문은 처음 N회 관련 청크를 찾지 못한다(근거 없음)."""

    def __init__(self, no_evidence=None):
        self.calls = 0
        self.log: list[dict] = []  # 호출별 질문·재작업 지시·워커 스레드에서 본 실행 설정
        self.no_evidence = dict(no_evidence or {})
        self._misses: Counter = Counter()

    def rag_answer(self, question: str, technology_filter: str, feedback: dict | None = None) -> dict:
        from langchain_core.runnables.config import var_child_runnable_config
        self.calls += 1
        self.log.append({"question": question, "feedback": dict(feedback or {}),
                         "config": var_child_runnable_config.get()})
        for key, limit in self.no_evidence.items():
            if key in question and self._misses[key] < limit:
                self._misses[key] += 1
                return {"answer": "근거 부족", "evidence": [], "sufficient": False,
                        "missing": [f"{question}: 관련 청크 0개(<2)"], "search_attempts": 3}
        doc_id, number, tech = DOCS[technology_filter]
        page = int(sha256(question.encode()).hexdigest(), 16) % 40 + 1
        chunk_id = f"{doc_id}#p{page}#{sha256(question.encode()).hexdigest()[:6]}"
        evidence = [{"source_id": chunk_id, "claim": question, "excerpt": (f"{doc_id} page {page} 근거 발췌 문장. " * 60)[:1200],
                     "doc_id": doc_id, "page": page, "technology": tech, "citation_number": number,
                     "source_type": "paper"}]
        return {"answer": f"{question} 에 대한 근거 기반 답변 [{number}, p.{page}]", "evidence": evidence,
                "sufficient": True, "missing": [], "search_attempts": 1}


class FakeWeb:
    """single_source_calls: {"market"|"stakeholder": K}  해당 검색의 처음 K회 호출은 같은 URL 1건만 돌려준다
    (검색이 출처 1개만 찾은 상황 → 고유 출처 2개 미만)."""
    PUBLISHERS = [f"news{i}.example.com" for i in range(5)]

    def __init__(self, single_source_calls=None):
        self.queries: list[str] = []
        self._n = 0
        self.single_source_calls = dict(single_source_calls or {})
        self._calls: Counter = Counter()

    def _single(self, kind: str, query: str) -> list[dict] | None:
        self._calls[kind] += 1
        if self._calls[kind] > self.single_source_calls.get(kind, 0):
            return None
        self.queries.append(query)
        return [{"url": f"https://single.example.com/{kind}", "title": "단일 기사",
                 "excerpt": (f"{query} 관련 단일 보도. " * 30)[:1000], "publisher": "single.example.com",
                 "published_at": "2026-09-01", "speaker": "언론", "score": 0.9}]

    def _results(self, query: str) -> list[dict]:
        self.queries.append(query)
        out = []
        for _ in range(2):
            self._n += 1
            publisher = self.PUBLISHERS[self._n % len(self.PUBLISHERS)]
            out.append({"url": f"https://{publisher}/a/{self._n}", "title": f"기사 {self._n}",
                        "excerpt": (f"{query} 관련 공개 발표 내용 {self._n}. " * 30)[:1000], "publisher": publisher,
                        "published_at": "2026-09-01", "speaker": "언론", "score": 0.9})
        return out

    def search_market(self, query: str, topic: str = "news") -> list[dict]:
        return self._single("market", query) or self._results(query)

    def search_stakeholder(self, query: str) -> list[dict]:
        return self._single("stakeholder", query) or self._results(query)


def _runs_left(table: dict, agent: str, run: int) -> bool:
    return run <= table.get(agent, 0)


class FakeLLM:
    """시나리오 옵션(모두 '처음 N회 실행'에 적용, math.inf = 항상):
    - insufficient: {agent: N}  판정을 막는 필수 결함(TRL 미기재·인용 없음·인용 없는 판정)과 세부 미확인 항목(missing) 반환.
                                LLM이 missing만 적고 결과가 성립하면 필수 결함이 아니므로 재조사하지 않는다(optional_only).
    - optional_only: {agent: N} 결과는 성립하고 LLM이 세부 미확인 항목만 적는다(재조사 대상 아님)
    - one_sided:    {agent: N}  시장·이해관계자 판정을 모두 '긍정'으로(편향 규칙 미달 유도)
    - raise_on:     {agent: N}  RuntimeError (에이전트 래퍼가 failed로 기록)
    - crash_on:     {agent: N}  SimulatedCrash (프로세스 중단 흉내, 재개 테스트용)
    - banned_report: N          보고서에 우열 표현 삽입(중립성 규칙 미달 유도)
    - judge_fail:   {criterion: (N, target_agent)}  LLM Judge 미달
    - judge_reason_varies: True  Judge 사유 문장이 실행마다 달라진다(실제 LLM처럼 같은 지적을 다른 문장으로)
    - needs_source: [agent]     종합이 추가 근거를 요청할 관점(처음 needs_source_runs회 종합 실행에서)
    - cite_limit:   {agent: {tech: n}}  인용 출처 수 제한(cite_limit_runs={agent: N}이면 처음 N회만, 없으면 매 시도)
    """

    def __init__(self, insufficient=None, one_sided=None, raise_on=None, crash_on=None,
                 banned_report=0, judge_fail=None, needs_source=None, cite_limit=None, optional_only=None,
                 needs_source_runs=1, cite_limit_runs=None, judge_reason_varies=False):
        self.insufficient = insufficient or {}
        self.optional_only = optional_only or {}
        self.one_sided = one_sided or {}
        self.raise_on = raise_on or {}
        self.crash_on = crash_on or {}
        self.banned_report = banned_report
        self.judge_fail = judge_fail or {}
        self.judge_reason_varies = judge_reason_varies
        self.needs_source = needs_source or []
        self.needs_source_runs = needs_source_runs
        self.cite_limit = cite_limit or {}  # {agent: {tech: n}} 인용 출처 수 제한
        self.cite_limit_runs = cite_limit_runs or {}
        self.runs: Counter = Counter()
        self.prompts: dict[str, list[str]] = {}

    # 시장·이해관계자는 기술별로 2번 호출되므로 mla 호출에서만 실행 회차를 센다.
    def _run(self, agent: str, prompt: str) -> int:
        if agent in ("market", "stakeholder"):
            if "web:mla:" in prompt or "web:stakeholder:mla:" in prompt:
                self.runs[agent] += 1
        else:
            self.runs[agent] += 1
        return self.runs[agent]

    def generate_structured(self, prompt: str, schema):
        agent = SCHEMA_AGENT[schema]
        self.prompts.setdefault(agent, []).append(prompt)
        run = self._run(agent, prompt)
        if _runs_left(self.crash_on, agent, run):
            raise SimulatedCrash(f"{agent} crashed")
        if _runs_left(self.raise_on, agent, run):
            raise RuntimeError(f"{agent} upstream API error")
        insufficient = _runs_left(self.insufficient, agent, run)
        self._optional = insufficient or _runs_left(self.optional_only, agent, run)
        return getattr(self, f"_{agent}")(prompt, run, insufficient)

    def _tech(self, prompt, run, insufficient):
        return TechnologyAssessment(
            summary="MLA와 ITME의 기술 범위와 한계를 원문 근거로 정리했다.",
            trl=PerTechnologyText(mla="" if insufficient else "개별 기술 TRL 4-6", itme="개별 기술 TRL 4-5"),
            trl_basis=PerTechnologyText(mla="공개 구현과 서빙 백엔드 근거", itme="FPGA 프로토타입 근거"),
            sufficient=not self._optional, missing=["MLA 상용 채택 직접 근거"] if self._optional else [])

    @staticmethod
    def _ids(prompt: str, prefix: str) -> list[str]:
        return list(dict.fromkeys(re.findall(rf"^({re.escape(prefix)}[0-9a-f]{{14}}):", prompt, re.M)))

    def _cite_cap(self, agent: str, tech: str, run: int) -> int | None:
        if agent in self.cite_limit_runs and run > self.cite_limit_runs[agent]:
            return None
        return self.cite_limit.get(agent, {}).get(tech)

    def _market(self, prompt, run, insufficient):
        tech = "mla" if "web:mla:" in prompt else "itme"
        verdicts = (("긍정",) * 4 if _runs_left(self.one_sided, "market", run) else ("긍정", "우려", "혼재", "혼재"))
        return MarketAssessment(
            summary=f"{tech} 시장 근거 요약", market_size_growth="시장 성장 근거", adoption="채택 근거",
            ecosystem="생태계 근거", market_size_growth_verdict=verdicts[0], adoption_verdict=verdicts[1],
            ecosystem_verdict=verdicts[2], verdict=verdicts[3],
            cited_ids=[] if insufficient else self._ids(prompt, f"web:{tech}:")[:self._cite_cap("market", tech, run)],
            sufficient=not self._optional, missing=[f"{tech} 시장 규모 정량 근거"] if self._optional else [])

    def _stakeholder(self, prompt, run, insufficient):
        tech = "mla" if "web:stakeholder:mla:" in prompt else "itme"
        verdicts = (("긍정",) * 4 if _runs_left(self.one_sided, "stakeholder", run) else ("우려", "긍정", "혼재", "혼재"))
        return StakeholderAssessment(
            summary=f"{tech} 이해관계자 반응 요약", competitors="경쟁 진영 반응", developers_adopters="개발자 반응",
            investors="투자 업계 반응", competitors_verdict=verdicts[0], developers_adopters_verdict=verdicts[1],
            investors_verdict=verdicts[2], verdict=verdicts[3],
            cited_ids=[] if insufficient else self._ids(prompt, f"web:stakeholder:{tech}:"),
            sufficient=not self._optional, missing=[f"{tech} 투자 업계 발언 근거"] if self._optional else [])

    def _domain(self, prompt, run, insufficient):
        ids = {"mla": re.findall(r"^(deepseek_v2_mla#p\d+#\w+):", prompt, re.M),
               "itme": re.findall(r"^(itme#p\d+#\w+):", prompt, re.M)}
        items = [DomainItem(dimension=dim, technology=tech, verdict="적합" if i % 2 == 0 else "조건부",
                            explanation=f"{tech} {dim} 근거 기반 판정",
                            cited_ids=[] if insufficient and i == 0 else ids[tech][:1])
                 for tech in ("mla", "itme") for i, dim in enumerate(DIMENSIONS)]
        return DomainAssessment(items=items, summary="D1~D7 판정", sufficient=not self._optional,
                                missing=["D5 운영 안정성 장기 측정 근거"] if self._optional else [])

    def _synthesis(self, prompt, run, insufficient):
        needs = self.needs_source if run <= self.needs_source_runs else []
        return SynthesisAssessment(
            summary="관점 간 일치와 상충을 정리했다.", perspective_matrix="| 관점 | MLA | ITME |",
            agreements=["두 기술 모두 KV cache 병목을 다룬다"],
            conflicts=PerTechnologyItems(mla=["생태계 vs TRL", "효율 vs 사후 적용"],
                                         itme=["처리량 vs 안정성", "확장성 vs 지연"]),
            evidence_gaps=[], implications="도입 전 확인사항", limitations="공개 정보 기반 추정",
            needs_source_agents=needs, needs_revision=False)

    def _report(self, prompt, run, insufficient):
        cites = re.findall(r"'cite': '(\[[^']+\])'", prompt)
        paper = [c for c in cites if not c.startswith("[W")][:2] or ["근거 부족"]
        web = [c for c in cites if c.startswith("[W")][:2] or ["근거 부족"]
        cite = " ".join(paper + web)
        banned = "\n\nMLA가 ITME보다 더 우수하다." if run <= self.banned_report else ""
        para = lambda topic: f"MLA와 ITME의 {topic}을 근거와 함께 서술한다 {cite}. 두 기술은 관점에 따라 평가가 달라진다."
        # 판정 한계 명시 지시가 있으면 그 절 서술 첫 문장을 '한계:'로 쓴다(실제 LLM이 지시를 따른 경우를 흉내)
        limited = set(re.findall(r"^- 4\.[1-4] \((perspective_\w+)\):", prompt.split("VERDICT LIMITATIONS", 1)[-1], re.M)
                      if "VERDICT LIMITATIONS" in prompt else [])
        lead = lambda field: "한계: 품질 평가가 지적한 근거 결함 때문에 위 표의 판정은 수집된 근거 범위 안에서만 유효하다.\n" \
            if field in limited else ""
        return ReportParts(
            summary=f"- MLA와 ITME의 TRL은 공개 정보 기반 추정이다 {paper[0]}.\n- 관점별 평가가 엇갈린다 {web[0]}.",
            background=para("배경"), selection=para("선정 사유"),
            technology_overview="| 항목 | MLA | ITME |\n|---|---|---|\n| 계층 | 모델 | 메모리 |\n\n" + para("개요"),
            perspective_trl=lead("perspective_trl") + para("기술 성숙도"),
            perspective_market=lead("perspective_market") + para("시장성") + banned,
            perspective_stakeholder=lead("perspective_stakeholder") + para("이해관계자 반응"),
            perspective_domain=lead("perspective_domain") + para("도메인 적용성"),
            synthesis=para("종합"), implications=para("시사점"), limitations=para("한계"))

    def _judge(self, prompt, run, insufficient):
        verdicts = {}
        for name in ("groundedness", "neutrality", "bias_control", "coverage"):
            runs, target = self.judge_fail.get(name, (0, None))
            failed = run <= runs
            reason = f"{name} 판정" + (f" (실행 {run}회차 표현)" if self.judge_reason_varies else "")
            verdicts[name] = CriterionVerdict(passed=not failed, score=2 if failed else 5,
                                              reason=reason, target_agent=target if failed else None)
        return EvalVerdict(**verdicts)


INF = math.inf
