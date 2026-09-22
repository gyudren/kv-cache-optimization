"""Neutral report using only approved results and verified citation catalog.

본문 서술은 LLM이 쓰고, 판정값이 그대로 드러나야 하는 표(설계 E의 기준별 신호표, D1~D7 판정표,
증거 균형표)와 설계 산출물의 고정 표(A-2 KV cache 규모, A-4 비선정 후보)는 코드가 State에서
직접 만든다. 표와 앞 단계 Agent 판정이 서로 어긋나지 않게 하기 위해서다.
"""
from __future__ import annotations
from collections import Counter
from typing import Any
from ..prompts import prompt_template
from ..schemas import ReportParts
from ..state import deduplicate_evidence
from ..reporting.sections import citation_catalog, citeable_evidence, used_references

TECHS = (("mla", "DeepSeek-V2 MLA"), ("itme", "ITME"))

import re

# 팀 설계 산출물. 1·2장의 고정 표와 설계 판단은 이 문서를 출처 [D]로 표기한다.
DESIGN_REF = ("[D] 판교 9반 2조 (2026). RAG-Design 설계 산출물: KV cache 최적화 기술 다관점 평가 "
              "(A-2 문제 상황, A-4 선정 사유, C 평가 기준). 내부 설계 문서.")

# 설계 산출물 A-2 (Llama-3.1-70B, FP16: 80 layers × 8 KV heads × 128 dim × K·V 2 × 2B = 320 KiB/token)
KV_SCALE_FORMULA = ("산식: 토큰당 KV = 레이어 80 × KV head 8(GQA) × head dim 128 × (K, V) 2 × FP16 2B = 327,680B = 320KiB. "
                    "요청 1건 = 문맥 토큰 수 × 320KiB (8K = 8,192 토큰, 128K = 131,072 토큰, 1M = 1,048,576 토큰). "
                    "HBM 대비 비율은 80GB를 기준으로 한 근사치 [D].")
KV_SCALE_TABLE = """| 문맥 길이 | 요청 1건 KV cache | 동시 8건 | GPU HBM 80GB 대비(1건) |
|---|---|---|---|
| 8K | 2.5 GiB | 20 GiB | 3% |
| 128K | 40 GiB | 320 GiB | 50% |
| 1M | 320 GiB | 2,560 GiB | 400% |"""

# 설계 산출물 A-4 비선정 후보
UNSELECTED_TABLE = """| 후보 | 진영 | 비선정 사유 | RAG 문서 활용 |
|---|---|---|---|
| InfiniGen | HW | 호스트 메모리 오프로딩으로 신규 메모리 인프라 없이 SW 관리 성격이 강함 | O (ITME 비교용 베이스라인) |
| CXL-PNM | HW | 연산까지 메모리 측으로 옮기는 확장형 접근으로, '공간 확장' 자체의 대표성은 ITME가 더 직접적 | O (ITME 비교용 베이스라인) |
| KIVI | SW | 사후 양자화의 대표 베이스라인이나 공개 채택 근거가 연구·라이브러리 수준 | X (선정 검토만) |
| TurboQuant | SW | 재학습 없이 적용 가능한 최신 양자화이나 공식 상용화 근거가 제한적 | X (선정 검토만) |"""

DESIGN_CONTEXT = """[팀 설계 산출물 — 조사 결과가 아니라 과제 설계 내용이므로 인용 없이 '설계 기준'으로 서술 가능]
- 도메인(A-1): 데이터센터·클라우드 기반 장문맥 LLM 서빙. 대규모 동시 요청·비용 민감·문맥 길이 민감 조건이 동시에 존재해 KV cache 문제가 가장 크게 드러난다.
- 문제(A-2): KV cache는 레이어 수·attention head 수·문맥 길이·동시 사용자/batch·저장 정밀도에 비례해 커진다. HBM을 점유하면 동시 요청 수·최대 batch 감소, 장문맥 요청 제한, GPU 추가 도입 비용, KV 이동·로딩 지연, 폐기 후 재계산 비용이 생긴다. (KV cache 규모 표는 코드가 섹션 끝에 삽입한다.)
- 핵심 질문(A-3): 동일한 KV cache 메모리 병목을 해결하는 SW 압축 방식과 HW 메모리 확장 방식은 기술 성숙도, 시장성, 이해관계자, 데이터센터 적용성 관점에서 각각 어떻게 평가되는가? 목적은 우승 기술 선정이 아니라 발전 단계·장단점·관점 간 일치/충돌·도입 전 확인사항을 근거와 함께 제시하는 것.
- 선정 방식(B-1): 2안(Human 기반, 조 토의 후 직접 선정).
- SW 선정(A-4): DeepSeek-V2 MLA — Key·Value를 저차원 잠재 벡터로 압축하도록 어텐션 구조 자체를 재설계. 기존 모델 사후 적용이 어렵다는 한계가 있으나 기술 성숙도·상용화·생태계·데이터센터 적용성을 종합해 선정. (편의성만 보면 KIVI, 찬반 의견 조사에는 TurboQuant가 적합하다는 검토 의견도 있었음)
- HW 선정(A-4): ITME(SK hynix) — CXL 기반 DRAM-NVMe Hybrid Memory로 TB급 원격 메모리 계층을 구성하고, 모델 가중치와 prefix KV cache의 예측 가능한 접근 패턴을 활용해 데이터를 미리 이동. HBM·호스트 DRAM 용량 한계를 넘어 장문맥·다중 턴 누적 KV cache를 저장하고 폐기·재계산을 줄일 것으로 기대. (비선정 후보 표는 코드가 섹션 끝에 삽입한다.)
- RAG 문서(B-3): DeepSeek-V2(52p), ITME(13p), InfiniGen(18p, 베이스라인), CXL-PNM(13p, 베이스라인) 총 96p ≤ 200p. 임베딩 Qwen3-Embedding-0.6B, Dense(FAISS)+BM25 → RRF.
- 평가 기준(C): TRL 1~3 논문/특허, 4~6 코드 공개·도구 탑재 등 구현 근거, 7~9 제품 출시·상용 서비스 적용 발표. 시장성·이해관계자 결과는 긍정/우려/혼재, 도메인은 적합/조건부/제약/근거 부족. SW 추론 최적화 시장과 HW CXL 시장은 다른 시장이므로 시장 규모 숫자로 우열을 정하지 않는다.
- 객관성(C-5): 종합 Agent는 신규 검색 없이 검증된 Evidence만 사용, SW·HW를 동일 형식(쟁점/관점 A/관점 B/이유)으로 병기, HW 베이스라인(InfiniGen, CXL-PNM)으로 ITME 한계를 교차 확인.
"""

VERDICT_FIELDS = {
    "market": [("시장 규모·성장성", "market_size_growth_verdict"), ("상용화·채택 현황", "adoption_verdict"),
               ("생태계 지지", "ecosystem_verdict"), ("종합", "verdict")],
    "stakeholder": [("경쟁 진영", "competitors_verdict"), ("도입 기업·개발자", "developers_adopters_verdict"),
                    ("투자 업계", "investors_verdict"), ("종합", "verdict")],
}


def _cell(text: Any) -> str:
    return " ".join(str(text or "").split()).replace("|", "/")


_OWN_HEADING = re.compile(r"^\s*#{1,6}\s*(REFERENCE|참고\s*문헌|참고자료|References?)\b", re.IGNORECASE)


def sanitize_field(text: str) -> str:
    """LLM이 섹션 안에 자체 REFERENCE 목록이나 '## ' 장 제목을 넣으면 필수 목차가 중복된다.

    REFERENCE는 본문 인용에서 코드가 한 번만 만들므로, 필드 안의 참고문헌 블록은 잘라내고
    '## '(장 제목) 수준 헤딩은 소제목('#### ')으로 낮춘다.
    """
    lines = []
    for line in text.strip().splitlines():
        if _OWN_HEADING.match(line):
            break
        if re.match(r"^\s*#{1,3}\s", line):
            line = "#### " + line.lstrip("# ").strip()
        lines.append(line)
    return "\n".join(lines).strip()


def _paper_citations(ids: list[str], evidence: list[dict]) -> str:
    by_id = {ev["source_id"]: ev for ev in evidence if ev.get("source_type") == "paper"}
    cites = []
    for cid in ids:
        ev = by_id.get(cid)
        if ev and ev.get("citation_number") and ev.get("page"):
            cites.append(f"[{ev['citation_number']}, p.{ev['page']}]")
    return " ".join(dict.fromkeys(cites))


def _web_citations(ids: list[str], evidence: list[dict]) -> str:
    web_number = {ev["url"]: n for n, ev in citation_catalog(evidence)["web"].items()}
    by_id = {ev["source_id"]: ev for ev in evidence if ev.get("source_type") == "web"}
    cites = [f"[W{web_number[by_id[cid]['url']]}]" for cid in ids
             if cid in by_id and by_id[cid].get("url") in web_number]
    return " ".join(dict.fromkeys(cites))


def trl_table(state: dict) -> str:
    tech = state.get("tech_result", {})
    rows = ["| 기술 | TRL 추정(공개 정보 기반) | 판단 근거 요약 |", "|---|---|---|"]
    for key, label in TECHS:
        rows.append(f"| {label} | {_cell((tech.get('trl') or {}).get(key) or '근거 부족')} | "
                    f"{_cell((tech.get('trl_basis') or {}).get(key) or '근거 부족')} |")
    return "\n".join(rows)


def verdict_table(state: dict, perspective: str) -> str:
    result = state.get(f"{perspective}_result", {}).get("technologies", {})
    evidence = deduplicate_evidence(state["evidence"])
    rows = ["| 평가 대상 | DeepSeek-V2 MLA | ITME |", "|---|---|---|"]
    for label, field in VERDICT_FIELDS[perspective]:
        rows.append(f"| {label} | " + " | ".join(
            (result.get(key, {}).get(field) or "근거 부족") for key, _ in TECHS) + " |")
    rows.append("| 근거 | " + " | ".join(
        _web_citations(result.get(key, {}).get("cited_ids", []), evidence) or "근거 부족"
        for key, _ in TECHS) + " |")
    return "\n".join(rows)


def domain_table(state: dict) -> str:
    items = state.get("domain_result", {}).get("items", [])
    evidence = deduplicate_evidence(state["evidence"])
    by_key = {(item["technology"], item["dimension"]): item for item in items}
    dimensions = list(dict.fromkeys(item["dimension"] for item in items)) or []
    dimensions.sort(key=lambda d: d.split()[0])
    rows = ["| 평가항목 | MLA 판정 | MLA 근거 | ITME 판정 | ITME 근거 |", "|---|---|---|---|---|"]
    for dim in dimensions:
        cols = [dim]
        for key, _ in TECHS:
            item = by_key.get((key, dim))
            cols.append(item["verdict"] if item else "근거 부족")
            cols.append(_paper_citations(item.get("cited_ids", []), evidence) if item else "")
        rows.append("| " + " | ".join(c or "-" for c in cols) + " |")
    return "\n".join(rows)


def evidence_balance_table(state: dict) -> str:
    evidence = deduplicate_evidence(state["evidence"])
    agent_label = {"tech": "기술 조사", "market": "시장", "stakeholder": "이해관계자", "domain": "도메인"}
    rows = ["| 구분 | DeepSeek-V2 MLA | ITME | HW 베이스라인(InfiniGen·CXL-PNM) |", "|---|---|---|---|"]
    groups = {"mla": Counter(), "itme": Counter(), "baseline": Counter()}
    for ev in evidence:
        tech = ev.get("technology")
        if tech in groups:
            groups[tech][(ev.get("agent"), ev.get("source_type"))] += 1
    for agent, label in agent_label.items():
        for source_type, kind in (("paper", "논문 청크"), ("web", "웹 문서")):
            counts = [groups[t][(agent, source_type)] for t in ("mla", "itme", "baseline")]
            if any(counts):
                rows.append(f"| {label} · {kind} | " + " | ".join(str(c) for c in counts) + " |")
    totals = [sum(groups[t].values()) for t in ("mla", "itme", "baseline")]
    rows.append("| 합계 | " + " | ".join(str(c) for c in totals) + " |")
    return "\n".join(rows)


def render_report(state: dict, llm: Any) -> str:
    sources = citeable_evidence(state["evidence"])
    tables = {"4.1": trl_table(state), "4.2": verdict_table(state, "market"),
              "4.3": verdict_table(state, "stakeholder"), "4.4": domain_table(state)}
    citation_context = [{"cite": ev["citation"], "technology": ev.get("technology"), "agent": ev.get("agent"),
                         "claim": ev.get("claim", ""), "excerpt": ev.get("excerpt", "")[:900],
                         "url": ev.get("url", "")}
                        for ev in sources]
    feedback = state.get("review_feedback", {}).get("report", {}).get("issues", [])
    revision = ("\nPREVIOUS DRAFT WAS REJECTED BY THE REPORT GATE. Fix every issue below:\n- "
                + "\n- ".join(feedback) + "\n") if feedback else ""
    sections = llm.generate_structured(
        prompt_template("report") + "\n" + DESIGN_CONTEXT + revision
        + "Write a Korean evidence-based multi-perspective evaluation report of DeepSeek-V2 MLA and ITME for "
        "datacenter/cloud long-context LLM serving, returning one Markdown string per field. "
        "Do NOT write the chapter heading of a field yourself (the code adds '## ...' and '### 4.x' headings); "
        "inside a field use '#### ' sub-headings, '- ' bullets and Markdown pipe tables. "
        "Field requirements: "
        "summary = conclusions only (not an overview), 4-6 bullets, at most ~650 Korean characters in total: "
        "TRL of each technology, where perspectives disagree, how the two approaches relate. "
        "background = role of KV cache, memory bottleneck, datacenter/cloud long-context problem, why a perspective comparison instead of a ranking. "
        "selection = selection method (2안: 조 직접 선정), criteria, reasons for MLA and ITME. "
        "technology_overview = MUST contain a Markdown comparison table of both technologies "
        "(rows such as 진영, 작동 계층, 핵심 접근, 저자 보고 효과, 전제조건, 한계), then core approach, author-reported effects and limitations for each. "
        "perspective_trl / perspective_market / perspective_stakeholder / perspective_domain = narrative per technology for 4.1-4.4 "
        "(a verdict signal table is inserted by code above each; do not repeat it, explain the reasoning and who said what). "
        "perspective_trl must separate the maturity of the specific technology from its family (e.g. CXL memory in general) "
        "and state that it is a public-information estimate. "
        "synthesis = perspective matrix table (rows TRL/시장/이해관계자/도메인, columns MLA/ITME), agreements between perspectives, "
        "and a conflict table per technology with columns 쟁점 | 관점 A 평가 | 관점 B 평가 | 엇갈리는 이유 (at least 2 rows each, same format for both), "
        "then the relationship of the two approaches (complementary layers, not rivals). "
        "implications = where evaluations diverge by perspective, and pre-adoption checks per stakeholder "
        "(클라우드 사업자, 모델 개발사, 메모리·HW 벤더, 투자자) as a table. "
        "limitations = MUST state explicitly that every TRL judgement is an estimate based on public information only "
        "(papers, patents, commercial announcements), that there is a lag between publication and adoption, the TRL 4-6 non-disclosure gap, "
        "and MUST list the confirmation-bias countermeasures actually taken (HW baseline papers used to cross-check ITME, "
        "identical reporting format for both technologies, synthesis agent restricted to already-verified evidence); "
        "an evidence balance table is appended by code. "
        "No ranking/endorsement. No invented deployment or quantitative results. "
        "Use the EXACT citation strings from the verified source catalog ([n, p.X] or [Wn]) after every supported fact. "
        "Never cite non-catalog IDs. If evidence is missing, write 근거 부족 and show the gap. Keep design category labels unchanged. "
        "Every number must literally appear in the excerpt of the catalog entry you cite; a figure found only in a web article must be cited "
        "to that [Wn] as a third-party report, never to a paper citation. Do not describe a web source beyond what its excerpt says. "
        "Statements taken from the team design document (selection reasons, non-selected candidates, KV cache scale assumptions) "
        "must end with [D]. Do NOT write any REFERENCE / 참고문헌 list inside any field; the code builds the single REFERENCE chapter. "
        "The following signal tables are inserted verbatim by the code; your narrative MUST use exactly the same verdicts "
        "(if you believe a verdict is wrong, explain the nuance but do not state a different verdict):\n"
        + "\n\n".join(f"[{k}]\n{v}" for k, v in tables.items()) + "\n"
        + repr({k: state.get(k, {}) for k in ("tech_result", "market_result", "stakeholder_result", "domain_result", "synthesis_result")})
        + "\nVerified source catalog:\n" + repr(citation_context), ReportParts,
    )
    parts = {k: sanitize_field(v) for k, v in sections.model_dump().items()}
    report = "\n\n".join([
        f"## SUMMARY\n{parts['summary']}",
        f"## 1. 분석 배경\n{parts['background']}\n\n#### 표 1. KV cache 규모 예시 (Llama-3.1-70B, FP16 — 설계 산출물 A-2) [D]\n{KV_SCALE_TABLE}\n\n{KV_SCALE_FORMULA}",
        f"## 2. 기술 선정\n{parts['selection']}\n\n#### 표 2. 비선정 후보와 사유 (설계 단계 팀 판단 — 설계 산출물 A-4) [D]\n{UNSELECTED_TABLE}",
        f"## 3. 기술 개요\n{parts['technology_overview']}",
        "## 4. 관점별 평가\n"
        f"### 4.1 기술 성숙도(TRL)\n{tables['4.1']}\n\n{parts['perspective_trl']}\n\n"
        f"### 4.2 시장성\n{tables['4.2']}\n\n{parts['perspective_market']}\n\n"
        f"### 4.3 이해관계자\n{tables['4.3']}\n\n{parts['perspective_stakeholder']}\n\n"
        f"### 4.4 도메인 적용성(D1-D7)\n{tables['4.4']}\n\n{parts['perspective_domain']}",
        f"## 5. 종합 의견\n{parts['synthesis']}",
        f"## 6. 시사점\n{parts['implications']}",
        f"## 7. 한계점\n{parts['limitations']}\n\n#### 표 7. 증거 균형표 (중복 제거 후 수집·검증된 Evidence 수)\n{evidence_balance_table(state)}",
    ])
    refs, _ = used_references(report, state["evidence"])
    if "[D]" in report:
        refs.append(DESIGN_REF)
    report += "\n\n## REFERENCE\n" + ("\n".join(f"- {ref}" for ref in refs) if refs else "근거 부족: 실제 사용한 검증 가능 참고자료 없음") + "\n"
    return report


def report_node(state: dict, llm: Any) -> dict:
    report = render_report(state, llm)
    return {"report_draft": report,
            "logs": [{"node": "report", "attempt": state["retry_counts"]["report"], "result": "complete"}]}
