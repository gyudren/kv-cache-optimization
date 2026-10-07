"""외부 리뷰(6c709c0) 지적 사항 검증: 충분성 기준, 원인 귀속, 후속 재조사 한도, 출처 단위·등급, 규칙 보완."""
from __future__ import annotations

from fakes import INF, FakeLLM, FakeWeb
from kv_eval.agents.domain import downgrade_single_document
from kv_eval.agents.report import readable_gaps
from kv_eval.config import PERSPECTIVES, RETRY_LIMITS
from kv_eval.evaluation.quality import check_groundedness, neutrality_issues
from kv_eval.evidence_store import source_units
from kv_eval.observability import read_decisions
from kv_eval.state import followup_key, merge_evidence
from kv_eval.tools.web_search import COMMUNITY_SPEAKER, speaker_hint


def decisions(state: dict) -> list[str]:
    return [d["decision"] for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]


# ---- 충분성: LLM이 적은 세부 미확인 항목은 재조사 사유가 아니다 ---------------------------------
def test_optional_missing_does_not_trigger_rework(run_graph):
    llm = FakeLLM(optional_only={name: INF for name in PERSPECTIVES})
    state = run_graph(llm)
    assert all(llm.runs[name] == 1 for name in PERSPECTIVES)
    assert decisions(state)[:2] == ["dispatch:tech,market,stakeholder,domain", "synthesis"]
    for name in PERSPECTIVES:
        result = state["perspectives"][name]
        assert result["sufficient"] is True and result["llm_sufficient"] is False
        assert result["missing"] == [] and result["missing_optional"]
    assert state["status"] == "completed" and not state["gaps"]


def test_blocking_defect_reworks_only_that_perspective_with_searchable_hints(run_graph):
    llm, web = FakeLLM(insufficient={"stakeholder": 1}), FakeWeb()
    state = run_graph(llm, web=web)
    assert decisions(state)[:2] == ["dispatch:tech,market,stakeholder,domain", "dispatch:stakeholder"]
    feedback = state["feedback"]["stakeholder"]
    assert any("인용 없음" in m for m in feedback["missing"])           # 사유 = 필수 결함
    assert any("투자 업계 발언 근거" in q for q in web.queries)          # 검색어 = 세부 미확인 항목
    assert not any("인용 없음" in q for q in web.queries)


# ---- 원인 귀속: 에이전트 판정에서 온 근거 결함은 그 관점을 재조사 -------------------------------------
def test_groundedness_defect_in_agent_table_reinvestigates_agent(run_graph):
    llm = FakeLLM(judge_fail={"groundedness": (1, "tech")})
    state = run_graph(llm)
    log = decisions(state)
    assert "reinvestigate:tech" in log
    assert "rewrite:report" not in log[:log.index("reinvestigate:tech")]
    assert llm.runs["tech"] == 2 and state["retry_counts"][followup_key("tech")] == 1
    assert state["retry_counts"]["tech"] == 0  # 충분성 재조사 한도는 쓰지 않는다
    assert state["status"] == "completed"


def test_groundedness_defect_in_prose_rewrites_report(run_graph):
    state = run_graph(FakeLLM(judge_fail={"groundedness": (1, "report")}))
    assert "rewrite:report" in decisions(state) and not any(d.startswith("reinvestigate") for d in decisions(state))


# ---- 후속 재조사 한도: 충분성 한도를 다 써도 종합 요청으로 한 번 더 조사 ------------------------------
def test_synthesis_request_runs_even_after_sufficiency_retries_exhausted(run_graph):
    limit = RETRY_LIMITS["stakeholder"]
    llm = FakeLLM(insufficient={"stakeholder": limit + 1}, needs_source=["stakeholder"])
    state = run_graph(llm)
    log = decisions(state)
    assert llm.runs["stakeholder"] == limit + 2
    assert log.index("dispatch:stakeholder", log.index("synthesis")) > log.index("synthesis")
    assert state["retry_counts"]["stakeholder"] == limit and state["retry_counts"][followup_key("stakeholder")] == 1
    assert state["perspective_status"]["stakeholder"] == "sufficient"  # 제외됐던 관점이 후속 조사로 회복


def test_followup_limit_exhausted_is_recorded_as_readable_gap(run_graph):
    llm = FakeLLM(needs_source=["market"], judge_fail={"bias_control": (2, "market")})
    state = run_graph(llm)
    assert state["retry_counts"][followup_key("market")] == 1
    assert any(g.startswith("market:") and "후속 조사 한도" in g for g in state["gaps"])
    section = state["report"].split("#### 근거 공백 (Supervisor 기록)", 1)[1].split("####", 1)[0]
    assert "한도" not in section and "**시장성**" in section


# ---- 출처 단위·등급 -----------------------------------------------------------------------------
def test_paper_sources_count_per_document():
    pages = [{"source_type": "paper", "doc_id": "deepseek_v2", "page": p, "technology": "mla"} for p in (3, 7, 16)]
    web = [{"source_type": "web", "url": f"https://a.example/{i}", "technology": "mla"} for i in range(2)]
    assert len(source_units(pages)["mla"]) == 1
    assert len(source_units(pages + web)["mla"]) == 3


def test_single_document_fit_verdict_is_downgraded():
    evidence = [{"source_id": "a", "doc_id": "itme"}, {"source_id": "b", "doc_id": "itme"},
                {"source_id": "c", "doc_id": "infinigen"}]
    items = [{"verdict": "적합", "explanation": "e", "cited_ids": ["a", "b"]},
             {"verdict": "적합", "explanation": "e", "cited_ids": ["a", "c"]},
             {"verdict": "제약", "explanation": "e", "cited_ids": ["a"]}]
    out = downgrade_single_document(items, evidence)
    assert [i["verdict"] for i in out] == ["조건부", "적합", "제약"]
    assert "원문 1편" in out[0]["explanation"]


def test_community_posts_cannot_carry_competitor_or_investor_verdicts(run_graph):
    assert speaker_hint("https://discussion.fool.com/t/1") == COMMUNITY_SPEAKER
    assert speaker_hint("https://news.ycombinator.com/item?id=1") == COMMUNITY_SPEAKER
    assert speaker_hint("https://someone.github.io/post") == COMMUNITY_SPEAKER
    assert speaker_hint("https://github.com/vllm-project/vllm") != COMMUNITY_SPEAKER

    class CommunityWeb(FakeWeb):
        def search_stakeholder(self, query):
            return [{**r, "speaker": COMMUNITY_SPEAKER} for r in self._results(query)]

    state = run_graph(FakeLLM(), web=CommunityWeb())
    for tech in ("mla", "itme"):
        result = state["perspectives"]["stakeholder"]["technologies"][tech]
        assert result["competitors_verdict"] is None and result["investors_verdict"] is None
        assert result["developers_adopters_verdict"]  # 개발자 반응 판정은 유지
    assert any("개인·커뮤니티 글뿐" in m for m in state["perspectives"]["stakeholder"]["missing_optional"])


# ---- 이전 시도 근거 교체 ---------------------------------------------------------------------------
def test_rerun_replaces_previous_attempt_evidence():
    old = [{"source_id": "w1", "claim": "c", "url": "u1", "agent": "market"},
           {"source_id": "p1", "claim": "c", "page": 1, "agent": "domain"}]
    new = [{"source_id": "w2", "claim": "c", "url": "u2", "agent": "market"}]
    merged = merge_evidence(old, new)
    assert {ev["source_id"] for ev in merged} == {"p1", "w2"}
    assert len(merge_evidence(merged, [{"source_id": "x", "claim": "c", "url": "u"}])) == 3  # agent 없는 입력은 추가


# ---- 규칙 보완 -------------------------------------------------------------------------------
def test_neutrality_rule_catches_comparisons_but_not_normal_sentences():
    flagged = ["MLA가 ITME보다 효율적이다.", "ITME는 MLA를 능가한다.", "데이터센터 사업자는 MLA를 채택해야 한다.",
               "MLA가 ITME보다 우수하지만 비용은 확인되지 않았다."]
    clean = ["본 보고서는 순위, 최종 승자 또는 단일 추천을 제시하지 않는다.", "SK hynix는 HBM 시장 1위로 보도됐다 [W3].",
             "ITME를 쓰려면 CXL 스위치를 도입해야 한다.", "MLA가 ITME보다 낫다고 단정할 수 없다.",
             "MLA는 MHA보다 KV cache를 93.3% 줄인다 [1, p.2]."]
    assert all(neutrality_issues(s) for s in flagged), [s for s in flagged if not neutrality_issues(s)]
    assert not any(neutrality_issues(s) for s in clean), [s for s in clean if neutrality_issues(s)]


def test_design_document_citation_only_allowed_for_design_conditions(run_graph):
    state = run_graph(FakeLLM())
    assert check_groundedness(state)["passed"] is True
    in_design = state["report"].replace("## 2. 기술 선정\n", "## 2. 기술 선정\n비선정 후보는 4개다 [D].\n", 1)
    assert check_groundedness({**state, "report": in_design})["passed"] is True
    as_fact = state["report"].replace("## 6. 시사점\n", "## 6. 시사점\nITME는 TB급 원격 메모리 계층을 구성한다 [D].\n", 1)
    result = check_groundedness({**state, "report": as_fact})
    assert result["passed"] is False and any("[D]" in i for i in result["issues"])


def test_readable_gaps_drop_internal_wording():
    grouped = readable_gaps([
        "tech: 기술 성숙도(TRL) 재조사 2회 후에도 근거 부족 — tech/mla: TRL 판정·근거 미기재; "
        "tech/itme: 필수 질문(정량 성능 결과)의 원문 근거 없음",
        "tech: 종합 단계에서 추가 근거가 필요하다고 판단했으나 후속 조사 한도 소진",
        "tech: 종합 단계에서 추가 근거가 필요하다고 판단했으나 후속 조사 한도 소진",
    ])
    assert grouped == {"tech": ["MLA: TRL 판정·근거 미기재", "ITME: 필수 질문(정량 성능 결과)의 원문 근거 없음",
                                "종합 단계에서 추가 근거가 필요하다고 판단됨"]}
