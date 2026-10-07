"""충분성 = 필수 결함 없음: 구조적 결함 종류마다 그 관점만 다시 부르고, LLM 자기 보고만으로는 다시 부르지 않는다.

결함은 모두 Fake 도구·LLM이 만든다(운영 코드에 실패 주입 없음).
- tech        : 필수 질문(작동 원리) RAG 검색이 관련 청크를 못 찾음 → 원문 근거 없음
- market      : ITME 인용 출처 1개 → Supervisor 결정적 검사(고유 출처 < 2)
- stakeholder : 웹 검색이 같은 URL 1건만 돌려줌 → 에이전트는 성립을 보고하지만 고유 출처 < 2
- domain      : D1 판정에 인용 없음
"""
from __future__ import annotations

import pytest

from fakes import INF, FakeLLM, FakeRAG, FakeWeb
from kv_eval.agents.stakeholder import STAKEHOLDER_QUERIES
from kv_eval.agents.technology import MECHANISM_QUESTIONS
from kv_eval.config import PERSPECTIVES
from kv_eval.observability import read_decisions

FIRST = "dispatch:tech,market,stakeholder,domain"
STAKEHOLDER_CALLS_PER_RUN = sum(len(qs) for qs in STAKEHOLDER_QUERIES.values())


def _log(state: dict) -> list[dict]:
    return [d for d in read_decisions(state["trace_id"]) if d["node"] == "supervisor"]


# 관점 → (Fake 구성, 재작업 사유에 들어가야 할 필수 결함). Fake는 상태를 가지므로 테스트마다 새로 만든다.
DEFECTS = {
    "tech": (lambda: {"rag": FakeRAG(no_evidence={MECHANISM_QUESTIONS["mla"]: 1})},
             "tech/mla: 필수 질문(작동 원리)의 원문 근거 없음"),
    "market": (lambda: {"llm": FakeLLM(cite_limit={"market": {"itme": 1}}, cite_limit_runs={"market": 1})},
               "market/itme: 고유 출처 1개"),
    "stakeholder": (lambda: {"web": FakeWeb(single_source_calls={"stakeholder": STAKEHOLDER_CALLS_PER_RUN})},
                    "stakeholder/mla: 고유 출처 1개"),
    "domain": (lambda: {"llm": FakeLLM(insufficient={"domain": 1})}, "판정 근거 인용 없음"),
}


@pytest.mark.parametrize("target", list(DEFECTS))
def test_structural_defect_reruns_only_that_perspective(run_graph, target):
    make, reason = DEFECTS[target]
    fakes = make()
    llm = fakes.get("llm") or FakeLLM()
    state = run_graph(llm, web=fakes.get("web"), rag=fakes.get("rag"))
    log = _log(state)
    assert [d["decision"] for d in log[:2]] == [FIRST, f"dispatch:{target}"]
    assert reason in log[1]["reason"]
    assert llm.runs[target] == 2
    assert all(llm.runs[name] == 1 for name in PERSPECTIVES if name != target)
    assert any(reason in m for m in state["feedback"][target]["missing"])
    assert state["perspective_status"][target] == "sufficient" and state["status"] == "completed"


def test_supervisor_rechecks_sources_even_when_agent_reports_no_defect(run_graph):
    """웹이 출처 1개만 돌려주면 에이전트 결과는 성립(필수 결함 없음)해도 Supervisor가 고유 출처를 다시 세어 재조사한다."""
    web = FakeWeb(single_source_calls={"stakeholder": STAKEHOLDER_CALLS_PER_RUN})
    llm = FakeLLM()
    seen: list[dict] = []
    original = llm._stakeholder

    def spy(prompt, run, insufficient):
        result = original(prompt, run, insufficient)
        seen.append({"run": run, "cited": len(result.cited_ids)})
        return result

    llm._stakeholder = spy
    state = run_graph(llm, web=web)
    assert [s["cited"] for s in seen if s["run"] == 1] == [1, 1]   # 1회차: 기술별 인용 1개(결과는 성립)
    assert [d["decision"] for d in _log(state)[:2]] == [FIRST, "dispatch:stakeholder"]
    assert len(state["perspectives"]["stakeholder"]["source_units"]["mla"]) >= 2  # 2회차 출처로 다시 판정


def test_llm_self_report_alone_does_not_rerun(run_graph):
    llm = FakeLLM(optional_only={name: INF for name in PERSPECTIVES})
    state = run_graph(llm)
    assert [d["decision"] for d in _log(state)[:2]] == [FIRST, "synthesis"]
    assert all(llm.runs[name] == 1 for name in PERSPECTIVES)
    assert all(state["perspectives"][name]["missing_optional"] for name in PERSPECTIVES)  # 한계점용으로만 남는다
