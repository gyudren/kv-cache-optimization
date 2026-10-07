"""Supervisor 패턴 그래프 (DEV_PLAN §3).

    START → supervisor ─(단일 add_conditional_edges: State 기반 라우팅)→ {tech, market, stakeholder, domain}
                                                                        (Send 동적 fan-out)
                       → synthesis / report / quality_evaluator / END
    모든 작업 노드(4관점·종합·보고서·품질 평가) → supervisor (작업 노드끼리 잇는 엣지 없음)
    품질 평가는 보고서가 정상 완료된 뒤 Supervisor의 evaluate 결정으로만 실행된다.
"""
from __future__ import annotations
from functools import partial
from typing import Any
from langgraph.graph import StateGraph, START, END
from .agents import technology, market, stakeholder, domain, synthesis, report
from .evaluation.quality import quality_evaluator_node
from .state import GraphState
from .supervisor.guard import guarded
from .supervisor.policy import Policy
from .supervisor.router import route, supervisor_node

# 하위 에이전트(작업자). 모두 실행 후 supervisor로만 돌아간다.
AGENT_NODES = ("tech", "market", "stakeholder", "domain", "synthesis", "report")
# Supervisor가 부르는 작업 노드 전체(품질 평가 포함). 각 노드의 유일한 후속 노드는 supervisor다.
WORKER_NODES = (*AGENT_NODES, "quality_evaluator")


def build_graph(rag: Any, web: Any, llm: Any, checkpointer: Any = None, policy: Policy | None = None):
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", partial(supervisor_node, policy=policy or Policy()))
    agents = {
        "tech": partial(technology.technology_node, rag=rag, llm=llm, web=web),
        "market": partial(market.market_node, web=web, llm=llm),
        "stakeholder": partial(stakeholder.stakeholder_node, web=web, llm=llm),
        "domain": partial(domain.domain_node, rag=rag, llm=llm),
        "synthesis": partial(synthesis.synthesis_node, llm=llm),
        "report": partial(report.report_node, llm=llm),
    }
    for name, fn in agents.items():
        graph.add_node(name, guarded(name, fn))
    graph.add_node("quality_evaluator", guarded("quality_evaluator", partial(quality_evaluator_node, llm=llm)))

    graph.add_edge(START, "supervisor")
    # 라우팅은 이 한 곳뿐이다. 실행 순서는 엣지가 아니라 State(policy.decide)가 정한다.
    graph.add_conditional_edges("supervisor", route, [*WORKER_NODES, END])
    for name in WORKER_NODES:
        graph.add_edge(name, "supervisor")
    return graph.compile(checkpointer=checkpointer)
