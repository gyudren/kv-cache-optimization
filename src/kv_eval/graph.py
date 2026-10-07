"""Supervisor 패턴 그래프.

작업 노드는 모두 supervisor로만 돌아가고, 다음 노드는 supervisor의 conditional edge 하나가 정한다.
작업 노드가 async라서 ainvoke/astream으로 실행한다.
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

AGENT_NODES = ("tech", "market", "stakeholder", "domain", "synthesis", "report")
WORKER_NODES = (*AGENT_NODES, "quality_evaluator")


def build_graph(rag: Any, web: Any, llm: Any, checkpointer: Any = None, policy: Policy | None = None):
    policy = policy or Policy()
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", partial(supervisor_node, policy=policy))
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
    # 라우팅은 여기 한 곳뿐이고, 순서는 policy.decide가 State를 보고 정한다.
    graph.add_conditional_edges("supervisor", route, [*WORKER_NODES, END])
    for name in WORKER_NODES:
        graph.add_edge(name, "supervisor")
    # 호출 config에 recursion_limit을 주면 그 값이 우선한다.
    return graph.compile(checkpointer=checkpointer).with_config(recursion_limit=policy.recursion_limit)
