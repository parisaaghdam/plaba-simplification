from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.agents import run_analyzer, run_quality_gate, run_simplifier
from app.metrics import build_metric_snapshot
from app.models import (
    AnalyzerInput,
    GraphState,
    QualityGateInput,
    SimplifierInput,
)


def make_default_model(model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)


def _analyze_node(state: GraphState, model: ChatOpenAI) -> dict:
    result = run_analyzer(model, AnalyzerInput(source_text=state.source_text))
    return {"analysis": result}


def _simplifier_node(state: GraphState, model: ChatOpenAI) -> dict:
    revision_notes = state.quality_feedback.revision_notes if state.quality_feedback else None
    result = run_simplifier(
        model,
        SimplifierInput(
            source_text=state.source_text,
            analysis=state.analysis,
            revision_notes=revision_notes,
        ),
    )
    return {"simplification": result.simplified_text, "iteration": state.iteration + 1}


def _quality_gate_node(state: GraphState, model: ChatOpenAI) -> dict:
    snapshot = build_metric_snapshot(
        state.source_text,
        state.simplification,
        state.references or None,
    )
    result = run_quality_gate(
        model,
        QualityGateInput(
            source_text=state.source_text,
            simplified_text=state.simplification,
            analysis=state.analysis,
            references=state.references,
            metric_snapshot=snapshot,
        ),
    )
    return {"quality_feedback": result, "accepted": result.accepted}


def _route_after_quality_gate(state: GraphState) -> Literal["simplifier", END]:
    if state.accepted:
        return END
    if state.iteration >= state.max_iterations:
        return END
    return "simplifier"


def build_graph(model: ChatOpenAI | None = None):
    chat_model = model or make_default_model()
    graph = StateGraph(GraphState)

    graph.add_node("analyze", lambda s: _analyze_node(s, chat_model))
    graph.add_node("simplifier", lambda s: _simplifier_node(s, chat_model))
    graph.add_node("quality_gate", lambda s: _quality_gate_node(s, chat_model))

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "simplifier")
    graph.add_edge("simplifier", "quality_gate")
    graph.add_conditional_edges("quality_gate", _route_after_quality_gate, ["simplifier", END])

    return graph.compile()


def simplify_with_refinement(
    source_text: str,
    *,
    references: list[str] | None = None,
    max_iterations: int = 4,
) -> GraphState:
    app = build_graph()
    final_state = app.invoke(
        GraphState(
            source_text=source_text,
            references=references or [],
            max_iterations=max_iterations,
        )
    )
    return GraphState.model_validate(final_state)
