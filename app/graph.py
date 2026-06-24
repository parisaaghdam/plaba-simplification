from __future__ import annotations

import os
import time
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.agents import run_analyzer, run_quality_gate, run_simplifier
from app.llm_factory import get_analyzer_model, get_default_model, get_quality_gate_model, get_simplifier_model
from app.metrics import build_metric_snapshot
from app.models import (
    AnalyzerInput,
    GraphState,
    QualityGateInput,
    SimplifierInput,
)


def make_default_model(model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> BaseChatModel:
    return get_default_model(model_name=model_name, temperature=temperature)


def _using_ollama_simplifier() -> bool:
    return os.getenv("USE_OLLAMA_SIMPLIFIER", "").strip() in {"1", "true", "True"}


def _using_hf_simplifier() -> bool:
    return os.getenv("USE_HF_SIMPLIFIER", "").strip() in {"1", "true", "True"}


def _using_local_simplifier() -> bool:
    return _using_ollama_simplifier() or _using_hf_simplifier()


def _eval_verbose() -> bool:
    return os.getenv("EVAL_VERBOSE", "").strip() in {"1", "true", "True"}


def _log_step(message: str) -> None:
    if _eval_verbose():
        print(f"    {message}", flush=True)


def _analyze_node(state: GraphState, model: BaseChatModel) -> dict:
    t0 = time.perf_counter()
    _log_step("analyzer: starting...")
    result = run_analyzer(model, AnalyzerInput(source_text=state.source_text))
    _log_step(f"analyzer: done ({time.perf_counter() - t0:.1f}s)")
    return {"analysis": result}


def _simplifier_node(state: GraphState, model: BaseChatModel) -> dict:
    t0 = time.perf_counter()
    attempt = state.iteration + 1
    _log_step(f"simplifier: starting (attempt {attempt})...")
    revision_notes = state.quality_feedback.revision_notes if state.quality_feedback else None
    result = run_simplifier(
        model,
        SimplifierInput(
            source_text=state.source_text,
            analysis=state.analysis,
            revision_notes=revision_notes,
        ),
        # Fine-tuned local models emit plain text, not structured JSON.
        use_structured_output=not _using_local_simplifier(),
    )
    _log_step(f"simplifier: done ({time.perf_counter() - t0:.1f}s)")
    return {"simplification": result.simplified_text, "iteration": state.iteration + 1}


def _quality_gate_node(state: GraphState, model: BaseChatModel) -> dict:
    t0 = time.perf_counter()
    _log_step("quality_gate: starting...")
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
    verdict = "accepted" if result.accepted else "needs revision"
    _log_step(f"quality_gate: {verdict} ({time.perf_counter() - t0:.1f}s)")
    return {"quality_feedback": result, "accepted": result.accepted}


def _route_after_quality_gate(state: GraphState) -> Literal["simplifier", END]:
    if state.accepted:
        return END
    if state.iteration >= state.max_iterations:
        return END
    return "simplifier"


def build_graph(
    model: BaseChatModel | None = None,
    *,
    skip_analyzer: bool = False,
    single_pass: bool = False,
):
    chat_model = model or get_analyzer_model()
    simplifier_model = get_simplifier_model()
    quality_gate_model = get_quality_gate_model()
    graph = StateGraph(GraphState)

    graph.add_node("analyze", lambda s: _analyze_node(s, chat_model))
    graph.add_node("simplifier", lambda s: _simplifier_node(s, simplifier_model))
    graph.add_node("quality_gate", lambda s: _quality_gate_node(s, quality_gate_model))

    if single_pass:
        graph.add_edge(START, "simplifier")
        graph.add_edge("simplifier", END)
    elif skip_analyzer:
        graph.add_edge(START, "simplifier")
        graph.add_edge("simplifier", "quality_gate")
        graph.add_conditional_edges("quality_gate", _route_after_quality_gate, ["simplifier", END])
    else:
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
    skip_analyzer: bool = False,
    single_pass: bool = False,
) -> GraphState:
    if _eval_verbose():
        mode = "single-pass" if single_pass else ("no-analyzer" if skip_analyzer else "full")
        print(f"  pipeline: building graph ({mode})...", flush=True)
    app = build_graph(skip_analyzer=skip_analyzer, single_pass=single_pass)
    if _eval_verbose():
        print("  pipeline: running agents...", flush=True)
    final_state = app.invoke(
        GraphState(
            source_text=source_text,
            references=references or [],
            max_iterations=max_iterations,
            skip_analyzer=skip_analyzer,
            single_pass=single_pass,
        )
    )
    state = GraphState.model_validate(final_state)
    if single_pass:
        state.accepted = True
    return state
