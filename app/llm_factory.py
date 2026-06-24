"""Model factories for the agents.

All three agents (analyzer, simplifier, quality gate) can be routed to
open-source models served locally via Ollama or HuggingFace, enabling
fully reproducible runs without any paid API.

Environment variables:
    # Analyzer
    USE_OLLAMA_ANALYZER=1              -> route analyzer to Ollama
    OLLAMA_ANALYZER_MODEL=qwen2.5:7b   -> Ollama model name (default qwen2.5:7b)
    USE_HF_ANALYZER=1                  -> route analyzer to a local HF model on GPU
    HF_ANALYZER_PATH=...               -> HF model dir or hub ID (default Qwen/Qwen2.5-7B-Instruct)

    # Simplifier
    USE_HF_SIMPLIFIER=1                -> route simplifier to a local HF model on GPU (HPC)
    HF_SIMPLIFIER_PATH=outputs/...     -> merged model directory (default outputs/plaba-merged)
    USE_OLLAMA_SIMPLIFIER=1            -> route simplifier to Ollama (local PC)
    OLLAMA_SIMPLIFIER_MODEL=plaba-...  -> Ollama model name (default plaba-simplifier)

    # Quality gate
    USE_OLLAMA_QUALITY_GATE=1          -> route quality gate to Ollama
    OLLAMA_QUALITY_GATE_MODEL=...      -> Ollama model name (default qwen2.5:7b)
    USE_HF_QUALITY_GATE=1              -> route quality gate to a local HF model on GPU
    HF_QUALITY_GATE_PATH=...           -> HF model dir or hub ID (default Qwen/Qwen2.5-7B-Instruct)
    QUALITY_GATE_MODEL=gpt-4o          -> OpenAI model name when using the API (default gpt-4o)

    # Shared
    OLLAMA_BASE_URL=http://...         -> optional custom Ollama host
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def _is_set(env_key: str) -> bool:
    return os.getenv(env_key, "").strip() in {"1", "true", "True"}


def _make_ollama(model_name: str, temperature: float) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "Ollama routing requires langchain-ollama. Run: pip install langchain-ollama"
        ) from exc
    base_url = os.getenv("OLLAMA_BASE_URL")
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "900"))
    kwargs: dict = {"model": model_name, "temperature": temperature, "timeout": timeout_s}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOllama(**kwargs)


def _make_hf(model_path: str, temperature: float) -> BaseChatModel:
    from app.hf_simplifier import get_hf_chat_model

    return get_hf_chat_model(model_path=model_path, temperature=temperature)


def get_default_model(model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=temperature)


def get_analyzer_model(temperature: float = 0.1) -> BaseChatModel:
    """Analyzer agent model — open-source via Ollama/HF or OpenAI fallback."""
    if _is_set("USE_HF_ANALYZER"):
        path = os.getenv("HF_ANALYZER_PATH", "Qwen/Qwen2.5-7B-Instruct")
        return _make_hf(path, temperature)
    if _is_set("USE_OLLAMA_ANALYZER"):
        model_name = os.getenv("OLLAMA_ANALYZER_MODEL", "qwen2.5:7b")
        return _make_ollama(model_name, temperature)
    return get_default_model(temperature=temperature)


def get_quality_gate_model(temperature: float = 0.0) -> BaseChatModel:
    """Quality gate model — open-source via Ollama/HF or OpenAI fallback."""
    if _is_set("USE_HF_QUALITY_GATE"):
        path = os.getenv("HF_QUALITY_GATE_PATH", "Qwen/Qwen2.5-7B-Instruct")
        return _make_hf(path, temperature)
    if _is_set("USE_OLLAMA_QUALITY_GATE"):
        model_name = os.getenv("OLLAMA_QUALITY_GATE_MODEL", "qwen2.5:7b")
        return _make_ollama(model_name, temperature)
    from langchain_openai import ChatOpenAI

    model_name = os.getenv("QUALITY_GATE_MODEL", "gpt-4o")
    return ChatOpenAI(model=model_name, temperature=temperature)


def get_simplifier_model(temperature: float = 0.3) -> BaseChatModel:
    """Fine-tuned local model (HF GPU or Ollama) when enabled, else the API model."""
    if _is_set("USE_HF_SIMPLIFIER"):
        path = os.getenv("HF_SIMPLIFIER_PATH", "outputs/plaba-merged")
        return _make_hf(path, temperature)
    if _is_set("USE_OLLAMA_SIMPLIFIER"):
        model_name = os.getenv("OLLAMA_SIMPLIFIER_MODEL", "plaba-simplifier")
        return _make_ollama(model_name, temperature)
    return get_default_model(temperature=temperature)
