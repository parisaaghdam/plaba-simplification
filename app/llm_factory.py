"""Model factories for the agents.

Lets the simplifier optionally use a locally fine-tuned model served by Ollama,
while the analyzer and quality gate keep using a reliable API model by default.

Environment variables:
    USE_HF_SIMPLIFIER=1                -> route the simplifier to a local HF model on GPU (HPC)
    HF_SIMPLIFIER_PATH=outputs/...     -> merged model directory (default outputs/plaba-merged)
    USE_OLLAMA_SIMPLIFIER=1            -> route the simplifier to Ollama (local PC)
    OLLAMA_SIMPLIFIER_MODEL=plaba-...  -> Ollama model name (default plaba-simplifier)
    OLLAMA_BASE_URL=http://...         -> optional custom Ollama host
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_default_model(model_name: str = "gpt-4o-mini", temperature: float = 0.1) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=temperature)


def get_simplifier_model(temperature: float = 0.3) -> BaseChatModel:
    """Fine-tuned local model (HF GPU or Ollama) when enabled, else the API model."""
    if os.getenv("USE_HF_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
        from app.hf_simplifier import get_hf_chat_model

        return get_hf_chat_model(temperature=temperature)

    if os.getenv("USE_OLLAMA_SIMPLIFIER", "").strip() in {"1", "true", "True"}:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "USE_OLLAMA_SIMPLIFIER is set but langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            ) from exc

        model_name = os.getenv("OLLAMA_SIMPLIFIER_MODEL", "plaba-simplifier")
        base_url = os.getenv("OLLAMA_BASE_URL")
        # Avoid hanging forever if Ollama stops responding (seconds).
        timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "900"))
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "timeout": timeout_s,
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    return get_default_model(temperature=temperature)
