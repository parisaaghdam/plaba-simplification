"""Model factories for the agents.

Lets the simplifier optionally use a locally fine-tuned model served by Ollama,
while the analyzer and quality gate keep using a reliable API model by default.

Environment variables:
    USE_OLLAMA_SIMPLIFIER=1            -> route the simplifier to Ollama
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
    """Fine-tuned Ollama model when enabled, otherwise the default API model."""
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
        kwargs = {"model": model_name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    return get_default_model(temperature=temperature)
