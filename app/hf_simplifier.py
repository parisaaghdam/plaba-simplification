"""GPU inference wrapper for local HuggingFace models on HPC (no Ollama required).

Loads each model path once and caches the weights globally so multiple agents
that share the same path (e.g. analyzer + quality gate both using Qwen2.5-7B)
do not load the model twice and exhaust GPU VRAM.

Memory budget on A100 40 GB (fp16):
  - Qwen2.5-7B-Instruct  (analyzer + quality gate, shared): ~14 GB
  - Fine-tuned 7B simplifier:                                ~14 GB
  - Total:                                                   ~28 GB  ✓
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

# Global cache: model_path -> (hf_model, tokenizer)
# Allows multiple HFSimplifierChatModel instances with the same path to share
# one set of loaded GPU weights.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


class HFSimplifierChatModel(BaseChatModel):
    """Thin LangChain wrapper around a local merged HF model on GPU."""

    model_path: str = "outputs/plaba-merged"
    temperature: float = 0.3
    max_new_tokens: int = 512

    _hf_model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "hf-simplifier"

    def _load(self) -> None:
        if self._hf_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "HF model routing requires a CUDA GPU. "
                f"PyTorch {torch.__version__} sees cuda=False. "
                "On HPC, reinstall CUDA PyTorch: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

        if self.model_path in _MODEL_CACHE:
            self._hf_model, self._tokenizer = _MODEL_CACHE[self.model_path]
            return

        print(f"Loading HF model from {self.model_path} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        _MODEL_CACHE[self.model_path] = (model, tokenizer)
        self._hf_model, self._tokenizer = model, tokenizer
        print(f"HF model loaded on GPU ({self.model_path}).", flush=True)

    @staticmethod
    def _to_chat(messages: list[BaseMessage]) -> list[dict[str, str]]:
        role_map = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
        }
        chat: list[dict[str, str]] = []
        for message in messages:
            role = role_map.get(message.type, "user")
            chat.append({"role": role, "content": str(message.content)})
        return chat

    def _run_inference(self, messages: list[BaseMessage]) -> str:
        import torch

        self._load()
        chat = self._to_chat(messages)
        inputs = self._tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._hf_model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "top_p": 0.9,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self._hf_model.generate(inputs, **gen_kwargs)

        return self._tokenizer.decode(
            output[0][inputs.shape[1] :],
            skip_special_tokens=True,
        ).strip()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._run_inference(messages)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def get_hf_chat_model(
    temperature: float = 0.3,
    model_path: str | None = None,
) -> HFSimplifierChatModel:
    path = model_path or os.getenv("HF_SIMPLIFIER_PATH", "outputs/plaba-merged")
    max_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
    return HFSimplifierChatModel(
        model_path=path,
        temperature=temperature,
        max_new_tokens=max_tokens,
    )
