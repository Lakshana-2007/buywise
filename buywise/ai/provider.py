"""
LLM provider abstraction.

Supports OpenAI, Groq, and a deterministic Mock provider for tests.
Never hard-codes API keys — reads from environment variables only.

Usage:
    provider = get_llm_provider()
    llm = provider.get_chat_model()
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from buywise.core.config import (
    GROQ_API_KEY,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    MOCK_LLM,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    def get_chat_model(self, **kwargs) -> Any:
        """Return a LangChain chat model instance."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier."""


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        self._model = model
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def get_chat_model(self, **kwargs) -> Any:
        from langchain_openai import ChatOpenAI
        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        return ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=api_key,
            **kwargs,
        )


class GroqProvider(LLMProvider):
    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = LLM_TEMPERATURE):
        self._model = model
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    def get_chat_model(self, **kwargs) -> Any:
        from langchain_groq import ChatGroq
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        return ChatGroq(
            model=self._model,
            temperature=self._temperature,
            api_key=api_key,
            **kwargs,
        )


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        self._model = model
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model

    def get_chat_model(self, **kwargs) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Always re-read from env at call time so tests can monkeypatch reliably.
        api_key = os.getenv("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=self._model,
            temperature=self._temperature,
            google_api_key=api_key,
            **kwargs,
        )


class MockProvider(LLMProvider):
    """
    Deterministic mock provider for tests and offline runs.
    Returns empty/safe evidence — the deterministic engine handles the rest.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-deterministic"

    def get_chat_model(self, **kwargs) -> Any:
        from unittest.mock import MagicMock
        mock = MagicMock()
        # Default: return empty structured output
        mock.invoke.return_value = MagicMock(content="{}")
        return mock


# Token usage tracker (shared across the pipeline run)
class UsageTracker:
    """Tracks LLM token usage across the full pipeline run."""

    def __init__(self):
        self.calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.provider: str = ""
        self.model: str = ""

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost_usd(self) -> float:
        """Rough cost estimate — assumes GPT-4o-mini pricing as default."""
        # $0.15 per 1M input, $0.60 per 1M output (GPT-4o-mini)
        return (self.input_tokens * 0.00000015) + (self.output_tokens * 0.0000006)

    def to_report_dict(self, n_requests: int = 1) -> dict:
        return {
            "provider": str(self.provider) if self.provider else "mock/deterministic",
            "model": str(self.model) if self.model else "N/A",
            "total_calls": int(self.calls),
            "input_tokens": int(self.input_tokens),
            "output_tokens": int(self.output_tokens),
            "total_tokens": int(self.total_tokens),
            "estimated_cost_usd": round(self.estimated_cost_usd(), 4),
            "avg_tokens_per_request": round(self.total_tokens / max(n_requests, 1), 1),
            "avg_cost_per_request_usd": round(self.estimated_cost_usd() / max(n_requests, 1), 6),
        }


# Global usage tracker instance
usage_tracker = UsageTracker()


def get_llm_provider() -> LLMProvider:
    """
    Return the configured LLM provider.
    MOCK_LLM must be explicitly enabled for offline mode. Otherwise both a
    supported provider and its API key must be configured explicitly.

    All env-var reads happen at call time so that test monkeypatching works
    correctly — never use module-level constants for provider/key selection.
    """
    mock_enabled = os.getenv("MOCK_LLM", "false").lower() == "true"
    if mock_enabled:
        logger.info("MOCK_LLM=true - using deterministic mock provider")
        return MockProvider()

    # Read provider fresh at call time (not from module-level constant)
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:
        raise RuntimeError(
            "LLM_PROVIDER is required when MOCK_LLM is not true; "
            "set LLM_PROVIDER=openai, groq, or gemini"
        )
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "") or OPENAI_API_KEY
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        model = os.getenv("LLM_MODEL", LLM_MODEL)
        logger.info("Using OpenAI provider, model=%s", model)
        p = OpenAIProvider(model=model)
        usage_tracker.provider = "openai"
        usage_tracker.model = model
        return p
    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY
        if not key:
            raise RuntimeError("GROQ_API_KEY is required for LLM_PROVIDER=groq")
        model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        logger.info("Using Groq provider, model=%s", model)
        p = GroqProvider(model=model)
        usage_tracker.provider = "groq"
        usage_tracker.model = model
        return p
    elif provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for LLM_PROVIDER=gemini")
        model = os.getenv("LLM_MODEL", LLM_MODEL)
        logger.info("Using Gemini provider, model=%s", model)
        p = GeminiProvider(model=model)
        usage_tracker.provider = "gemini"
        usage_tracker.model = model
        return p
    else:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={provider!r}; use openai, groq, or gemini"
        )
