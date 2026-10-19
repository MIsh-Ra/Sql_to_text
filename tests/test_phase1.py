"""Unit tests for Phase 1: Config, Cache, and CachedChatModel."""

from pathlib import Path
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.cache import DiskLLMCache, compute_cache_key, serialize_prompt
from app.config import settings
from app.llm import CachedChatModel


class FakeUnderlyingChatModel(BaseChatModel):
    """Deterministic in-memory fake model for offline unit testing."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_count += 1
        prompt_text = serialize_prompt(messages)
        res = AIMessage(
            content=f"SELECT 1; -- Response #{self.call_count} for: {prompt_text[:20]}",
            response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        return ChatResult(generations=[ChatGeneration(message=res)])


def test_config_loaded():
    assert settings.llm_provider in ["groq", "gemini", "openai"]
    assert settings.temperature == 0.0
    assert settings.llm_concurrency > 0


def test_cache_serialization_and_key():
    p1 = serialize_prompt("Hello world")
    p2 = serialize_prompt([("user", "Hello world")])
    assert "Hello world" in p1
    assert "Hello world" in p2

    k1 = compute_cache_key("groq", "llama-3.3-70b-versatile", 0.0, "Test prompt")
    k2 = compute_cache_key("groq", "llama-3.3-70b-versatile", 0.0, "Test prompt")
    k3 = compute_cache_key("groq", "llama-3.3-70b-versatile", 0.0, "Different prompt")
    assert k1 == k2
    assert k1 != k3


def test_cached_chat_model_offline(tmp_path: Path):
    test_cache = DiskLLMCache(cache_dir=tmp_path / "cache", enabled=True)
    fake_model = FakeUnderlyingChatModel()

    wrapper = CachedChatModel(
        underlying_model=fake_model,
        provider="fake",
        model_name="fake-v1",
        temperature=0.0,
        custom_cache=test_cache,
    )

    # First call: cache miss, underlying invoked
    res1 = wrapper.invoke("What is 1+1?")
    assert fake_model.call_count == 1
    assert res1.response_metadata.get("cached") is False
    assert "Response #1" in res1.content

    # Second call with identical prompt: served from cache!
    res2 = wrapper.invoke("What is 1+1?")
    assert fake_model.call_count == 1  # Underlying was NOT called again
    assert res2.response_metadata.get("cached") is True
    assert res2.content == res1.content

    # Different prompt: cache miss again
    res3 = wrapper.invoke("What is 2+2?")
    assert fake_model.call_count == 2
    assert res3.response_metadata.get("cached") is False
