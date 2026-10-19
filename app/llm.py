"""Unified LLM factory supporting Groq, Google Gemini, and OpenAI with caching and retry backoff."""

import asyncio
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.cache import DiskLLMCache, llm_cache, serialize_prompt
from app.config import settings

logger = logging.getLogger(__name__)


# Global concurrency semaphores
_THREAD_SEMAPHORE: Optional[threading.Semaphore] = None
_ASYNC_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _get_thread_semaphore() -> threading.Semaphore:
    global _THREAD_SEMAPHORE
    if _THREAD_SEMAPHORE is None:
        _THREAD_SEMAPHORE = threading.Semaphore(settings.llm_concurrency)
    return _THREAD_SEMAPHORE


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        indicator in msg
        for indicator in [
            "rate limit",
            "ratelimit",
            "429",
            "resourceexhausted",
            "too many requests",
            "quota",
            "tokens per minute",
            "requests per minute",
        ]
    )


class CachedChatModel(BaseChatModel):
    """Wrapper around any BaseChatModel providing deterministic disk caching and backoff."""

    underlying_model: BaseChatModel
    provider: str
    model_name: str
    temperature: float = 0.0
    bypass_cache: bool = False
    custom_cache: Optional[DiskLLMCache] = None

    _semaphore: threading.Semaphore = PrivateAttr(default_factory=_get_thread_semaphore)

    @property
    def _llm_type(self) -> str:
        return f"cached_{self.provider}_{self.model_name}"

    def _get_cache(self) -> DiskLLMCache:
        return self.custom_cache or llm_cache

    def _call_with_retry(self, messages: List[BaseMessage], **kwargs: Any) -> AIMessage:
        """Call underlying LLM with concurrency control and exponential backoff on rate limits."""
        retries = 0
        backoff = settings.llm_retry_min_wait

        while True:
            with self._semaphore:
                try:
                    res = self.underlying_model.invoke(messages, **kwargs)
                    return res
                except Exception as exc:
                    if _is_rate_limit_error(exc) and retries < settings.llm_max_retries:
                        retries += 1
                        import re
                        wait_hint = 0.0
                        msg_str = str(exc).lower()
                        match_s = re.search(r"try again in ([\d\.]+)s", msg_str)
                        match_ms = re.search(r"try again in ([\d\.]+)ms", msg_str)
                        if match_s:
                            wait_hint = float(match_s.group(1)) + 1.5
                        elif match_ms:
                            wait_hint = (float(match_ms.group(1)) / 1000.0) + 1.5

                        jitter = random.uniform(0.5, 2.0)
                        sleep_time = max(wait_hint, min(backoff + jitter, settings.llm_retry_max_wait))
                        logger.warning(
                            "Rate limit hit for %s/%s. Retry %d/%d sleeping %.2fs. Error: %s",
                            self.provider,
                            self.model_name,
                            retries,
                            settings.llm_max_retries,
                            sleep_time,
                            exc,
                        )
                        time.sleep(sleep_time)
                        backoff = min(backoff * 1.5, settings.llm_retry_max_wait)
                    else:
                        raise

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        cache = self._get_cache()

        # 1. Check cache
        if not self.bypass_cache and cache.enabled:
            cached_item = cache.get(
                self.provider, self.model_name, self.temperature, messages
            )
            if cached_item is not None:
                metadata = cached_item.get("metadata", {})
                metadata["cached"] = True
                usage_metadata = metadata.get("usage_metadata")
                msg = AIMessage(
                    content=cached_item["content"],
                    response_metadata=metadata,
                    usage_metadata=usage_metadata,
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])

        # 2. Invoke underlying model
        res = self._call_with_retry(messages, stop=stop, **kwargs)

        # 3. Extract usage
        usage_meta: Dict[str, Any] = {}
        if hasattr(res, "usage_metadata") and res.usage_metadata:
            usage_meta = dict(res.usage_metadata)
        elif hasattr(res, "response_metadata") and "token_usage" in res.response_metadata:
            tu = res.response_metadata["token_usage"]
            usage_meta = {
                "input_tokens": tu.get("prompt_tokens", 0),
                "output_tokens": tu.get("completion_tokens", 0),
                "total_tokens": tu.get("total_tokens", 0),
            }
        else:
            # Approximate fallback if provider omits tokens
            prompt_str = serialize_prompt(messages)
            usage_meta = {
                "input_tokens": max(1, len(prompt_str) // 4),
                "output_tokens": max(1, len(str(res.content)) // 4),
                "total_tokens": max(2, (len(prompt_str) + len(str(res.content))) // 4),
            }

        metadata = dict(getattr(res, "response_metadata", {}))
        metadata["usage_metadata"] = usage_meta
        metadata["cached"] = False

        # 4. Save to cache
        if not self.bypass_cache and cache.enabled:
            cache.set(
                self.provider,
                self.model_name,
                self.temperature,
                messages,
                str(res.content),
                metadata=metadata,
            )

        res.response_metadata = metadata
        res.usage_metadata = usage_meta
        return ChatResult(generations=[ChatGeneration(message=res)])


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    bypass_cache: bool = False,
    custom_cache: Optional[DiskLLMCache] = None,
) -> CachedChatModel:
    """Factory creating a unified, cached ChatModel instance."""
    prov = (provider or settings.llm_provider).lower()
    mod = model or settings.llm_model

    if prov == "groq":
        from langchain_groq import ChatGroq

        underlying = ChatGroq(
            model=mod,
            temperature=temperature,
            api_key=settings.groq_api_key,
        )
    elif prov == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        underlying = ChatGoogleGenerativeAI(
            model=mod,
            temperature=temperature,
            google_api_key=settings.gemini_api_key,
        )
    elif prov == "openai":
        from langchain_openai import ChatOpenAI

        underlying = ChatOpenAI(
            model=mod,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{prov}'. Must be one of: 'groq', 'gemini', 'openai'."
        )

    return CachedChatModel(
        underlying_model=underlying,
        provider=prov,
        model_name=mod,
        temperature=temperature,
        bypass_cache=bypass_cache,
        custom_cache=custom_cache,
    )
