"""Deterministic disk cache for LLM calls keyed by provider, model, temp, and prompt hash."""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings


def serialize_prompt(prompt_input: Any) -> str:
    """Deterministically serialize prompt inputs (strings, message lists, etc.)."""
    if isinstance(prompt_input, str):
        return prompt_input
    if isinstance(prompt_input, list):
        items = []
        for msg in prompt_input:
            if hasattr(msg, "content"):
                role = getattr(msg, "type", "message")
                items.append(f"{role}:{msg.content}")
            elif isinstance(msg, tuple) and len(msg) == 2:
                items.append(f"{msg[0]}:{msg[1]}")
            elif isinstance(msg, dict):
                items.append(f"{msg.get('role', 'user')}:{msg.get('content', '')}")
            else:
                items.append(str(msg))
        return "\n---\n".join(items)
    return str(prompt_input)


def compute_cache_key(
    provider: str,
    model: str,
    temperature: float,
    prompt_str: str,
) -> str:
    """Compute deterministic SHA256 cache key."""
    canonical_repr = f"provider={provider}|model={model}|temp={temperature:.4f}|prompt={prompt_str}"
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


class DiskLLMCache:
    """Local disk cache storing LLM responses deterministically."""

    def __init__(self, cache_dir: Optional[Path] = None, enabled: Optional[bool] = None) -> None:
        self.cache_dir = Path(cache_dir or settings.llm_cache_dir)
        self.enabled = settings.llm_cache_enabled if enabled is None else enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        provider: str,
        model: str,
        temperature: float,
        prompt_input: Any,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available, else None."""
        if not self.enabled:
            return None

        prompt_str = serialize_prompt(prompt_input)
        key = compute_cache_key(provider, model, temperature, prompt_str)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return None

    def set(
        self,
        provider: str,
        model: str,
        temperature: float,
        prompt_input: Any,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store response in disk cache atomically."""
        prompt_str = serialize_prompt(prompt_input)
        key = compute_cache_key(provider, model, temperature, prompt_str)

        if not self.enabled:
            return key

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "prompt": prompt_str,
            "content": content,
            "metadata": metadata or {},
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        target_file = self.cache_dir / f"{key}.json"
        # Atomic write
        temp_dir = self.cache_dir / ".tmp"
        temp_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            json.dump(record, tf, ensure_ascii=False, indent=2)
            temp_name = tf.name

        os.replace(temp_name, target_file)
        return key

    def clear(self) -> int:
        """Remove all cache files, returning the count of deleted entries."""
        count = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        return count


llm_cache = DiskLLMCache()

