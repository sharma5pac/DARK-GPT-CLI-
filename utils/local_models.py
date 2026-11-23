"""Best-effort discovery of local OpenAI-compatible model servers.

Supports Ollama (``127.0.0.1:11434``) and vLLM / llama.cpp / LM Studio style
``/v1/models`` endpoints. Every probe has a short timeout and failures are
silent: the model picker still works with OpenRouter when nothing is local.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, List

OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
OPENAI_COMPAT_URLS = [
    "http://127.0.0.1:8000/v1/models",
    "http://127.0.0.1:1234/v1/models",
    "http://127.0.0.1:8080/v1/models",
]


def _get_json(url: str, timeout: float = 1.2) -> Any:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return None


def discover_local_models() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    seen = set()

    payload = _get_json(OLLAMA_URL)
    if isinstance(payload, dict):
        for item in payload.get("models") or []:
            name = item.get("name") or item.get("model") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            models.append(
                {
                    "id": name,
                    "name": f"{name} (Ollama)",
                    "context": 32768,
                    "price_in": 0.0,
                    "price_out": 0.0,
                    "free": True,
                    "uncensored": False,
                    "vision": False,
                    "local": True,
                    "provider": "Ollama",
                }
            )

    for url in OPENAI_COMPAT_URLS:
        payload = _get_json(url)
        if not isinstance(payload, dict):
            continue
        for item in payload.get("data") or []:
            name = item.get("id") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            models.append(
                {
                    "id": name,
                    "name": f"{name} (local)",
                    "context": int(item.get("context_length") or 32768),
                    "price_in": 0.0,
                    "price_out": 0.0,
                    "free": True,
                    "uncensored": False,
                    "vision": False,
                    "local": True,
                    "provider": "Local",
                }
            )
    return models
