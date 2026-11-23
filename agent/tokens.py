"""Model-aware token accounting for the context meter.

Uses tiktoken when its encoding is available; otherwise falls back to a stable
character heuristic. The context limit comes from the live OpenRouter catalogue
when the model is known, and from a conservative default otherwise.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

DEFAULT_CONTEXT = 128_000

_enc_lock = threading.Lock()
_encoding = None
_encoding_failed = False


def _get_encoding():
    global _encoding, _encoding_failed
    if _encoding is not None or _encoding_failed:
        return _encoding
    with _enc_lock:
        if _encoding is not None or _encoding_failed:
            return _encoding
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoding_failed = True
            _encoding = None
    return _encoding


def count_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    # Heuristic: ~4 chars/token, never zero for non-empty text.
    return max(1, (len(text) + 3) // 4)


def _messages_text(messages: Optional[List[Dict[str, Any]]]) -> str:
    if not messages:
        return ""
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False, default=str)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def context_limit(model: str) -> int:
    try:
        from utils.get_response import get_model_info

        info = get_model_info(model)
        if info:
            ctx = int(info.get("context") or 0)
            if ctx > 0:
                return ctx
    except Exception:
        pass
    return DEFAULT_CONTEXT


def usage(
    *,
    model: str,
    system: str = "",
    skills: str = "",
    tools: Optional[List[Dict[str, Any]]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    prompt: str = "",
    input_text: str = "",
) -> Dict[str, Any]:
    tool_text = json.dumps(tools or [], ensure_ascii=False, default=str)
    combined = "\n".join(
        part
        for part in (
            system,
            skills,
            tool_text,
            _messages_text(history),
            prompt,
            input_text,
        )
        if part
    )
    tokens = count_tokens(combined)
    limit = context_limit(model)
    pct = min(100.0, (tokens / limit) * 100.0) if limit else 0.0
    return {"tokens": tokens, "limit": limit, "pct": pct}


def format_usage(data: Dict[str, Any]) -> str:
    tokens = int(data.get("tokens") or 0)
    pct = float(data.get("pct") or 0.0)
    if tokens >= 1_000_000:
        amount = f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        amount = f"{tokens / 1_000:.1f}K"
    else:
        amount = str(tokens)
    return f"{amount} ({pct:.0f}%)"
