"""Cheap-model routing for planning, review and bookkeeping calls.

The main model does the coding; a free/cheap model handles the plan, diff
review and post-turn memory note. Any failure is non-fatal - the main turn
continues without the optional step.
"""

from __future__ import annotations

from typing import Dict, List, Optional

PREFERRED_HINTS = ("nano", "mini", "flash", "light", "small", "haiku", "lite", "free")


def pick_cheap_model(main_model: str = "", preferred: str = "") -> str:
    preferred = (preferred or "").strip()
    if preferred:
        return preferred
    try:
        from utils.get_response import get_catalog

        catalog = get_catalog() or {}
        free: List[Dict] = list(catalog.get("free") or [])
    except Exception:
        free = []
    ids = [str(item.get("id") or "") for item in free if item.get("id")]
    ids = [mid for mid in ids if mid != main_model]
    for hint in PREFERRED_HINTS:
        for mid in ids:
            if hint in mid.lower():
                return mid
    return ids[0] if ids else "openrouter/free"


PLAN_SYSTEM = """You are a planning assistant for a terminal coding agent.
Given a task, return a terse numbered plan (max 6 steps) of files to inspect,
changes to make, and how to verify. No prose, no markdown fences."""


def make_plan(prompt: str, model: str, cwd: str = "") -> str:
    try:
        from utils.get_response import chat

        answer = chat(
            [
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": f"cwd: {cwd}\n\ntask: {prompt[:4000]}"},
            ],
            model=model,
            timeout=60,
        )
        return (answer or "").strip()[:3000]
    except Exception:
        return ""


def make_memory_note(prompt: str, answer: str, model: str) -> str:
    text = f"Task: {prompt[:500]}\nOutcome: {answer[:800]}"
    try:
        from utils.get_response import chat

        note = chat(
            [
                {"role": "system", "content": "Summarize one durable project fact or convention learned from this turn in one sentence. No preamble."},
                {"role": "user", "content": text},
            ],
            model=model,
            timeout=45,
        )
        return (note or "").strip()[:600]
    except Exception:
        return ""
