"""Independent diff review and test-failure analysis on a cheap model."""

from __future__ import annotations

import asyncio
from typing import List

REVIEW_SYSTEM = """You are a strict code reviewer. Read the diff and test output.
Report only concrete bugs, regressions, unsafe changes, or missing tests.
Return a numbered list, max 8 items, each one sentence. If nothing is wrong, reply OK."""

TEST_SYSTEM = """You are a debugging assistant. Given test failures and the diff,
identify the most likely root cause and the single smallest fix. Reply in at most
4 sentences. If tests passed, reply OK."""


def _ask(system: str, body: str, model: str) -> str:
    try:
        from utils.get_response import chat

        return (chat(
            [{"role": "system", "content": system}, {"role": "user", "content": body[:9000]}],
            model=model,
            timeout=60,
        ) or "").strip()[:3000]
    except Exception:
        return ""


async def isolated_checks(cwd: str) -> str:
    """Run the project's checks on a detached copy of the current diff."""
    try:
        from utils import verify, worktree

        info = await asyncio.to_thread(worktree.create, cwd)
        if not info:
            return ""
        if not info.get("applied"):
            output = "Patch did not apply cleanly in the isolated worktree."
        else:
            commands = await asyncio.to_thread(verify.detect, str(info.get("path") or ""))
            results = await asyncio.to_thread(verify.run, str(info.get("path") or ""), commands)
            output = verify.format_results(results)
        await asyncio.to_thread(worktree.cleanup, info)
        return "Isolated worktree checks:\n" + output
    except Exception:
        return ""


async def review(diff: str, tests: str, model: str) -> str:
    """Run reviewer and test analyst concurrently; never raises."""
    if not diff and not tests:
        return ""
    body = f"DIFF:\n{diff[:6000]}\n\nTEST OUTPUT:\n{tests[:3000]}"
    results = await asyncio.gather(
        asyncio.to_thread(_ask, REVIEW_SYSTEM, body, model),
        asyncio.to_thread(_ask, TEST_SYSTEM, body, model),
        return_exceptions=True,
    )
    parts: List[str] = []
    names = ("Review", "Tests")
    for name, result in zip(names, results):
        text = "" if isinstance(result, BaseException) else str(result or "").strip()
        if text and text.upper() != "OK":
            parts.append(f"{name}: {text}")
    return "\n".join(parts)
