"""True subagents with isolated contexts and bounded concurrency.

Unlike the cheap-model *review* gate (a single prompt on the diff), a subagent
is a small autonomous worker: it receives only its own task and a focused
context, can call a restricted set of read-only tools, and returns a concise
report. Workers never share the main agent's message history and never write
files, so they cannot corrupt the primary turn.

Roles
-----
explore : map the codebase, find relevant files/symbols, explain how things fit
review  : independent critic of a diff / plan / answer
test    : test analyst — find missing cases and likely root causes

A shared semaphore bounds concurrency (default 2) and a per-turn cap bounds
cost. Everything is best-effort: a failed worker returns a structured error
and the main loop continues.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.get_response import chat

MAX_STEPS = 4
DEFAULT_CONCURRENCY = 2
READONLY_TOOLS = {
    "read", "ls", "grep", "glob", "symbols", "find_refs",
    "git_status", "git_diff", "git_log", "git_blame",
    "lsp_definition", "lsp_references", "lsp_diagnostics",
}

ROLE_SYSTEM: Dict[str, str] = {
    "explore": """You are an exploration subagent for a terminal coding agent.
You have read-only tools. Use them to locate files, symbols and call chains.
Return a terse report: relevant files, key symbols, how the pieces connect, and
the single best next step for the main agent. Max 250 words. No preamble.""",
    "review": """You are an independent review subagent. Be adversarial but concrete.
Given the diff/plan/answer and any test output, list only real defects: bugs,
regressions, unsafe changes, missed edge cases. Max 8 numbered one-line items,
or exactly OK. Do not restate the diff.""",
    "test": """You are a test-analysis subagent. Given a change and test output,
identify the most likely root cause of any failure and the smallest fix, plus
the 1-3 missing test cases that would prevent regression. Max 200 words.""",
    "recon": """You are a reconnaissance subagent. Build target state, not noise.
Use read-only tools to identify hosts, services, endpoints, HTTP methods, auth surfaces,
tech fingerprints, and exposed inputs. Return structured lines the main agent can ingest:
HOST <host> <notes> | SERVICE <host> <port> <proto> <name> | ENDPOINT <url> <notes>
FINDING <severity> <title> :: <details>. End with the highest-value next probe.""",
    "vuln_analysis": """You are a vulnerability-analysis subagent. Reason from evidence.
Correlate endpoints, params, auth state, and responses to form testable hypotheses.
Do not claim exploitation. For each hypothesis give: asset, hypothesis, why plausible,
minimal non-destructive test, expected signal, and how it could chain with other findings.""",
    "exploit_validation": """You are an exploit-validation subagent. You are proof-first.
You cannot execute destructive actions; propose or analyze validation using available evidence.
For each candidate: status (attempted/proved/false_positive), exact reproduction steps,
evidence needed/captured, impact, and safe remediation. Never mark proved without reproducible evidence.""",
    "reporting": """You are a reporting subagent. Produce structured, defensible output.
Summarize scope, target state, proved vs attempted findings, reproduction steps, evidence refs,
severity, impact, and remediation. Be precise and omit unsupported claims.""",
}

SUBTOOL_RE = re.compile(r"```subtool\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class SubagentResult:
    role: str
    task: str
    ok: bool
    text: str
    model: str
    duration: float
    error: str = ""
    steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_subtool(text: str) -> Optional[Dict[str, Any]]:
    match = SUBTOOL_RE.search(text or "")
    if not match:
        # Tolerate a bare JSON object when the model ignores the fence.
        match2 = re.search(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*\}\s*\})", text or "", re.DOTALL)
        raw = match2.group(1) if match2 else None
    else:
        raw = match.group(1)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("name") and isinstance(obj.get("arguments", {}), dict):
            return {"name": str(obj["name"]), "arguments": obj.get("arguments") or {}}
    except Exception:
        pass
    return None


def _strip_subtool(text: str) -> str:
    return SUBTOOL_RE.sub("", text or "").strip()


async def run_subagent(
    role: str,
    task: str,
    *,
    model: str,
    cwd: str,
    context: str = "",
    max_steps: int = MAX_STEPS,
    timeout: int = 60,
    registry_factory: Optional[Callable[[], Any]] = None,
) -> SubagentResult:
    """Run one isolated worker. Never raises."""
    started = time.perf_counter()
    role = role if role in ROLE_SYSTEM else "explore"
    system = ROLE_SYSTEM[role] + f"\n\nWorking directory: {cwd}\nRead-only tools: {', '.join(sorted(READONLY_TOOLS))}"
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"TASK:\n{task[:4000]}\n\nCONTEXT:\n{(context or '')[:9000]}"},
    ]
    text = ""
    steps = 0
    try:
        from agent.core import ToolRegistry

        registry = registry_factory() if registry_factory else ToolRegistry(disabled=set())
        for steps in range(1, max_steps + 1):
            answer = (await asyncio.to_thread(chat, messages, model=model, timeout=timeout)) or ""
            tool = _extract_subtool(answer)
            if tool is None:
                text = _strip_subtool(answer) or answer.strip()
                break
            name = tool["name"]
            if name not in READONLY_TOOLS:
                messages.append({"role": "user", "content": f"Tool '{name}' is not available to you. Use only: {', '.join(sorted(READONLY_TOOLS))}. Continue."})
                continue
            visible = _strip_subtool(answer)
            result = await registry.run(name, tool["arguments"])
            messages.append({"role": "assistant", "content": visible or answer.strip()})
            messages.append({
                "role": "user",
                "content": f"TOOL RESULT ({name}):\n{result[:8000]}\n\nContinue. Return a final answer without a subtool block when done.",
            })
        else:
            text = "Step budget exhausted without a final answer; report partial findings only."
        return SubagentResult(role=role, task=task, ok=True, text=text[:6000], model=model, duration=_dur(started), steps=steps)
    except Exception as exc:  # noqa: BLE001
        return SubagentResult(role=role, task=task, ok=False, text="", model=model, duration=_dur(started), error=str(exc)[:300], steps=steps)


def _dur(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


class SubagentPool:
    """Bounded-concurrency runner with a per-turn cap and result aggregation."""

    def __init__(self, model: str, cwd: str, *, max_concurrent: int = DEFAULT_CONCURRENCY, max_total: int = 3):
        self.model = model
        self.cwd = cwd
        self.sem = asyncio.Semaphore(max(1, max_concurrent))
        self.max_total = max(0, max_total)
        self.used = 0
        self.results: List[SubagentResult] = []

    def available(self) -> bool:
        return self.used < self.max_total

    async def run(self, role: str, task: str, context: str = "") -> SubagentResult:
        if not self.available():
            return SubagentResult(role=role, task=task, ok=False, text="", model=self.model, duration=0.0, error="subagent budget reached this turn")
        self.used += 1
        async with self.sem:
            result = await run_subagent(role, task, model=self.model, cwd=self.cwd, context=context)
        self.results.append(result)
        return result

    async def run_many(self, tasks: List[Tuple[str, str, str]]) -> List[SubagentResult]:
        """Run several independent workers concurrently; each respects the cap."""
        picked: List[Tuple[str, str, str]] = []
        for item in tasks:
            if not self.available():
                break
            self.used += 1
            picked.append(item)
        async def guarded(role: str, task: str, ctx: str) -> SubagentResult:
            async with self.sem:
                return await run_subagent(role, task, model=self.model, cwd=self.cwd, context=ctx)

        gathered = await asyncio.gather(
            *[guarded(role, task, ctx) for role, task, ctx in picked],
            return_exceptions=True,
        )
        results: List[SubagentResult] = []
        for (role, task, _ctx), item in zip(picked, gathered):
            if isinstance(item, BaseException):
                item = SubagentResult(role=role, task=task, ok=False, text="", model=self.model, duration=0.0, error=str(item)[:300])
            results.append(item)
            self.results.append(item)
        return results


def format_results(results: List[SubagentResult]) -> str:
    if not results:
        return "(no subagent results)"
    blocks: List[str] = ["# Subagent findings"]
    for result in results:
        header = f"[{result.role}] {result.task[:120]}"
        body = result.text or (f"FAILED: {result.error}" if result.error else "no output")
        blocks.append(f"## {header}\n{body}")
    return "\n\n".join(blocks)
