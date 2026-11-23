"""Token-aware context packing, compaction summaries and project rules.

Keeps the system prompt and the most recent turns intact, drops exact
duplicates, and *compacts* older messages into a structured summary instead of
silently throwing them away. This is what lets long coding sessions continue
past the model's context window without losing decisions or file state.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from agent import tokens as token_util

DEFAULT_CHAR_BUDGET = 48_000


def _content(message: Dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return str(content)


_FILE_RE = re.compile(r"(?:[\w.-]+/)+[\w.-]+\.\w{1,6}|\b[\w-]+\.(?:py|js|jsx|ts|tsx|rs|go|md|json|toml|yaml|yml|css|html)\b")


def summarize(messages: List[Dict[str, Any]]) -> str:
    """Deterministic, offline compaction: preserve facts that matter later."""
    if not messages:
        return ""
    users: List[str] = []
    decisions: List[str] = []
    files: List[str] = []
    tools: Dict[str, int] = {}
    for message in messages:
        role = str(message.get("role") or "")
        text = _content(message)
        if role == "user" and text.strip() and not text.lstrip().startswith("Tool result"):
            users.append(text.strip().replace("\n", " ")[:300])
        if "REVIEW FINDINGS" in text or "Verification results" in text:
            decisions.append(text.strip().replace("\n", " ")[:300])
        for found in _FILE_RE.findall(text):
            if found not in files:
                files.append(found)
        for tool in re.findall(r'\b(?:symbols|find_refs|rename|patch|write|edit|git_status|git_diff|git_log|git_blame|lsp_definition|lsp_references|lsp_diagnostics|lsp_rename|web_search|mcp__[\w:]+)\b', text):
            tools[tool] = tools.get(tool, 0) + 1
    blocks: List[str] = []
    if users:
        blocks.append("User requests (chronological):\n" + "\n".join(f"- {u}" for u in users[-6:]))
    if files:
        blocks.append("Files touched or discussed: " + ", ".join(files[:30]))
    if tools:
        top = sorted(tools.items(), key=lambda pair: pair[1], reverse=True)[:10]
        blocks.append("Tools used: " + ", ".join(f"{name}({count})" for name, count in top))
    if decisions:
        blocks.append("Verification / review notes:\n" + "\n".join(f"- {d}" for d in decisions[-4:]))
    blocks.append(f"Compacted {len(messages)} earlier message(s); newest turns follow verbatim.")
    return "\n\n".join(blocks)


def compact(messages: List[Dict[str, Any]], keep_recent: int = 6) -> List[Dict[str, Any]]:
    """Replace all but the newest turns with a structured summary message."""
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) <= keep_recent:
        return list(messages)
    older, newer = rest[:-keep_recent], rest[-keep_recent:]
    summary = summarize(older)
    marker = {"role": "system", "content": "[Earlier conversation summary]\n" + summary}
    return system + [marker] + newer


def checkpoint_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact, persistable state for /checkpoint and /resume."""
    from datetime import datetime, timezone

    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "turns": len(turns),
        "summary": summarize(messages),
        "last_user": next((_content(m) for m in reversed(turns) if m.get("role") == "user"), ""),
        "last_assistant": next((_content(m) for m in reversed(turns) if m.get("role") == "assistant"), ""),
    }


def pack(messages: List[Dict[str, Any]], char_budget: int = DEFAULT_CHAR_BUDGET, summarize_dropped: bool = True) -> List[Dict[str, Any]]:
    if not messages:
        return []
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if not rest:
        return system

    # De-duplicate identical consecutive turns (a common ReAct artifact).
    cleaned: List[Dict[str, Any]] = []
    for message in rest:
        text = _content(message).strip()
        if cleaned and _content(cleaned[-1]).strip() == text and not message.get("tool_calls"):
            continue
        cleaned.append(message)

    # Keep the newest turns; older ones are compacted into a summary.
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    used = 0
    for message in reversed(cleaned):
        text = _content(message)
        if len(text) > char_budget:
            text = text[-char_budget:]
            message = dict(message)
            message["content"] = f"[...earlier content truncated...]\n" + text
        if used + len(text) > char_budget and kept:
            dropped.append(message)
            continue
        used += len(text)
        kept.append(message)
    kept.reverse()
    dropped.reverse()
    result = list(system)
    if dropped and summarize_dropped:
        digest = summarize(dropped)
        if digest:
            result.append({"role": "system", "content": "[Earlier conversation summary]\n" + digest})
    return result + kept


def system_extras(
    cwd: str,
    prompt: str,
    *,
    git: bool = True,
    memory: bool = True,
    self_learning: bool = True,
) -> str:
    """Build the git + memory block that should be prepended to the system prompt."""
    parts: List[str] = []
    try:
        from agent import rules as rule_util

        rules = rule_util.format_rules(cwd)
        if rules:
            parts.append(rules)
        facts = rule_util.detected_facts(cwd)
        if facts:
            parts.append("# Detected project facts\n" + facts)
    except Exception:
        pass
    if git:
        from utils import vcs

        snapshot = vcs.snapshot(cwd)
        if snapshot:
            parts.append(snapshot)
    if memory:
        from agent import memory as mem

        entries = mem.recall(cwd, prompt, k=5)
        block = mem.format_memories(entries)
        if block:
            parts.append(block)
    if self_learning:
        from agent import memory as mem

        lessons = mem.format_lessons(mem.by_kind(cwd, {"lesson", "failed-approach", "correction"}, k=8))
        if lessons:
            parts.append(lessons)
        patterns = mem.format_patterns(mem.by_kind(cwd, {"pattern"}, k=6))
        if patterns:
            parts.append(patterns)
    # Active target state — gives the agent dynamic memory across turns.
    try:
        from utils import appconfig as _cfg
        from agent import knowledge_graph as kg, evidence as ev, planner as pl
        target = str(_cfg.get("active_target") or "").strip()
        if target:
            g = kg.summary(target)
            c = g.get("counts") or {}
            lines = [f"Active target: {target}", f"Graph: hosts={c.get('host',0)} services={c.get('service',0)} endpoints={c.get('endpoint',0)} findings={c.get('finding',0)}"]
            for f in g.get("recent_findings") or []:
                lines.append(f"- finding [{f.get('severity','info')}/{f.get('status','attempted') if 'status' in f else ''}] {f.get('title','')}: {(f.get('details') or '')[:120]}")
            e = ev.summary(target)
            lines.append(f"Evidence: {e.get('count',0)} entries, last={e.get('last_ts') or '-'}")
            p = pl.summary(target)
            if p.get("objective"):
                lines.append(f"Objective: {p['objective'][:160]}")
            for h in p.get("open_hypotheses") or []:
                lines.append(f"- hypothesis [{h.get('status','open')}] {str(h.get('text',''))[:120]}")
            for o in (p.get("recent_observations") or [])[-3:]:
                lines.append(f"- obs: {str(o.get('observed',''))[:100]} | delta: {str(o.get('hypothesis_delta',''))[:80]} | next: {str(o.get('next_action',''))[:80]}")
            for d in (p.get("dead_ends") or [])[-3:]:
                lines.append(f"- dead_end: {str(d.get('signature',''))[:80]} -> pivot: {str(d.get('pivot',''))[:80]}")
            parts.append("Target state (dynamic memory — update it as you learn):\n" + "\n".join(lines))
    except Exception:
        pass
    return "\n\n".join(parts)


def budget_for(model: str, fraction: float = 0.35) -> int:
    """Char budget derived from the model's context limit."""
    limit = token_util.context_limit(model)
    return max(8_000, int(limit * fraction * 4))
