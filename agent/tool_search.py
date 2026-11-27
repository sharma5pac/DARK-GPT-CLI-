"""Deferred tool loading with an indexed tool catalog.

The agent used to publish every built-in and all 361 MCP tool schemas on every
model turn. This module implements the modern "tool search" pattern: a small
always-on set plus one ``tool_search`` meta tool stays in the request, and the
model loads the full schema of any deferred tool on demand.

The index is deterministic and dependency-free (token/TF-IDF scoring), so it
works offline and produces repeatable results in tests. Dispatch is unchanged:
tools are still resolved by name in ``ToolRegistry._dispatch``; search only
controls which *schemas* are advertised to the model.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from utils import appconfig

VERSION = 1

SEARCH_TOOL = "tool_search"

# One shared index per process. The agent rebuilds tools from it every turn and
# the TUI previews/loads/resets the same object, so GUI controls affect real
# runtime state instead of a decorative counter.
_CURRENT_INDEX: Optional[ToolIndex] = None

# Tools that must stay in every request. Keep this deliberately small: the
# point is to stop shipping hundreds of unused MCP schemas on every turn.
ALWAYS_ON: Tuple[str, ...] = (
    "shell", "read", "write", "edit", "ls", "grep", "glob",
    "script", "math_attack", "ask_question", "submit_and_exit",
    "task_start", "task_list", "task_wait", "task_log", "task_kill",
    "discover_tools", "autonomous_route",
    "semantic_recall", "semantic_remember", "consult_advisor",
    "guard_status", "replay_run",
    SEARCH_TOOL,
)

STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "to", "of", "in", "is", "for", "on",
    "with", "that", "this", "it", "as", "at", "by", "from", "you", "use",
    "using", "tool", "function", "parameters", "object", "string", "required",
}

_WORD_RE = re.compile(r"[a-z0-9_]+")

# Common pentest synonyms keep recall high when the model and a tool use
# different words for the same capability.
SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "sqli": ("sql", "injection", "database"),
    "sql": ("sqli", "injection", "database"),
    "xss": ("cross", "site", "scripting", "html"),
    "rce": ("remote", "code", "execution", "command"),
    "ssrf": ("server", "side", "request", "forgery", "request"),
    "lfi": ("local", "file", "inclusion", "read"),
    "auth": ("authentication", "login", "session", "token"),
    "fuzz": ("fuzzing", "brute", "wordlist", "enumerate"),
    "scan": ("scanner", "enumerate", "recon", "discovery"),
    "web": ("http", "url", "browser", "site"),
    "gui": ("desktop", "window", "click", "accessibility"),
    "firmware": ("flash", "rom", "binwalk", "hardware"),
    "report": ("export", "markdown", "html", "findings"),
    "memory": ("recall", "remember", "knowledge"),
    "guard": ("injection", "poisoning", "security", "prompt"),
    "math": ("symbolic", "number", "entropy", "statistics", "prime", "graph"),
    "crypto": ("rsa", "modular", "prime", "factor", "cipher", "key"),
    "riemann": ("rh", "grh", "zeta", "lfunction", "proof"),
    "entropy": ("prng", "random", "token", "bias", "distribution"),
    "decide": ("route", "choose", "plan", "should", "what to do"),
}


def _tokens(text: str) -> Set[str]:
    words = {w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS and len(w) > 1}
    expanded = set(words)
    for word in tuple(words):
        for related in SYNONYMS.get(word, ()):
            expanded.add(related)
    return expanded


def spec_name(spec: Dict[str, Any]) -> str:
    try:
        return str(spec["function"]["name"])
    except Exception:
        return ""


def spec_description(spec: Dict[str, Any]) -> str:
    try:
        return str(spec["function"].get("description") or "")
    except Exception:
        return ""


def search_tool_spec(default_limit: int = 6) -> Dict[str, Any]:
    """The only catalog tool advertised by default when deferral is enabled."""
    return {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL,
            "description": (
                "Search the full DarkGPT tool catalog (built-ins + all MCP servers) and load "
                "the complete JSON schemas of the best matching tools into context. Call this "
                "whenever the tool you need is not in the always-on set, when you need an MCP/"
                "Kala/desktop tool, or when you are unsure which tool fits. The result includes "
                "ready-to-call parameter schemas; use the returned 'name' values on the next turn. "
                "Use query text like 'nmap port scan', 'GUI click button', 'SQL injection payload', "
                "'generate report', or an exact tool name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Capability, tool name, or goal."},
                    "limit": {"type": "integer", "description": f"Number of schemas to load (1-12, default {max(1, min(12, int(default_limit or 6)))})."},
                    "include_deferred_only": {
                        "type": "boolean",
                        "description": "Skip tools already loaded this session (default true).",
                    },
                },
                "required": ["query"],
            },
        },
    }


class ToolIndex:
    """Searchable index over tool specs that are not advertised by default."""

    def __init__(self, specs: Sequence[Dict[str, Any]], always_on: Iterable[str] = ALWAYS_ON) -> None:
        self._specs: Dict[str, Dict[str, Any]] = {}
        self._docs: Dict[str, Set[str]] = {}
        self.always_on: Set[str] = set(always_on)
        for spec in specs or []:
            name = spec_name(spec)
            if not name or name in self.always_on:
                continue
            self._specs[name] = spec
            self._docs[name] = _tokens(
                name + " " + spec_description(spec) + " " + self._schema_text(spec)
            )
        self._idf = self._compute_idf()
        self.loaded: Set[str] = set()

    @staticmethod
    def _schema_text(spec: Dict[str, Any]) -> str:
        try:
            return json.dumps(spec["function"].get("parameters") or {}, ensure_ascii=False)[:1200]
        except Exception:
            return ""

    def _compute_idf(self) -> Dict[str, float]:
        n = max(1, len(self._docs))
        df: Counter[str] = Counter()
        for words in self._docs.values():
            df.update(words)
        return {word: math.log(1.0 + n / (1.0 + count)) for word, count in df.items()}

    def names(self) -> List[str]:
        return sorted(self._specs)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._specs.get(str(name) or "")

    def load(self, names: Iterable[str]) -> List[str]:
        """Mark tools as loaded; returns names that actually exist in the index."""
        loaded: List[str] = []
        for name in names or []:
            key = str(name)
            if key in self._specs and key not in self.loaded:
                self.loaded.add(key)
                loaded.append(key)
        return loaded

    def reset(self) -> int:
        count = len(self.loaded)
        self.loaded.clear()
        return count

    def search(
        self,
        query: str,
        limit: int = 6,
        *,
        skip_loaded: bool = True,
    ) -> List[Dict[str, Any]]:
        q = _tokens(query)
        if not q:
            return []
        needles = [w for w in _WORD_RE.findall((query or "").lower()) if len(w) > 1]
        results: List[Tuple[float, str]] = []
        for name, words in self._docs.items():
            if skip_loaded and name in self.loaded:
                continue
            score = sum(self._idf.get(word, 0.0) for word in (q & words))
            compact_name = re.sub(r"[^a-z0-9]", "", name.lower())
            for needle in needles:
                if needle and needle in compact_name:
                    score += 4.0
            if score <= 0:
                continue
            results.append((score, name))
        results.sort(key=lambda pair: (-pair[0], pair[1]))
        limit = max(1, min(12, int(limit or 6)))
        return [
            {
                "name": name,
                "score": round(score, 3),
                "summary": spec_description(self._specs[name])[:300],
                "spec": self._specs[name],
            }
            for score, name in results[:limit]
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "indexed": len(self._specs),
            "always_on": len(self.always_on),
            "loaded": len(self.loaded),
            "loaded_names": sorted(self.loaded),
            "enabled": deferral_enabled(),
        }


def current_index(specs: Optional[Sequence[Dict[str, Any]]] = None) -> ToolIndex:
    """Return the process-wide index, rebuilding it when the tool catalog changes."""
    global _CURRENT_INDEX
    if _CURRENT_INDEX is None or specs is not None:
        _CURRENT_INDEX = ToolIndex(specs or [])
    return _CURRENT_INDEX


def set_index(index: Optional[ToolIndex]) -> None:
    global _CURRENT_INDEX
    _CURRENT_INDEX = index


def reset_index() -> int:
    """Clear loaded schemas; returns how many were unloaded."""
    if _CURRENT_INDEX is None:
        return 0
    return _CURRENT_INDEX.reset()


def deferral_enabled() -> bool:
    return bool(appconfig.get("deferred_tools_enabled", True))


def build_active_specs(
    all_specs: Sequence[Dict[str, Any]],
    index: Optional[ToolIndex] = None,
) -> Tuple[List[Dict[str, Any]], ToolIndex]:
    """Return (specs to advertise, index of deferred tools).

    When deferral is disabled the full catalog is advertised and the index is
    still built so ``tool_search`` can be used as a plain catalog browser.
    """
    tool_index = index or current_index(all_specs)
    caps = _clamp_limit(appconfig.get("tool_search_top_k", 6), 6, 1, 12)
    if not deferral_enabled():
        full = list(all_specs)
        if not any(spec_name(s) == SEARCH_TOOL for s in full):
            full.append(search_tool_spec(caps))
        return full, tool_index
    tool_index.always_on = set(ALWAYS_ON)
    active = [s for s in all_specs if spec_name(s) in ALWAYS_ON]
    if not any(spec_name(s) == SEARCH_TOOL for s in active):
        active.append(search_tool_spec(caps))
    for name in sorted(tool_index.loaded):
        spec = tool_index.get(name)
        if spec is not None:
            active.append(spec)
    # Cap the catalog meta tool itself out of the index when reusing an index
    # built from a list that already contained it.
    return active, tool_index


def _clamp_limit(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except Exception:
        return default


def format_search_results(
    query: str,
    results: Sequence[Dict[str, Any]],
    index: Optional[ToolIndex] = None,
) -> str:
    """Tool-result payload: includes full schemas so the next turn can call them."""
    tools = []
    for item in results:
        spec = item.get("spec") or {}
        fn = spec.get("function") or {}
        tools.append({
            "name": item.get("name"),
            "score": item.get("score"),
            "description": fn.get("description") or item.get("summary") or "",
            "parameters": fn.get("parameters") or {},
        })
    payload: Dict[str, Any] = {
        "ok": True,
        "query": query,
        "count": len(tools),
        "tools": tools,
        "hint": (
            "These schemas are now loaded into context. Call any of them by name in your next "
            "tool call. Use tool_search again with a different query if none fit."
        ),
    }
    if index is not None:
        payload["catalog"] = index.stats()
    return json.dumps(payload, ensure_ascii=False)


def status_lines(index: Optional[ToolIndex] = None) -> List[str]:
    stats = index.stats() if index is not None else {
        "indexed": 0, "always_on": len(ALWAYS_ON), "loaded": 0,
        "loaded_names": [], "enabled": deferral_enabled(),
    }
    return [
        f"Deferred tool loading: {'ON' if stats['enabled'] else 'OFF'}",
        f"  always-on: {stats['always_on']} | deferred index: {stats['indexed']} | loaded: {stats['loaded']}",
        "  /tools-search <query> previews matches | /tools-reset clears loaded schemas",
    ]
