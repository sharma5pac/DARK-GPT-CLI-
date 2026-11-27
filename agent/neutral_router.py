"""Universal neutral router for DarkGPT capabilities.

One decision surface over built-in tools, MCP tools, active skills, and verified
local binaries. The router is deliberately conservative:

* ``decide`` is read-only: it never runs a tool, loads a skill, or writes state.
* Execution is opt-in and handled by ``ToolRegistry`` so hooks and permission
  gates apply to the selected call.
* At most one tool is executed per call. There is no recursive chaining.
* Skills are instructions, not actions: auto-execution loads the skill text.
* Local binaries are routed, never silently executed.

The module does not import ``agent.core``; the registry passes its catalog in.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from agent import tool_discovery
from agent import tool_search as tool_search_util

VERSION = 1
SCOPES: Tuple[str, ...] = ("all", "tools", "skills", "mcp", "local")
SELF_NAMES = {"neutral_route", "tool_search"}
# Control-plane tools must never be executed as a side effect of routing.
NO_AUTO_EXECUTE = {
    "ask_question", "submit_and_exit", "delegate", "spawn_agent",
    "neutral_route", "tool_search",
}
WORD_RE = re.compile(r"[a-z0-9_]+")
URL_RE = re.compile(r"https?://[^\s'\"<>]+")
PATH_RE = re.compile(r"(?:(?:~|\.{0,2})/[^\s'\"<>]+)")

# Generic retrieval verbs carry intent but not capability. Leaving them in the
# query makes the lexical index prefer tools whose *names* contain "find" or
# "list" over the tool that actually does the work.
ROUTER_STOPWORDS: Set[str] = {
    "find", "show", "list", "get", "check", "look", "help", "need", "want",
    "please", "can", "could", "would", "should", "how", "what", "which",
    "where", "who", "why", "do", "does", "me", "my", "our", "your",
}

# Small, explicit intent boosts for common objectives where a pure lexical
# match is known to be noisy. Everything else still comes from the catalog
# index, so unknown MCP/local tools remain discoverable.
INTENT_RULES: Tuple[Tuple[re.Pattern[str], Dict[str, float]], ...] = (
    (
        re.compile(r"\b(list|show|enumerate)\b[^.]*\b(files?|director(?:y|ies)|folders?)\b|\b(files?|director(?:y|ies)|folders?)\b[^.]*\b(list|show)\b"),
        {"ls": 30.0},
    ),
    (
        re.compile(r"\b(admin|panels?|dashboard|login page|web page|website)\b"),
        {"browser": 16.0, "http_request": 10.0, "fetch_web_content": 8.0},
    ),
    (
        re.compile(r"\b(screenshots?|capture (?:the )?screen|screen captures?)\b"),
        {"desktop_screenshot": 40.0, "browser": 12.0},
    ),
    (
        re.compile(r"\b(nmap|port scan|service scan|open ports?)\b"),
        {"autonomous_route": 28.0, "discover_tools": 10.0},
    ),
    (
        re.compile(r"\b(entropy|prng|token bias|chi[- ]?square|timing side|attack graph)\b"),
        {"math_attack": 26.0},
    ),
    (
        re.compile(r"\b(riemann|grh|zeta|l-function|proof audit)\b"),
        {"math_attack": 22.0},
    ),
    (
        re.compile(r"\b(scan|recon|enumerat(?:e|ion))\b"),
        {"autonomous_route": 16.0, "discover_tools": 10.0},
    ),
)


def _tokens(text: str) -> Set[str]:
    words = {
        word
        for word in WORD_RE.findall((text or "").lower())
        if word not in tool_search_util.STOPWORDS and len(word) > 1
    }
    expanded = set(words)
    for word in tuple(words):
        expanded.update(tool_search_util.SYNONYMS.get(word, ()))
    return expanded


def _clean_query(text: str) -> str:
    """Remove generic retrieval verbs before lexical catalog scoring."""
    words = [
        word
        for word in WORD_RE.findall((text or "").lower())
        if word not in ROUTER_STOPWORDS
    ]
    return " ".join(words)


def _intent_boosts(goal: str) -> Dict[str, Tuple[float, str]]:
    lowered = (goal or "").lower()
    boosts: Dict[str, Tuple[float, str]] = {}
    for pattern, names in INTENT_RULES:
        match = pattern.search(lowered)
        if not match:
            continue
        label = match.group(0).strip()[:80]
        for name, weight in names.items():
            old = boosts.get(name)
            if old is None or weight > old[0]:
                boosts[name] = (weight, f"intent: {label}")
    return boosts


def _boost_matches(name: str, boosts: Dict[str, Tuple[float, str]]) -> Tuple[float, str]:
    for key, (weight, reason) in boosts.items():
        if name == key or name.endswith("__" + key):
            return weight, reason
    return 0.0, ""


def _spec_name(spec: Dict[str, Any]) -> str:
    try:
        return str(spec["function"]["name"])
    except Exception:
        return ""


def _spec_parameters(spec: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = spec["function"].get("parameters") or {}
        return params if isinstance(params, dict) else {}
    except Exception:
        return {}


def _spec_required(spec: Dict[str, Any]) -> List[str]:
    required = _spec_parameters(spec).get("required") or []
    return [str(item) for item in required if str(item).strip()]


def _spec_properties(spec: Dict[str, Any]) -> Dict[str, Any]:
    props = _spec_parameters(spec).get("properties") or {}
    return props if isinstance(props, dict) else {}


def _kind_for_tool(name: str) -> str:
    return "mcp" if name.startswith("mcp__") else "tool"


def _server_for_tool(name: str) -> str:
    if not name.startswith("mcp__"):
        return ""
    parts = name.split("__", 2)
    return parts[1] if len(parts) == 3 else ""


def _active_skills(skills: Any) -> List[Any]:
    if skills is None:
        return []
    if hasattr(skills, "active") and callable(skills.active):
        try:
            return list(skills.active())
        except Exception:
            return []
    return [skill for skill in skills if skill is not None]


def _rank_skills(goal: str, skills: Sequence[Any], limit: int) -> List[Dict[str, Any]]:
    lowered = goal.lower()
    query = _tokens(goal)
    ranked: List[Dict[str, Any]] = []
    for skill in skills:
        name = str(getattr(skill, "name", "") or "").strip()
        if not name:
            continue
        description = str(getattr(skill, "description", "") or "")
        triggers = [str(item) for item in (getattr(skill, "triggers", []) or [])]
        preferred_tools = [str(item) for item in (getattr(skill, "tools", []) or [])]
        text = " ".join([name, description, *triggers, *preferred_tools])
        score = float(len(query & _tokens(text)))
        reasons: List[str] = []
        if name.lower() in lowered:
            score += 20.0
            reasons.append(f"explicit skill name: {name}")
        for trigger in triggers:
            needle = trigger.strip().lower()
            if needle and needle in lowered:
                score += 8.0
                reasons.append(f"trigger: {trigger}")
        for tool in preferred_tools:
            needle = tool.strip().lower()
            if needle and needle in lowered:
                score += 4.0
                reasons.append(f"preferred tool: {tool}")
        if score <= 0:
            continue
        ranked.append({
            "kind": "skill",
            "name": name,
            "score": round(score, 3),
            "reason": "; ".join(dict.fromkeys(reasons)) or "description/token match",
            "summary": description[:300],
            "triggers": triggers[:12],
            "preferred_tools": preferred_tools[:12],
            "path": str(getattr(getattr(skill, "path", None), "parent", "") or ""),
        })
    ranked.sort(key=lambda item: (-item["score"], item["name"].lower()))
    return ranked[: max(1, limit)]


def _mentions(name: str, text: str) -> bool:
    if not name:
        return False
    return bool(
        re.search(
            r"(?<![a-z0-9_])" + re.escape(name.lower()) + r"(?![a-z0-9_])",
            (text or "").lower(),
        )
    )


def _candidate(name: str, spec: Dict[str, Any], score: float, reason: str) -> Dict[str, Any]:
    return {
        "kind": _kind_for_tool(name),
        "name": name,
        "score": round(score, 3),
        "reason": reason,
        "summary": tool_search_util.spec_description(spec)[:300],
        "required": _spec_required(spec),
        "properties": list(_spec_properties(spec))[:24],
        "server": _server_for_tool(name),
        "spec": spec,
    }


def _rank_registry_tools(
    goal: str,
    specs: Sequence[Dict[str, Any]],
    limit: int,
    *,
    mcp_only: bool = False,
) -> List[Dict[str, Any]]:
    index = tool_search_util.ToolIndex(specs, always_on=())
    query = _clean_query(goal) or goal
    results = index.search(query, limit=12, skip_loaded=False)
    boosts = _intent_boosts(goal)
    lowered = goal.lower()
    by_name: Dict[str, Dict[str, Any]] = {}

    for result in results:
        name = str(result.get("name") or "")
        if not name or name in SELF_NAMES:
            continue
        spec = index.get(name)
        if not spec:
            continue
        score = float(result.get("score") or 0.0)
        reasons = ["catalog token match"]
        if _mentions(name, lowered):
            score += 50.0
            reasons.insert(0, f"explicit tool name: {name}")
        boost, boost_reason = _boost_matches(name, boosts)
        if boost:
            score += boost
            reasons.append(boost_reason)
        by_name[name] = _candidate(name, spec, score, "; ".join(reasons))

    # Add boosted tools that lexical scoring did not return, so a known intent
    # is not lost just because the catalog description uses different words.
    for key in boosts:
        if any(name == key or name.endswith("__" + key) for name in by_name):
            continue
        for spec in specs:
            name = _spec_name(spec)
            if not name or name in SELF_NAMES:
                continue
            if not (name == key or name.endswith("__" + key)):
                continue
            boost, boost_reason = _boost_matches(name, boosts)
            by_name[name] = _candidate(name, spec, boost, boost_reason)
            break

    ranked = [
        item for item in by_name.values()
        if not mcp_only or item["kind"] == "mcp"
    ]
    ranked.sort(key=lambda item: (-item["score"], item["name"].lower()))
    return ranked[: max(1, limit)]


def _rank_local(
    goal: str,
    specs: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    try:
        found = tool_discovery.discover_tools()
        ranked = tool_discovery.rank_tools(
            _clean_query(goal) or goal,
            found,
            mcp_tool_names=[_spec_name(spec) for spec in specs],
            limit=max(1, limit),
        )
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for item in ranked:
        reason = str(item.get("reason") or "")
        # A Kala/MCP wrapper mapping alone is not evidence that the local
        # binary matches this goal; require a real name/tag/category hit.
        if float(item.get("score") or 0.0) <= 35.0 and reason.startswith("Kala/MCP wrapper"):
            continue
        out.append({
            "kind": "local",
            "name": str(item.get("name") or ""),
            "score": float(item.get("score") or 0.0),
            "reason": str(item.get("reason") or "local capability match")[:300],
            "summary": str(item.get("category") or "")[:200],
            "path": str(item.get("path") or ""),
            "mcp_tool": str(item.get("mcp_tool") or ""),
            "gui": bool(item.get("gui")),
        })
    return out[: max(1, limit)]


def _suggest_args(
    goal: str,
    spec: Dict[str, Any],
    target: str = "",
    name: str = "",
) -> Dict[str, Any]:
    props = _spec_properties(spec)
    suggested: Dict[str, Any] = {}
    url = URL_RE.search(goal or "")
    path = PATH_RE.search(goal or "")
    if "url" in props and url:
        suggested["url"] = url.group(0).rstrip(").,;")
    if "path" in props and path:
        suggested["path"] = path.group(0).rstrip(").,;")
    if "query" in props:
        suggested["query"] = goal
    if "target" in props and target:
        suggested["target"] = target
    if "question" in props:
        suggested["question"] = goal
    if "objective" in props:
        suggested["objective"] = goal
    if "task" in props:
        suggested["task"] = goal
    if "prompt" in props:
        suggested["prompt"] = goal
    if "files" in props and path:
        suggested["files"] = [path.group(0).rstrip(").,;")]
    if name == "math_attack" and "action" in props:
        # Let math_attack make its own math decision. "auto" only computes when
        # that decision is confident; otherwise it asks one question.
        suggested["action"] = "auto"
        if "goal" in props:
            suggested["goal"] = goal
    return suggested


def _missing_args(spec: Dict[str, Any], args: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for name in _spec_required(spec):
        value = args.get(name)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(name)
    return missing


def _choose_primary(
    goal: str,
    tools: Sequence[Dict[str, Any]],
    skills: Sequence[Dict[str, Any]],
    local: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    lowered = goal.lower()
    tool = tools[0] if tools else None
    skill = skills[0] if skills else None
    explicit_tool = bool(tool and _mentions(str(tool.get("name") or ""), lowered))
    explicit_skill = bool(skill and _mentions(str(skill.get("name") or ""), lowered))
    if explicit_skill and not explicit_tool:
        return skill
    if explicit_tool:
        return tool
    # A directly executable tool wins over an instructional skill unless the
    # tool match is only lexical noise and the skill match is strong.
    if tool and (not skill or tool["score"] >= 2.0 or skill["score"] < 8.0):
        return tool
    if skill:
        return skill
    if local:
        return local[0]
    return None


def _public(item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not item:
        return item
    out = dict(item)
    out.pop("spec", None)
    return out


def _confidence(primary: Optional[Dict[str, Any]], goal: str) -> str:
    if not primary:
        return "low"
    name = str(primary.get("name") or "").lower()
    lowered = goal.lower()
    if name and name in lowered:
        return "high"
    score = float(primary.get("score") or 0.0)
    if score >= 8.0:
        return "high"
    if score >= 2.0:
        return "medium"
    return "low"


def decide(
    goal: str,
    specs: Sequence[Dict[str, Any]],
    skills: Any = None,
    *,
    provided_args: Optional[Dict[str, Any]] = None,
    scope: str = "all",
    limit: int = 6,
    target: str = "",
) -> Dict[str, Any]:
    """Return a read-only route decision over every capability catalog."""
    goal = str(goal or "").strip()
    scope = str(scope or "all").strip().lower()
    if scope not in SCOPES:
        return {
            "ok": False,
            "action": "decide",
            "error": f"unknown scope: {scope}",
            "supported": list(SCOPES),
        }
    limit = max(1, min(12, int(limit or 6)))
    provided = provided_args if isinstance(provided_args, dict) else {}
    active_skills = _active_skills(skills)

    tools: List[Dict[str, Any]] = []
    if scope in ("all", "tools", "mcp"):
        tools = _rank_registry_tools(
            goal, specs, limit, mcp_only=scope == "mcp"
        )
    ranked_skills = _rank_skills(goal, active_skills, limit) if scope in ("all", "skills") else []
    local = _rank_local(goal, specs, limit) if scope in ("all", "local") else []

    primary = _choose_primary(goal, tools, ranked_skills, local)
    confidence = _confidence(primary, goal)
    catalog = {
        "tools": len([
            s for s in specs
            if _spec_name(s)
            and _spec_name(s) not in SELF_NAMES
            and not _spec_name(s).startswith("mcp__")
        ]),
        "mcp": len([s for s in specs if _spec_name(s).startswith("mcp__")]),
        "skills": len(active_skills),
        "local": len(local),
    }
    decision: Dict[str, Any] = {
        "ok": True,
        "action": "decide",
        "version": VERSION,
        "goal": goal,
        "scope": scope,
        "catalog": catalog,
        "primary": primary,
        "tool": tools[0] if tools else None,
        "skill": ranked_skills[0] if ranked_skills else None,
        "local": local[0] if local else None,
        "alternatives": [
            *tools[1:4],
            *ranked_skills[1:3],
            *local[1:3],
        ],
        "confidence": confidence,
        "decided": bool(primary),
        "executed": False,
        "next_action": "ask",
    }
    if not primary:
        decision["question"] = (
            "What concrete target, file, URL, or action should I route?"
            if not goal
            else "No capability matched. Add a target/action or name the tool, skill, or MCP server."
        )
        decision["options"] = ["describe the action", "provide a path or URL", "name a tool or skill"]
        decision["text"] = format_decision(decision)
        return decision

    kind = str(primary.get("kind") or "")
    if kind in ("tool", "mcp"):
        spec = primary.get("spec") or {}
        suggested = _suggest_args(goal, spec, target, str(primary.get("name") or ""))
        merged = dict(suggested)
        merged.update(provided)
        missing = _missing_args(spec, merged)
        decision["suggested_args"] = suggested
        decision["args"] = merged
        decision["missing"] = missing
        if str(primary.get("name") or "") in NO_AUTO_EXECUTE:
            decision["can_execute"] = False
            decision["next_action"] = "blocked"
            decision["note"] = "control-plane tools are not auto-executed by the router"
        else:
            decision["can_execute"] = not missing
            decision["next_action"] = "call_tool" if not missing else "provide_args"
    elif kind == "skill":
        decision["args"] = {}
        decision["missing"] = []
        decision["can_execute"] = True
        decision["next_action"] = "load_skill"
    else:
        decision["args"] = dict(provided)
        decision["missing"] = []
        decision["can_execute"] = False
        decision["next_action"] = "local_route"
        decision["note"] = (
            "Local binaries are routed but not auto-executed. "
            "Use autonomous_route or provide an explicit command."
        )
    for key in ("primary", "tool", "skill", "local"):
        decision[key] = _public(decision.get(key))
    decision["alternatives"] = [
        _public(item) for item in decision.get("alternatives") or [] if item
    ]
    decision["text"] = format_decision(decision)
    return decision


def format_decision(data: Dict[str, Any], *, executed: bool = False) -> str:
    catalog = data.get("catalog") or {}
    lines = [
        "NEUTRAL ROUTER",
        f"  goal: {str(data.get('goal') or '-')[:240]}",
        f"  scope: {data.get('scope')} | catalog: {catalog.get('tools', 0)} tools, "
        f"{catalog.get('mcp', 0)} MCP, {catalog.get('skills', 0)} skills, "
        f"{catalog.get('local', 0)} local matches",
    ]
    primary = data.get("primary")
    if not primary:
        lines.extend([
            "  decision: NOT DECIDED",
            f"  question: {data.get('question') or 'What should I route?'}",
            f"  options: {', '.join(data.get('options') or [])}",
        ])
        return "\n".join(lines)
    lines.extend([
        f"  primary: {primary.get('kind')}:{primary.get('name')} "
        f"score={primary.get('score')} confidence={data.get('confidence')}",
        f"  reason: {primary.get('reason') or '-'}",
        f"  next: {data.get('next_action')}",
    ])
    skill = data.get("skill")
    if skill and skill is not primary:
        lines.append(f"  skill to load: {skill.get('name')} ({skill.get('reason')})")
    tool = data.get("tool")
    if tool and tool is not primary:
        lines.append(f"  tool candidate: {tool.get('name')} ({tool.get('reason')})")
    if data.get("missing"):
        lines.append(f"  missing args: {', '.join(data.get('missing') or [])}")
    if data.get("args"):
        lines.append(
            "  args: "
            + json.dumps(data.get("args") or {}, ensure_ascii=False, default=str)[:700]
        )
    if executed:
        lines.append(
            f"  executed: yes | tool={data.get('executed_tool') or '-'} "
            f"| ok={data.get('executed_ok')}"
        )
        preview = str(data.get("result_preview") or "").strip()
        if preview:
            lines.append(f"  result: {preview[:700]}")
    else:
        lines.append("  executed: no | use execute=true to run the selected route")
    blocked = str(data.get("execute_blocked") or "").strip()
    if blocked:
        lines.append(f"  blocked: {blocked[:400]}")
    alternatives = data.get("alternatives") or []
    if alternatives:
        labels = [f"{item.get('kind')}:{item.get('name')}" for item in alternatives[:6]]
        lines.append(f"  alternatives: {', '.join(labels)}")
    return "\n".join(lines)
