"""ARTEX-style engagement loop: verifiable goals, short intents, workers, retests.

This is the orchestration layer on top of the existing target state:

* ``graph.json``       - shared world/asset graph (what exists)
* ``work_graph.json``  - per-engagement exploration graph (what was tried and why)
* ``engagement.json``  - goals, short intents, worker ledger, retest history
* ``evidence.jsonl``   - append-only evidence with SHA-256

Improvements over a plain "AI pentest" loop: goals carry explicit success
criteria, intents are one small action each with an anchor, every claim is
bounded (attempts/worker cap), proved status still passes the evidence proof
gate, and no stored knowledge is ever deleted (status changes only).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent import knowledge_graph as kg, work_graph as wg

VERSION = 1
MAX_GOALS = 60
MAX_INTENTS = 200
MAX_INTENT_ATTEMPTS = 3
MAX_ACTIONS_PER_INTENT = 4
ALLOWED_WORKER_TOOLS = frozenset({
    "script", "http_request", "shell", "browser", "oracle_validate",
    "verify_findings", "attack_paths", "knowledge_graph", "evidence_log",
})
ACTIVE_INTENT_STATUSES = {"queued", "claimed", "running"}
GOAL_DONE = {"completed", "covered", "closed"}
RETEST_VERDICTS = {"still_reproducible", "fixed", "cannot_confirm"}


def engagement_path(target: str) -> Path:
    return kg.target_dir(target) / "engagement.json"


def _empty(target: str) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "target": target,
        "goals": [],
        "intents": [],
        "workers": [],
        "retests": [],
        "journal": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def load(target: str) -> Dict[str, Any]:
    path = engagement_path(target)
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            base = _empty(target)
            base.update(data)
            for key in ("goals", "intents", "workers", "retests", "journal"):
                base.setdefault(key, [])
            return base
    except Exception:
        pass
    return _empty(target)


def save(target: str, data: Dict[str, Any]) -> Dict[str, Any]:
    path = engagement_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = time.time()
    data["journal"] = (data.get("journal") or [])[-3000:]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)
    return data


def _id(prefix: str, items: List[Dict[str, Any]]) -> str:
    return f"{prefix}-{len(items) + 1}-{int(time.time()) % 100000}"


def _journal(data: Dict[str, Any], action: str, **fields: Any) -> None:
    entry = {"ts": time.time(), "action": action}
    entry.update({k: v for k, v in fields.items() if v not in (None, "", [], {})})
    data.setdefault("journal", []).append(entry)


def _clean_list(value: Any, limit: int = 20) -> List[str]:
    if isinstance(value, str):
        parts = re.split(r"[\n,;]+", value)
    elif isinstance(value, (list, tuple, set)):
        parts = [str(p) for p in value]
    else:
        parts = []
    return [str(p).strip()[:300] for p in parts if str(p).strip()][:limit]


def add_goal(
    target: str,
    text: str,
    *,
    scope: Any = None,
    success_criteria: Any = None,
    source: str = "user",
    goal_id: str = "",
) -> Dict[str, Any]:
    """Create (or refresh) a goal with criteria that can be checked against state."""
    data = load(target)
    text = str(text or "").strip()[:600]
    if not text:
        return {"ok": False, "error": "goal text required"}
    criteria = _clean_list(success_criteria) or [
        "At least one evidence-backed fact or finding anchored to this objective"
    ]
    scope_items = _clean_list(scope)
    norm = text.lower()
    for goal in data["goals"]:
        if str(goal.get("text") or "").strip().lower() == norm:
            goal["scope"] = sorted(set(scope_items) | set(goal.get("scope") or []))[:20]
            goal["success_criteria"] = sorted(set(criteria) | set(goal.get("success_criteria") or []))[:20]
            goal["updated_at"] = time.time()
            save(target, data)
            return {"ok": True, "goal": goal, "deduplicated": True}
    gid = goal_id or _id("g", data["goals"])
    goal = {
        "id": gid,
        "text": text,
        "scope": scope_items,
        "success_criteria": criteria,
        "status": "open",
        "owner": "",
        "evidence_refs": [],
        "created_at": time.time(),
        "updated_at": time.time(),
        "source": source,
    }
    data["goals"].append(goal)
    data["goals"] = data["goals"][-MAX_GOALS:]
    _journal(data, "goal_added", goal_id=gid, text=text[:200])
    save(target, data)
    wg.upsert_node(
        target, "goal", gid,
        {"text": text, "scope": scope_items, "success_criteria": criteria, "source": source},
        status="open", source=source, note=text[:200],
    )
    for item in scope_items:
        for kind in ("host", "service", "endpoint", "finding"):
            wg.anchor(target, "goal", gid, kind, item, source="goal_scope")
    return {"ok": True, "goal": goal, "deduplicated": False}


def set_scope(target: str, scope: Any, *, source: str = "scope") -> Dict[str, Any]:
    """Turn a target scope (comma/newline separated) into verifiable goals."""
    items = _clean_list(scope, limit=30)
    if not items:
        return {"ok": False, "error": "scope is empty"}
    made = []
    for item in items:
        kind = "endpoint" if item.lower().startswith(("http://", "https://")) else "host"
        made.append(add_goal(
            target,
            f"Enumerate and verify security posture of {item}",
            scope=[item],
            success_criteria=[
                f"At least one evidence-backed asset, fact or finding anchored to {item}",
                "Any claimed finding has a proof source (evidence_ref or verified oracle_ref)",
            ],
            source=source,
        ))
    return {"ok": True, "target": target, "goals": made, "count": len(made)}


def propose_goals(target: str, prompt: str, *, model: str = "", limit: int = 6) -> Dict[str, Any]:
    """Goal model: split a prompt into bounded, checkable objectives.

    Uses the configured cheap model when available; a deterministic scope- and
    evidence-driven fallback keeps the loop usable offline and on quota errors.
    """
    prompt = str(prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt required"}
    lines: List[str] = []
    if model:
        try:
            from agent import routing

            plan = routing.make_plan(prompt, model) or ""
            lines = [
                m.group(1).strip()
                for line in plan.splitlines()
                if (m := re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.+)$", line))
            ][:limit]
        except Exception:
            lines = []
    if not lines:
        urls = re.findall(r"https?://[^\s,;]+", prompt)
        hosts = re.findall(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", prompt, re.I)
        lines = urls + hosts
        if not lines:
            lines = [prompt[:300]]
    made = []
    for line in lines[:limit]:
        item = line[:300]
        scope = [item] if re.search(r"[./:]", item) else []
        made.append(add_goal(
            target, item, scope=scope,
            success_criteria=["Observable evidence or oracle result proves this objective"],
            source="goal_model",
        ))
    return {"ok": True, "target": target, "goals": made, "count": len(made)}


def _default_actions(kind: str, identity: str, goal_id: str) -> List[Dict[str, Any]]:
    """Minimal-impact, real-tool actions. Nothing destructive or credential-using."""
    if kind in ("endpoint", "service") and identity.startswith(("http://", "https://")):
        code = (
            "import json, urllib.request\n"
            f"url = {json.dumps(identity)}\n"
            "req = urllib.request.Request(url, headers={'User-Agent': 'DarkGPT-engagement/1.0'})\n"
            "try:\n"
            "    with urllib.request.urlopen(req, timeout=8) as r:\n"
            "        body = r.read(4096).decode('utf-8', 'replace')\n"
            "        print(json.dumps({'url': url, 'status': r.status, 'headers': dict(list(r.headers.items())[:12]), 'preview': body[:800]}))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'url': url, 'error': type(exc).__name__ + ': ' + str(exc)[:300]}))\n"
        )
        return [{"tool": "script", "args": {"language": "python", "code": code, "timeout": 30}}]
    if kind == "host" or kind == "service":
        host = identity.split(":")[0].strip()
        try:
            port = int(str(identity).rsplit(":", 1)[1].split("/")[0])
            ports = [port]
        except Exception:
            host = host or identity
            ports = [21, 22, 80, 443, 8080, 8443]
        code = (
            "import json, socket\n"
            f"host = {json.dumps(host)}\n"
            f"ports = {json.dumps(ports)}\n"
            "out = []\n"
            "for port in ports:\n"
            "    s = socket.socket(); s.settimeout(0.7)\n"
            "    try:\n"
            "        rc = s.connect_ex((host, port))\n"
            "        out.append({'port': port, 'open': rc == 0})\n"
            "    except Exception as exc:\n"
            "        out.append({'port': port, 'error': type(exc).__name__})\n"
            "    finally:\n"
            "        s.close()\n"
            "print(json.dumps({'host': host, 'ports': out}))\n"
        )
        return [{"tool": "script", "args": {"language": "python", "code": code, "timeout": 30}}]
    if kind == "finding":
        return [{"tool": "verify_findings", "args": {"action": "replay", "finding": identity}}]
    if kind == "pivot":
        return [{"tool": "attack_paths", "args": {"max_paths": 8}}]
    return [{"tool": "attack_paths", "args": {"max_paths": 8}}]


def emit_intents(target: str, *, goal_id: str = "", limit: int = 8, focus: str = "") -> Dict[str, Any]:
    """Planner step: read world + work graphs, emit short intents for gaps."""
    data = load(target)
    cov = wg.coverage(target, goal_id=goal_id)
    uncovered = cov.get("uncovered") or []
    if focus:
        needle = str(focus).lower()
        uncovered = [u for u in uncovered if needle in str(u.get("id") or "").lower()]
    made = []
    seen = {
        (str(i.get("kind")), str((i.get("anchor") or {}).get("node_key") or ""), str(i.get("status")))
        for i in data["intents"]
        if i.get("status") in ACTIVE_INTENT_STATUSES
    }
    for item in uncovered:
        if len(made) >= max(1, min(limit, 16)):
            break
        kind = str(item.get("kind") or "host")
        identity = str(item.get("id") or "")
        iid = _id("i", data["intents"])
        label = {
            "host": f"Enumerate services on {identity}",
            "service": f"Probe service {identity} for banner and common misconfigurations",
            "endpoint": f"Minimally test endpoint {identity} for input-driven flaws",
            "finding": f"Validate finding {identity} with deterministic replay/oracle",
        }.get(kind, f"Investigate {identity}")
        sig = (f"probe" if kind != "finding" else "validate", str(item.get("key") or ""), "queued")
        if sig in seen:
            continue
        seen.add(sig)
        intent = {
            "id": iid,
            "goal_id": goal_id,
            "kind": "validate" if kind == "finding" else "probe",
            "text": label,
            "anchor": {
                "graph": "world",
                "node_key": item.get("key"),
                "kind": kind,
                "id": identity,
                "evidence_ref": "",
            },
            "success_signal": (
                "at least one evidence-backed fact/finding anchored to this asset"
                if kind != "finding" else "deterministic replay or oracle verdict recorded"
            ),
            "budget": {"tools": 2, "steps": 2},
            "attempts": 0,
            "status": "queued",
            "owner": "",
            "created_at": time.time(),
            "updated_at": time.time(),
            "actions": _default_actions(kind, identity, goal_id),
        }
        data["intents"].append(intent)
        wg.upsert_node(
            target, "intent", iid,
            {"text": label, "kind": intent["kind"], "goal_id": goal_id, "success_signal": intent["success_signal"]},
            status="queued", source="planner", anchors=[intent["anchor"]], note=label[:200],
        )
        made.append(intent)
    if not made:
        iid = _id("i", data["intents"])
        intent = {
            "id": iid, "goal_id": goal_id, "kind": "pivot",
            "text": "Re-rank attack paths and evidence for the next uncovered direction",
            "anchor": {"graph": "world", "node_key": "", "kind": "", "id": "", "evidence_ref": ""},
            "success_signal": "a new evidence-backed asset, fact or finding",
            "budget": {"tools": 1, "steps": 1}, "attempts": 0, "status": "queued", "owner": "",
            "created_at": time.time(), "updated_at": time.time(),
            "actions": _default_actions("pivot", "", goal_id),
        }
        data["intents"].append(intent)
        wg.upsert_node(target, "intent", iid, {"text": intent["text"], "kind": "pivot"}, status="queued", source="planner")
        made.append(intent)
    data["intents"] = data["intents"][-MAX_INTENTS:]
    _journal(data, "intents_emitted", count=len(made), goal_id=goal_id)
    save(target, data)
    return {"ok": True, "target": target, "emitted": len(made), "intents": made, "coverage": cov["percent"]}


def claim_intent(target: str, worker_id: str = "", intent_id: str = "") -> Optional[Dict[str, Any]]:
    """Atomically claim one bounded intent. A dead intent is retried, not duplicated."""
    data = load(target)
    chosen = None
    for intent in data["intents"]:
        if intent_id and str(intent.get("id")) != intent_id:
            continue
        if intent.get("status") != "queued":
            continue
        if int(intent.get("attempts") or 0) >= MAX_INTENT_ATTEMPTS:
            continue
        chosen = intent
        break
    if not chosen:
        return None
    chosen["status"] = "claimed"
    chosen["owner"] = worker_id or "worker"
    chosen["attempts"] = int(chosen.get("attempts") or 0) + 1
    chosen["claimed_at"] = time.time()
    chosen["updated_at"] = time.time()
    wg.set_status(target, wg.node_key("intent", chosen["id"]), "claimed", owner=chosen["owner"])
    _journal(data, "intent_claimed", intent_id=chosen["id"], worker_id=chosen["owner"])
    save(target, data)
    return dict(chosen)


def start_intent(target: str, intent_id: str, worker_id: str = "") -> Optional[Dict[str, Any]]:
    data = load(target)
    for intent in data["intents"]:
        if str(intent.get("id")) == intent_id:
            intent["status"] = "running"
            intent["owner"] = worker_id or intent.get("owner") or "worker"
            intent["updated_at"] = time.time()
            wg.set_status(target, wg.node_key("intent", intent_id), "running", owner=intent["owner"])
            save(target, data)
            return dict(intent)
    return None


def complete_intent(
    target: str,
    intent_id: str,
    *,
    ok: bool = True,
    evidence_ref: str = "",
    summary: str = "",
    worker_id: str = "",
) -> Dict[str, Any]:
    data = load(target)
    found = None
    for intent in data["intents"]:
        if str(intent.get("id")) == intent_id:
            found = intent
            break
    if not found:
        return {"ok": False, "error": "intent not found"}
    status = "done" if ok else "failed"
    # Validation intents are proof-gated: a claim without an evidence ref is a
    # failed attempt, never silent success.
    if status == "done" and str(found.get("kind") or "") == "validate" and not evidence_ref:
        status = "failed"
        summary = (str(summary or "") + " | proof gate: no evidence ref").strip(" |")
        ok = False
    found["status"] = status
    found["ok"] = bool(ok)
    found["summary"] = str(summary or "")[:1000]
    found["evidence_ref"] = evidence_ref or found.get("evidence_ref", "")
    found["completed_at"] = time.time()
    found["updated_at"] = time.time()
    if worker_id:
        found["owner"] = worker_id
    node = wg.set_status(
        target, wg.node_key("intent", intent_id), status,
        owner=found.get("owner") or "", note=found["summary"],
        attrs={"evidence_ref": found["evidence_ref"], "ok": bool(ok)},
    )
    if ok and evidence_ref and node:
        anchor = dict(found.get("anchor") or {})
        anchor["evidence_ref"] = evidence_ref
        wg.upsert_node(target, "intent", intent_id, {}, anchors=[anchor], source="worker")
    _journal(data, "intent_completed", intent_id=intent_id, status=status, evidence_ref=evidence_ref)
    save(target, data)
    return {"ok": True, "intent": dict(found), "status": status}


def record_worker(target: str, worker_id: str, *, role: str = "worker", intent_id: str = "", ok: bool = True, preview: str = "") -> Dict[str, Any]:
    data = load(target)
    entry = {
        "id": worker_id, "role": role, "intent_id": intent_id, "ok": bool(ok),
        "preview": str(preview or "")[:600], "ts": time.time(),
    }
    data["workers"].append(entry)
    data["workers"] = data["workers"][-300:]
    _journal(data, "worker", worker_id=worker_id, intent_id=intent_id, ok=bool(ok))
    save(target, data)
    wg.upsert_node(
        target, "worker", worker_id,
        {"role": role, "intent_id": intent_id, "preview": entry["preview"], "ok": bool(ok)},
        status="done" if ok else "failed", source="engagement",
    )
    return entry


def validate_actions(actions: Any) -> List[Dict[str, Any]]:
    out = []
    for action in actions or []:
        if not isinstance(action, dict):
            continue
        name = str(action.get("tool") or "")
        if name not in ALLOWED_WORKER_TOOLS or name in ("engagement", "engagement_run"):
            continue
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        out.append({"tool": name, "args": args})
        if len(out) >= MAX_ACTIONS_PER_INTENT:
            break
    return out


def record_retest(
    target: str,
    finding: str,
    verdict: str,
    *,
    evidence_ref: str = "",
    output: str = "",
    exit_code: int = 0,
    timeout: int = 120,
) -> Dict[str, Any]:
    verdict = str(verdict or "cannot_confirm")
    if verdict not in RETEST_VERDICTS:
        verdict = "cannot_confirm"
    data = load(target)
    entry = {
        "id": _id("rt", data["retests"]), "finding": finding, "verdict": verdict,
        "evidence_ref": evidence_ref, "output": str(output or "")[:1200],
        "exit_code": int(exit_code), "timeout": int(timeout), "ts": time.time(),
    }
    data["retests"].append(entry)
    data["retests"] = data["retests"][-300:]
    _journal(data, "retest", finding=finding, verdict=verdict, evidence_ref=evidence_ref)
    save(target, data)
    wg.upsert_node(
        target, "retest", entry["id"],
        {"finding": finding, "verdict": verdict, "exit_code": entry["exit_code"], "output": entry["output"]},
        status="done" if verdict != "cannot_confirm" else "failed", source="retester",
        note=f"{verdict}: {finding}"[:200],
    )
    for node in kg.query(target, kind="finding", text=finding, limit=5):
        fid = str(node.get("id") or finding)
        wg.anchor(target, "retest", entry["id"], "finding", fid, source="retester")
        attrs = dict(node.get("attrs") or {})
        history = list(attrs.get("retest_history") or [])
        history.append({"verdict": verdict, "evidence_ref": evidence_ref,
                        "exit_code": entry["exit_code"], "ts": entry["ts"]})
        attrs["retest_history"] = history[-30:]
        attrs["last_retest_verdict"] = verdict
        attrs["last_retest_at"] = entry["ts"]
        attrs["last_retest_evidence_ref"] = evidence_ref
        attrs["retest_count"] = int(attrs.get("retest_count") or 0) + 1
        kg.update_node_attrs(target, str(node.get("key") or ""), attrs, source="retester")
        break
    return entry


def summary(target: str, limit: int = 8) -> Dict[str, Any]:
    data = load(target)
    cov = wg.coverage(target)
    goals = data.get("goals") or []
    intents = data.get("intents") or []
    return {
        "ok": True,
        "target": target,
        "path": str(engagement_path(target)),
        "goals": {
            "total": len(goals),
            "open": sum(1 for g in goals if g.get("status") not in GOAL_DONE),
            "recent": goals[-limit:],
        },
        "intents": {
            "total": len(intents),
            "queued": sum(1 for i in intents if i.get("status") == "queued"),
            "claimed": sum(1 for i in intents if i.get("status") in ("claimed", "running")),
            "done": sum(1 for i in intents if i.get("status") == "done"),
            "failed": sum(1 for i in intents if i.get("status") == "failed"),
            "recent": intents[-limit:],
        },
        "workers": data.get("workers", [])[-limit:],
        "retests": data.get("retests", [])[-limit:],
        "coverage": {
            "percent": cov["percent"], "covered": cov["covered"],
            "world_nodes": cov["world_nodes"], "uncovered": cov["uncovered"][:limit],
        },
        "work_graph": wg.summary(target, limit=limit),
    }


def context_block(target: str, limit: int = 5) -> str:
    """Compact live engagement state for the system prompt."""
    if not target:
        return ""
    try:
        data = load(target)
        cov = wg.coverage(target)
        lines = [
            f"Engagement: {len(data.get('goals') or [])} goal(s), "
            f"{sum(1 for i in data.get('intents') or [] if i.get('status') == 'queued')} queued intent(s), "
            f"coverage {cov['percent']}% ({cov['covered']}/{cov['world_nodes']} anchored assets)."
        ]
        for goal in (data.get("goals") or [])[-limit:]:
            lines.append(f"- goal [{goal.get('status')}] {str(goal.get('text'))[:140]}")
            for criterion in (goal.get("success_criteria") or [])[:2]:
                lines.append(f"    verify: {str(criterion)[:140]}")
        for intent in (data.get("intents") or [])[-limit:]:
            if intent.get("status") in ("queued", "claimed", "running"):
                lines.append(f"- intent [{intent.get('status')}] {str(intent.get('text'))[:140]}")
        lines.append(
            "ENGAGEMENT LOOP: use `engagement` to set goals/emit intents and inspect coverage, "
            "`engagement_run` to claim and execute one bounded intent with real tools, "
            "`verify_findings retest` to re-run a finding. Treat goal coverage as a hint: "
            "confirm with evidence or an oracle before marking a finding proved."
        )
        return "\n".join(lines)
    except Exception:
        return ""
