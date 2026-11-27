"""Math-for-attacks learning and bounded analysis runtime.

This module makes the math skill operational instead of decorative:

* choose a math route for an attack-analysis question (crypto/PRNG/stats/graph/timing);
* run deterministic, size-bounded computations in ``math_worker.py``;
* verify claims with a residual gate;
* persist reusable lessons with evidence references and mastery progress;
* keep the Riemann-hypothesis status precise and reject "RH breaks RSA" claims.

State is append-only: lessons/events are never deleted, and prior evidence is
never overwritten. Deactivated skills never affect this tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent import evidence as evidence_util
from utils import appconfig

STATE_PATH = Path.home() / ".config" / "darkgpt" / "math_attack_state.json"
WORKER_PATH = Path(__file__).resolve().with_name("math_worker.py")
DEFAULT_PYTHON = "/usr/bin/python3"
MAX_WORKER_SECONDS = int(appconfig.get("math_worker_timeout", 60) or 60)
MAX_ANSWER_CHARS = 12000

DOMAINS: Dict[str, Dict[str, Any]] = {
    "crypto": {
        "title": "Cryptography / key math",
        "use_when": ["rsa", "modular", "cipher", "key", "prime", "lattice", "ecc", "hash", "jwt"],
        "questions": ["What is the modulus/exponent?", "Is the prime generation biased?", "Can a small key/small exponent be bounded directly?"],
        "ops": ["crypto.rsa_check", "number_theory.factor", "number_theory.modinv", "number_theory.isprime", "stats.entropy"],
    },
    "prng": {
        "title": "PRNG / token bias",
        "use_when": ["prng", "random", "csrf", "nonce", "session id", "token", "entropy", "seed"],
        "questions": ["How many samples?", "What alphabet/encoding?", "Are values sequential or reused?"],
        "ops": ["stats.entropy", "stats.runs", "stats.chi_square", "stats.ioc", "stats.summarize"],
    },
    "stats": {
        "title": "Statistics / anomaly triage",
        "use_when": ["anomaly", "outlier", "baseline", "traffic", "frequency", "distribution", "confidence"],
        "questions": ["What is the baseline?", "What sample window?", "Which delta is actionable?"],
        "ops": ["stats.summarize", "stats.chi_square", "stats.runs", "stats.timing"],
    },
    "graph": {
        "title": "Attack-graph / topology",
        "use_when": ["graph", "path", "pivot", "lateral", "centrality", "breadth", "topology", "chain"],
        "questions": ["Which nodes are observed?", "Which edges are proven?", "What is the shortest proven chain?"],
        "ops": ["graph.shortest_path", "graph.degree"],
    },
    "timing": {
        "title": "Timing / side-channel analysis",
        "use_when": ["timing", "latency", "side-channel", "response time", "oracle", "sleep", "delay"],
        "questions": ["Are samples paired?", "Is the clock stable?", "Is the delta statistically separated?"],
        "ops": ["stats.timing", "stats.summarize"],
    },
    "number_theory": {
        "title": "Number theory / L-functions",
        "use_when": ["riemann", "grh", "zeta", "l-function", "prime counting", "proof", "formalization"],
        "questions": ["Which theorem scope?", "Is the artifact verified?", "Is the implication actually algorithmic?"],
        "ops": ["number_theory.primepi", "number_theory.zeta", "number_theory.grh_bound", "number_theory.isprime"],
    },
    "proof": {
        "title": "Proof / claim audit",
        "use_when": ["proof", "claim", "verified", "formal", "lean", "theorem", "conjecture"],
        "questions": ["What is the exact statement?", "What evidence ref proves it?", "What is not covered?"],
        "ops": ["verify.residual", "rh_status"],
    },
}

RH_STATUS = {
    "as_of": "2026-11-27",
    "statement": (
        "As of 20-21 November 2026, Levent Alpoge released a Lean-formalized proof of the "
        "Riemann Hypothesis, and the Lean FRO + Mathlib Initiative certified the statement "
        "and proof. Classical RH and GRH for Dirichlet L-functions are treated as formally "
        "verified. Extended/Grand RH variants are not covered by that claim."
    ),
    "clay_status": "Prize status is pending; do not claim the Clay award was made.",
    "crypto_impact": (
        "RH/GRH does not break RSA, ECC, AES, Bitcoin, or practical factoring/discrete-log "
        "hardness. It validates Miller's original deterministic primality test under GRH and "
        "improves rigorous bounds for primes and primes in arithmetic progressions."
    ),
    "guardrail": "Never claim 'RH solves factoring' or 'RH breaks encryption'.",
    "sources": [
        "https://leanprover-community.github.io/blog/posts/riemann-hypothesis/",
        "https://alpo.ge/remark.pdf",
        "https://www.scientificamerican.com/article/the-ai-riemann-hypothesis-proof-is-almost-certainly-true-independent-mathematicians-attest/",
        "https://www.scientificamerican.com/article/why-solving-the-riemann-hypothesis-wont-break-encryption/",
        "https://www.nature.com/articles/d41586-026-03812-7",
        "https://math.mit.edu/~drew/grh/",
        "https://github.com/leanprover-community/riemann-hypothesis",
        "https://www.claymath.org/millennium/riemann-hypothesis/",
    ],
    "verified_by": "Lean FRO/Mathlib certification; independent shorter-proof checks reported in Nov 2026",
}

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STATUS_ORDER = {"candidate": 0, "verified": 1, "doctrine": 2}


def _now() -> float:
    return time.time()


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _blank_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "lessons": [],
        "events": [],
        "progress": {name: {"attempts": 0, "checks_ok": 0, "verified_lessons": 0, "mastery": 0.0, "last_at": 0.0} for name in DOMAINS},
    }


def load_state() -> Dict[str, Any]:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("state is not an object")
    except Exception:
        return _blank_state()
    base = _blank_state()
    data.setdefault("version", 1)
    data.setdefault("created_at", base["created_at"])
    data.setdefault("updated_at", _now())
    for key in ("lessons", "events"):
        if not isinstance(data.get(key), list):
            data[key] = []
    progress = data.get("progress")
    if not isinstance(progress, dict):
        progress = {}
    for domain, defaults in base["progress"].items():
        item = progress.get(domain)
        if not isinstance(item, dict):
            item = dict(defaults)
        else:
            for k, v in defaults.items():
                item.setdefault(k, v)
        progress[domain] = item
    data["progress"] = progress
    return data


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = _now()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _event(state: Dict[str, Any], kind: str, detail: Dict[str, Any]) -> None:
    state["events"].append({"ts": _now(), "kind": kind, **detail})


def _progress(state: Dict[str, Any], domain: str) -> Dict[str, Any]:
    progress = state.setdefault("progress", {})
    item = progress.get(domain)
    if not isinstance(item, dict):
        item = {"attempts": 0, "checks_ok": 0, "verified_lessons": 0, "mastery": 0.0, "last_at": 0.0}
    progress[domain] = item
    return item


def _recalculate_mastery(item: Dict[str, Any]) -> None:
    attempts = max(0, int(item.get("attempts", 0)))
    checks = max(0, int(item.get("checks_ok", 0)))
    verified = max(0, int(item.get("verified_lessons", 0)))
    item["mastery"] = round(min(1.0, attempts * 0.03 + checks * 0.05 + verified * 0.12), 3)
    item["last_at"] = _now()


def route_signals(goal: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Explainable evidence for a route decision; no side effects."""
    text = f"{goal or ''} {' '.join(str(k) for k in (payload or {}))}".lower()
    signals: Dict[str, List[str]] = {}
    for domain, meta in DOMAINS.items():
        matched = [phrase for phrase in meta["use_when"] if phrase in text]
        if matched:
            signals[domain] = matched
    if bool((payload or {}).get("samples_b")):
        signals.setdefault("timing", []).append("samples_b")
    return signals


def detect_domain(goal: str, op: str = "", payload: Optional[Dict[str, Any]] = None) -> str:
    text = f"{goal or ''} {op or ''} {' '.join(str(k) for k in (payload or {}))}".lower()
    scores: Dict[str, int] = {}
    for domain, meta in DOMAINS.items():
        scores[domain] = sum(1 for phrase in meta["use_when"] if phrase in text)
    if op:
        head = op.split(".")[0]
        mapping = {
            "crypto": "crypto", "stats": "stats", "graph": "graph",
            "number_theory": "number_theory", "verify": "proof",
        }
        if head in mapping:
            scores[mapping[head]] += 3
    if bool((payload or {}).get("samples_b")):
        scores["timing"] += 4
    best = max(scores, key=lambda key: (scores[key], -list(DOMAINS).index(key)))
    return best if scores[best] > 0 else "stats"


def _default_op(domain: str, payload: Dict[str, Any], goal: str = "") -> Tuple[str, Dict[str, Any]]:
    text = str(goal or "").lower()
    if domain == "graph":
        return "graph", {"nodes": payload.get("nodes") or [], "edges": payload.get("edges") or [],
                         "start": payload.get("start") or "", "goal": payload.get("goal") or ""}
    if domain == "number_theory":
        op = str(payload.get("number_op") or payload.get("op") or "").strip().lower()
        if not op:
            if "zeta" in text or "l-function" in text:
                op = "zeta"
            elif "bound" in text or "grh" in text:
                op = "grh_bound"
            else:
                op = "primepi"
        args: Dict[str, Any] = {"op": op, "n": payload.get("n", 100)}
        if op == "zeta":
            args["s"] = payload.get("s", 1.5)
        return "number_theory", args
    if domain == "proof":
        return "verify", {"expr": payload.get("expr") or "", "expected": payload.get("expected", 0),
                          "tolerance": payload.get("tolerance", 1e-9)}
    if domain == "crypto":
        if payload.get("n") is not None:
            op = str(payload.get("op") or "").strip().lower()
            if not op:
                op = "factor" if "factor" in text else "rsa_check"
            if op == "isprime":
                op = "prime"
            args = {"op": op, "n": payload["n"], "e": payload.get("e", 65537)}
            if op == "modinv":
                args["modulus"] = payload.get("modulus", payload.get("m", 3120))
            if op in ("factor", "factorint"):
                args["allow_slow"] = _boolish(payload.get("allow_slow", False))
            return "crypto", args
        if payload.get("samples"):
            return "crypto", {"op": "prng_bias", "samples": payload.get("samples"),
                              "test": payload.get("test", "entropy")}
        return "crypto", {
            "op": "modinv", "n": payload.get("a", 17),
            "modulus": payload.get("modulus", 3120),
        }
    if domain == "timing":
        return "stats", {"samples": payload.get("samples") or [], "samples_b": payload.get("samples_b") or [], "test": "timing"}
    test = str(payload.get("test") or ("entropy" if payload.get("samples") else "summarize"))
    return "stats", {"samples": payload.get("samples") or [], "test": test,
                     "threshold": payload.get("threshold"), "buckets": payload.get("buckets", 256)}


def _run_worker(op: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if op not in ("symbolic", "number_theory", "stats", "graph", "crypto", "verify"):
        return {"ok": False, "error": f"unknown math op: {op}"}
    python = str(appconfig.get("math_python") or DEFAULT_PYTHON or sys.executable)
    if not python or not os.path.exists(python):
        python = sys.executable
    request = json.dumps({"op": op, "args": payload or {}}, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [python, str(WORKER_PATH)],
            input=request,
            capture_output=True,
            text=True,
            timeout=MAX_WORKER_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"math worker timed out after {MAX_WORKER_SECONDS}s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"math worker failed: {exc}"}
    if proc.returncode != 0:
        return {"ok": False, "error": f"math worker exit {proc.returncode}: {(proc.stderr or '').strip()[:500]}"}
    try:
        result = json.loads(proc.stdout or "{}")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"math worker returned invalid JSON: {exc}", "stdout": (proc.stdout or "")[:500]}
    if not isinstance(result, dict):
        return {"ok": False, "error": "math worker returned a non-object result"}
    return result


def compute(op: str, payload: Dict[str, Any], *, domain: str = "", goal: str = "", target: str = "") -> Dict[str, Any]:
    domain = domain or detect_domain(goal, op, payload)
    result = _run_worker(op, payload)
    state = load_state()
    item = _progress(state, domain)
    item["attempts"] = int(item.get("attempts", 0)) + 1
    if result.get("ok"):
        item["checks_ok"] = int(item.get("checks_ok", 0)) + 1
    _recalculate_mastery(item)
    _event(state, "compute", {"domain": domain, "op": op, "ok": bool(result.get("ok"))})
    save_state(state)

    evidence_ref = ""
    resolved_target = str(target or appconfig.get("active_target") or "").strip()
    if resolved_target:
        try:
            entry = evidence_util.record(
                resolved_target,
                tool=f"math_attack:{op}",
                args={"domain": domain, "goal": str(goal)[:600], "op": op, "input": payload},
                result=json.dumps(result, ensure_ascii=False, default=str),
                ok=bool(result.get("ok")),
                source="math_attack",
                tags=["math", domain, op],
            )
            evidence_ref = str(entry.get("id") or "")
        except Exception:
            evidence_ref = ""
    result.update({
        "tool": "math_attack", "action": "compute", "domain": domain,
        "op": op, "evidence_ref": evidence_ref, "mastery": item.get("mastery"),
    })
    return result


def decide(
    goal: str = "",
    payload: Optional[Dict[str, Any]] = None,
    *,
    domain: str = "",
    auto: bool = False,
    target: str = "",
) -> Dict[str, Any]:
    """Neutral decision link.

    Given a plain-language goal and optional input, choose the math route,
    worker op, and bounded payload. By default this is side-effect free: no
    worker run, no state change, no evidence write. With ``auto=True`` it
    executes the chosen computation through ``compute``.
    """
    goal = str(goal or "").strip()
    payload = payload if isinstance(payload, dict) else {}
    override = str(domain or "").strip().lower()
    if override and override not in DOMAINS:
        return {"ok": False, "error": f"unknown math domain: {override}", "supported": sorted(DOMAINS)}

    signals = route_signals(goal, payload)
    chosen = override or detect_domain(goal, "", payload)
    matched = list(signals.get(chosen, []))
    if override:
        matched = ["explicit domain"] + matched
        confidence = "high"
    elif len(matched) >= 2:
        confidence = "high"
    elif len(matched) == 1 or bool(payload):
        confidence = "medium"
    else:
        confidence = "low"
    decided = confidence != "low"
    op, args = _default_op(chosen, payload, goal)

    decision: Dict[str, Any] = {
        "ok": True,
        "action": "decide",
        "goal": goal,
        "domain": chosen,
        "title": DOMAINS[chosen]["title"],
        "op": op,
        "input": args,
        "signals": matched,
        "confidence": confidence,
        "decided": decided,
        "executed": False,
        "route_questions": DOMAINS[chosen]["questions"],
    }
    if not decided:
        decision["question"] = "Which math route should I use for this goal?"
        decision["options"] = sorted(DOMAINS)
        decision["text"] = format_decision(decision)
        return decision

    if not auto:
        decision["text"] = format_decision(decision)
        return decision

    result = compute(op, args, domain=chosen, goal=goal, target=target)
    decision["executed"] = True
    decision["evidence_ref"] = str(result.get("evidence_ref") or "")
    decision["text"] = format_decision(decision, result)
    result.update({k: v for k, v in decision.items() if k != "ok"})
    result["ok"] = bool(result.get("ok"))
    result["action"] = "decide"
    return result


def learn(
    title: str,
    rule: str,
    *,
    domain: str = "",
    trigger: str = "",
    evidence_ref: str = "",
    confidence: float = 0.6,
    source: str = "agent",
    target: str = "",
) -> Dict[str, Any]:
    title = str(title or "").strip()
    rule = str(rule or "").strip()
    if not title or not rule:
        return {"ok": False, "error": "learn requires title and rule"}
    domain = domain or detect_domain(f"{title} {rule} {trigger}")
    resolved_target = str(target or appconfig.get("active_target") or "").strip()
    ref = str(evidence_ref or "").strip()
    ref_exists = bool(ref and resolved_target and evidence_util.has_ref(resolved_target, ref))
    status = "verified" if ref_exists else "candidate"
    state = load_state()
    lesson = {
        "id": hashlib.sha1(f"{_now()}|{title}|{rule[:200]}".encode("utf-8", "replace")).hexdigest()[:16],
        "ts": _now(),
        "title": title[:240],
        "domain": domain,
        "trigger": str(trigger or "")[:400],
        "rule": rule[:4000],
        "evidence_ref": ref,
        "evidence_target": resolved_target if ref_exists else "",
        "confidence": max(0.0, min(1.0, float(confidence or 0.6))),
        "status": status,
        "source": str(source or "agent")[:80],
        "uses": 0,
        "last_used": 0.0,
    }
    if not ref_exists and ref:
        lesson["note"] = "Evidence ref could not be checked against the active target; kept as candidate."
    state["lessons"].append(lesson)
    if status == "verified":
        _progress(state, domain)["verified_lessons"] = int(_progress(state, domain).get("verified_lessons", 0)) + 1
    _recalculate_mastery(_progress(state, domain))
    _event(state, "learn", {"domain": domain, "lesson": lesson["id"], "status": status})
    save_state(state)
    try:
        from agent import semantic_memory as semantic_util
        semantic_util.remember(
            f"[math/{domain}/{status}] {title}: {rule}",
            "math-pattern",
            tags=["math", domain, status],
            source="math_attack",
            target=resolved_target,
            cwd=os.getcwd(),
            importance=0.7 if status == "verified" else 0.45,
        )
    except Exception:
        pass
    return {"ok": True, "action": "learn", "lesson": lesson, "status": status}


def recall(query: str = "", *, domain: str = "", limit: int = 8) -> Dict[str, Any]:
    state = load_state()
    lessons = [item for item in state["lessons"] if isinstance(item, dict)]
    if domain:
        lessons = [item for item in lessons if str(item.get("domain", "")) == domain]
    q = set(_TOKEN_RE.findall(str(query or "").lower()))
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for item in lessons:
        text = " ".join(str(item.get(k, "")) for k in ("title", "rule", "trigger", "domain")).lower()
        tokens = set(_TOKEN_RE.findall(text))
        score = len(q & tokens) + (2 if str(item.get("status")) == "verified" else 0)
        scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], -float(pair[1].get("ts", 0))))
    limit = max(1, min(30, int(limit or 8)))
    picked = [item for score, item in scored[:limit] if score > 0 or not q]
    changed = False
    for item in picked:
        item["uses"] = int(item.get("uses", 0)) + 1
        item["last_used"] = _now()
        changed = True
    if changed:
        _event(state, "recall", {"query": str(query)[:200], "domain": domain, "count": len(picked)})
        save_state(state)
    return {"ok": True, "action": "recall", "count": len(picked), "lessons": picked, "text": format_lessons(picked)}


def _status_data() -> Dict[str, Any]:
    state = load_state()
    lessons = [item for item in state["lessons"] if isinstance(item, dict)]
    verified = sum(1 for item in lessons if item.get("status") == "verified")
    by_domain: Dict[str, int] = {}
    for item in lessons:
        name = str(item.get("domain") or "unknown")
        by_domain[name] = by_domain.get(name, 0) + 1
    return {
        "ok": True, "action": "status", "path": str(STATE_PATH),
        "lessons": len(lessons), "verified_lessons": verified,
        "events": len(state.get("events") or []),
        "lessons_by_domain": dict(sorted(by_domain.items())),
        "progress": state.get("progress", {}),
    }


def status() -> Dict[str, Any]:
    data = _status_data()
    data["text"] = format_status()
    return data


def domains() -> Dict[str, Any]:
    return {"ok": True, "action": "domains", "domains": DOMAINS, "text": format_domains()}


def rh_status() -> Dict[str, Any]:
    return {"ok": True, "action": "rh_status", **RH_STATUS, "text": format_rh()}


def format_decision(data: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> str:
    lines = [
        "NEUTRAL DECISION LINK",
        f"  goal: {str(data.get('goal') or '-')[:240]}",
    ]
    if not data.get("decided"):
        lines.extend([
            "  decision: NOT DECIDED - ask one clarifying question before computing",
            f"  question: {data.get('question') or 'Which route should I use?'}",
            f"  options: {', '.join(data.get('options') or [])}",
        ])
        return "\n".join(lines)
    lines.extend([
        f"  route: {data.get('domain')} ({data.get('title')}) confidence={data.get('confidence')}",
        f"  signals: {', '.join(data.get('signals') or []) or 'payload only'}",
        f"  op: {data.get('op')}",
        f"  input: {json.dumps(data.get('input') or {}, ensure_ascii=False, default=str)[:600]}",
        f"  questions: {' | '.join(data.get('route_questions') or [])}",
    ])
    if result is not None:
        lines.append(
            f"  executed: yes | ok={bool(result.get('ok'))} "
            f"| evidence_ref={result.get('evidence_ref') or '-'}"
        )
    else:
        lines.append("  executed: no | use execute=true or /math auto to run")
    return "\n".join(lines)


def format_card() -> str:
    """Compact status card used by the TUI and agent tool text output."""
    data = _status_data()
    progress = data.get("progress") or {}
    top_name, top_item = "", {"mastery": 0.0}
    for name, item in sorted(progress.items()):
        if isinstance(item, dict) and float(item.get("mastery", 0)) > float(top_item.get("mastery", 0)):
            top_name, top_item = name, item
    return "\n".join([
        "MATH LOOP  FORMALIZE -> COMPUTE -> VERIFY -> CRYPTO IMPACT",
        f"  state: {data['lessons']} lessons ({data['verified_lessons']} evidence-verified), {data['events']} events",
        f"  top route: {top_name or '-'} mastery={top_item.get('mastery', 0)}",
        "  gates: evidence_ref required for verified lessons; RH/GRH does not break encryption",
    ])


def format_domains() -> str:
    lines = [format_card(), "", "Math-for-attacks routes:"]
    for name, meta in DOMAINS.items():
        lines.append(f"- {name}: {meta['title']} | ops: {', '.join(meta['ops'][:4])}")
    return "\n".join(lines)


def format_status() -> str:
    data = _status_data()
    lines = [
        format_card(),
        "",
        f"Math learning state: {data['lessons']} lessons ({data['verified_lessons']} evidence-verified), {data['events']} events",
        f"State: {data['path']}",
    ]
    for name, item in sorted(data["progress"].items()):
        lines.append(
            f"- {name}: mastery={item.get('mastery', 0)} attempts={item.get('attempts', 0)} "
            f"checks_ok={item.get('checks_ok', 0)} verified_lessons={item.get('verified_lessons', 0)}"
        )
    return "\n".join(lines)


def format_lessons(lessons: Sequence[Dict[str, Any]]) -> str:
    if not lessons:
        return "No math lessons learned yet. Use `math_attack learn` after a verified computation."
    lines = []
    for item in lessons:
        head = f"[{item.get('status', 'candidate')}] {item.get('domain', 'math')}: {item.get('title', 'lesson')}"
        lines.append(head)
        lines.append(f"  Rule: {str(item.get('rule', ''))[:600]}")
        if item.get("trigger"):
            lines.append(f"  Trigger: {str(item.get('trigger'))[:200]}")
        if item.get("evidence_ref"):
            lines.append(f"  Evidence: {item.get('evidence_ref')}")
    return "\n".join(lines)


def format_rh() -> str:
    return "\n".join([
        format_card(),
        "",
        f"Riemann hypothesis status ({RH_STATUS['as_of']}): formally verified in Lean; Clay prize status pending.",
        RH_STATUS["statement"],
        f"Crypto: {RH_STATUS['crypto_impact']}",
        RH_STATUS["guardrail"],
        "Sources: " + " | ".join(RH_STATUS["sources"][:4]),
    ])


def run(args: Optional[Dict[str, Any]] = None, *, target: str = "") -> Dict[str, Any]:
    args = args or {}
    action = str(args.get("action") or "status").strip().lower()
    if action in ("rh", "riemann", "rh_status"):
        return rh_status()
    if action in ("domain", "domains", "routes"):
        return domains()
    if action in ("status", "summary"):
        return status()
    if action in ("review", "recall", "lessons", "search"):
        return recall(
            str(args.get("query") or ""),
            domain=str(args.get("domain") or ""),
            limit=int(args.get("limit", 8) or 8),
        )
    if action in ("decide", "route", "plan", "auto"):
        payload = args.get("input")
        if not isinstance(payload, dict):
            payload = args.get("params")
        if not isinstance(payload, dict):
            payload = {}
        flag = args.get("execute", args.get("auto", action == "auto"))
        return decide(
            str(args.get("goal") or args.get("objective") or ""),
            payload,
            domain=str(args.get("domain") or ""),
            auto=_boolish(flag),
            target=target,
        )
    if action in ("learn", "remember"):
        return learn(
            str(args.get("title") or ""),
            str(args.get("rule") or args.get("text") or ""),
            domain=str(args.get("domain") or ""),
            trigger=str(args.get("trigger") or ""),
            evidence_ref=str(args.get("evidence_ref") or ""),
            confidence=float(args.get("confidence", 0.6) or 0.6),
            source=str(args.get("source") or "agent"),
            target=target,
        )
    if action in ("verify", "check"):
        payload = {
            "expr": str(args.get("expr") or args.get("claim") or ""),
            "expected": args.get("expected", 0),
            "tolerance": args.get("tolerance", 1e-9),
        }
        result = compute("verify", payload, domain="proof", goal=str(args.get("goal") or "verify math claim"), target=target)
        result["action"] = "verify"
        return result
    if action in ("analyze", "compute", "solve", "math_solve"):
        goal = str(args.get("goal") or "")
        op = str(args.get("op") or "")
        payload = args.get("input")
        if not isinstance(payload, dict):
            payload = args.get("params")
        if not isinstance(payload, dict):
            payload = dict(args)
        domain = str(args.get("domain") or "").strip().lower()
        if not op:
            domain = domain or detect_domain(goal, "", payload)
            op, payload = _default_op(domain, payload, goal)
        domain = domain or detect_domain(goal, op, payload)
        result = compute(op, payload, domain=domain, goal=goal, target=target)
        result["action"] = "analyze" if action == "analyze" else action
        result["goal"] = goal
        if action == "analyze":
            result["route_questions"] = DOMAINS.get(domain, {}).get("questions", [])
        return result
    if action == "help":
        return {
            "ok": True, "action": "help",
            "actions": ["decide", "analyze", "compute", "verify", "learn", "recall", "status", "domains", "rh_status"],
            "example": {"action": "decide", "goal": "check session token entropy", "input": {"samples": [1, 2, 3, 4], "test": "entropy"}},
            "text": "Use math_attack decide as the neutral link: it picks the route without side effects; add execute=true to run. Then analyze/verify, learn after evidence, recall before repeat work.",
        }
    return {"ok": False, "error": f"unknown math_attack action: {action}", "supported": ["decide", "analyze", "compute", "verify", "learn", "recall", "status", "domains", "rh_status", "help"]}
