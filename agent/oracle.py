"""Machine oracle validation for DarkGPT.

Treats scanner/LLM claims as untrusted until deterministic checks pass.
Implements lightweight oracles: differential response, timing delta, reflection,
and basic header/status evidence checks. OAST/callback hooks are recorded as
external-oracle-required when no listener configured.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


def _safe_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15) -> Dict[str, Any]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout, allow_redirects=False, verify=False)
        return {"ok": True, "status": r.status_code, "len": len(r.content or b""), "text": r.text[:2000], "headers": dict(r.headers)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def differential_response(url_a: str, url_b: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    a = _safe_get(url_a, headers=headers)
    b = _safe_get(url_b, headers=headers)
    same_status = a.get("status") == b.get("status")
    len_a, len_b = int(a.get("len") or 0), int(b.get("len") or 0)
    delta = abs(len_a - len_b)
    ratio = delta / max(1, max(len_a, len_b))
    text_a, text_b = str(a.get("text") or ""), str(b.get("text") or "")
    # crude similarity
    same_prefix = text_a[:200] == text_b[:200]
    verdict = "different" if (not same_status or ratio > 0.15 or not same_prefix) else "similar"
    return {"oracle": "differential_response", "verdict": verdict, "a": a, "b": b, "delta_bytes": delta, "delta_ratio": round(ratio, 4), "same_status": same_status}


def timing_delta(url: str, baseline_url: Optional[str] = None, samples: int = 3, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    def measure(u):
        times = []
        for _ in range(max(2, min(samples, 6))):
            t0 = time.perf_counter()
            r = _safe_get(u, headers=headers)
            times.append((time.perf_counter() - t0) * 1000.0)
        return times, r
    t1, r1 = measure(url)
    out = {"oracle": "timing_delta", "url": url, "samples_ms": [round(x, 2) for x in t1], "median_ms": round(statistics.median(t1), 2), "response": {k: r1.get(k) for k in ("ok", "status", "len", "error")}}
    if baseline_url:
        t0, r0 = measure(baseline_url)
        out["baseline_url"] = baseline_url
        out["baseline_samples_ms"] = [round(x, 2) for x in t0]
        out["baseline_median_ms"] = round(statistics.median(t0), 2)
        out["delta_ms"] = round(out["median_ms"] - out["baseline_median_ms"], 2)
        out["verdict"] = "potential_delay" if out["delta_ms"] > 1500 else "no_significant_delay"
    else:
        out["verdict"] = "measured"
    return out


def reflection_check(url: str, marker: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    r = _safe_get(url, headers=headers)
    body = str(r.get("text") or "")
    found = marker in body
    return {"oracle": "reflection_check", "marker": marker, "found": found, "verdict": "reflected" if found else "not_reflected", "response": {k: r.get(k) for k in ("ok", "status", "len", "error")}}


def run_oracle(kind: str, **kwargs: Any) -> Dict[str, Any]:
    kind = str(kind or "").strip().lower()
    if kind == "differential_response":
        return differential_response(str(kwargs.get("url_a") or ""), str(kwargs.get("url_b") or ""), headers=kwargs.get("headers"))
    if kind == "timing_delta":
        return timing_delta(str(kwargs.get("url") or ""), kwargs.get("baseline_url"), int(kwargs.get("samples") or 3), headers=kwargs.get("headers"))
    if kind == "reflection_check":
        return reflection_check(str(kwargs.get("url") or ""), str(kwargs.get("marker") or ""), headers=kwargs.get("headers"))
    if kind in ("oast", "interactsh", "callback"):
        return {"oracle": kind, "verdict": "external_listener_required", "ok": False, "error": "No OAST/interactsh listener configured in this runtime. Configure a callback listener and provide evidence ref; oracle cannot auto-verify without it."}
    return {"ok": False, "error": f"unknown oracle kind: {kind}"}
