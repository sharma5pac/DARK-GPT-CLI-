#!/usr/bin/env python3
"""Bounded math worker for DarkGPT attack analysis.

The worker reads one JSON request on stdin and writes one JSON response on
stdout. It is deliberately whitelist-based: no arbitrary code is accepted from
the model, expressions are token-checked before SymPy parses them, and every
collection is size-bounded. Run it with system Python, which already has
sympy/mpmath/numpy/scipy even when the DarkGPT venv does not.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

MAX_EXPR = 2000
MAX_SAMPLES = 20000
MAX_TEXT = 4000
MAX_NODES = 300
MAX_FACTOR_N = 10 ** 12
BLOCKED_TOKENS = (
    "__", "import", "exec", "eval", "open", "lambda", "globals", "locals",
    "getattr", "setattr", "breakpoint", "compile", "input", "system",
)
EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().,= <>|:%]+$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


def _ok(data: Optional[Dict[str, Any]] = None, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True}
    if data:
        result.update(data)
    result.update(extra)
    return result


def _fail(message: Any, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "error": str(message)[:500]}
    result.update(extra)
    return result


def _sympy():  # pragma: no cover - exercised only when worker runs under system Python
    try:
        import sympy as sp  # type: ignore
        return sp
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"sympy unavailable in math worker: {exc}") from exc


def _check_expr(expr: Any) -> str:
    text = str(expr or "").strip()
    if not text:
        raise ValueError("expression is required")
    if len(text) > MAX_EXPR:
        raise ValueError(f"expression too long (max {MAX_EXPR})")
    if not EXPR_RE.fullmatch(text):
        raise ValueError("expression contains unsupported characters")
    lowered = text.lower()
    if any(token in lowered for token in BLOCKED_TOKENS):
        raise ValueError("expression contains a blocked token")
    return text


def _check_name(name: Any, default: str = "x") -> str:
    text = str(name or default).strip()
    if not NAME_RE.fullmatch(text):
        raise ValueError("symbol must be a simple identifier")
    return text


def _parse_expr(expr: Any):
    sp = _sympy()
    from sympy.parsing.sympy_parser import (  # type: ignore
        convert_xor,
        parse_expr,
        standard_transformations,
    )

    text = _check_expr(expr)
    symbol_names = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", text))
    local: Dict[str, Any] = {
        "E": sp.E,
        "I": sp.I,
        "pi": sp.pi,
        "Abs": sp.Abs,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "zeta": sp.zeta,
        "factorial": sp.factorial,
    }
    for name in symbol_names:
        if name not in local and NAME_RE.fullmatch(name):
            local[name] = sp.Symbol(name)
    # parse_expr's transformations reference a few SymPy classes in the global
    # namespace. Expose only those names and keep Python builtins empty; the
    # token gate above blocks import/eval/open/lambda style tokens.
    global_dict = {
        "__builtins__": {},
        "Integer": sp.Integer,
        "Float": sp.Float,
        "Rational": sp.Rational,
        "Symbol": sp.Symbol,
        "Function": sp.Function,
        "Lambda": sp.Lambda,
        "factorial": sp.factorial,
    }
    return parse_expr(
        text,
        local_dict=local,
        global_dict=global_dict,
        transformations=standard_transformations + (convert_xor,),
        evaluate=True,
    )


def _render(value: Any, precision: int = 12) -> Dict[str, Any]:
    sp = _sympy()
    return {
        "text": sp.sstr(value),
        "latex": sp.latex(value),
        "numeric": sp.sstr(sp.N(value, precision)) if value is not None else "",
    }


def _symbolic(args: Dict[str, Any]) -> Dict[str, Any]:
    sp = _sympy()
    expr = str(args.get("expr") or "")
    op = str(args.get("op") or "simplify").strip().lower()
    symbol = sp.Symbol(_check_name(args.get("symbol")))
    if "=" in expr:
        lhs, rhs = expr.split("=", 1)
        parsed = _parse_expr(lhs) - _parse_expr(rhs)
    else:
        parsed = _parse_expr(expr)
    if op in ("simplify", "expand", "factor", "diff", "integrate"):
        fn = {
            "simplify": sp.simplify,
            "expand": sp.expand,
            "factor": sp.factor,
            "diff": lambda value: sp.diff(value, symbol),
            "integrate": lambda value: sp.integrate(value, symbol),
        }[op]
        value = fn(parsed)
        rendered = _render(value, int(args.get("precision", 12)))
        residual = sp.simplify(value - parsed) if op == "simplify" else None
        return _ok({"op": op, "expression": expr, **rendered, "residual": _render(residual) if residual is not None else None})
    if op == "solve":
        solutions = sp.solve(parsed, symbol)
        rendered = [_render(s, int(args.get("precision", 12))) for s in solutions[:50]]
        checked = []
        for s in solutions[:50]:
            residual = sp.simplify(parsed.subs(symbol, s))
            checked.append({"solution": sp.sstr(s), "residual": sp.sstr(residual), "satisfies": residual == 0})
        return _ok({"op": op, "expression": expr, "symbol": str(symbol), "solutions": rendered, "checked": checked})
    if op == "limit":
        point = _parse_expr(str(args.get("point", 0)))
        value = sp.limit(parsed, symbol, point)
        return _ok({"op": op, "expression": expr, **_render(value, int(args.get("precision", 12)))})
    return _fail(f"unknown symbolic op: {op}", supported=["simplify", "expand", "factor", "diff", "integrate", "solve", "limit"])


def _as_int(value: Any, label: str = "n") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} must be an integer") from exc


def _number_theory(args: Dict[str, Any]) -> Dict[str, Any]:
    sp = _sympy()
    op = str(args.get("op") or "isprime").strip().lower()
    n = _as_int(args.get("n"), "n")
    if op in ("factor", "factorint"):
        if abs(n) > MAX_FACTOR_N and not bool(args.get("allow_slow")):
            return _ok({
                "op": op, "n": n, "factors": None,
                "note": f"|n| > {MAX_FACTOR_N}; refusing an unbounded factor run. Use a dedicated factoring route or opt in with allow_slow.",
            })
        factors = sp.factorint(n, limit=int(args.get("limit", 100000)) or 100000)
        return _ok({"op": op, "n": n, "factors": {str(k): int(v) for k, v in factors.items()}})
    if op == "isprime":
        return _ok({"op": op, "n": n, "is_prime": bool(sp.isprime(n)), "method": "sympy BPSW (deterministic for tested ranges)"})
    if op == "nextprime":
        return _ok({"op": op, "n": n, "next_prime": int(sp.nextprime(n))})
    if op == "primepi":
        return _ok({"op": op, "n": n, "prime_pi": int(sp.primepi(n))})
    if op == "totient":
        return _ok({"op": op, "n": n, "totient": int(sp.totient(n))})
    if op == "gcd":
        a = _as_int(args.get("a"), "a")
        return _ok({"op": op, "a": a, "b": n, "gcd": math.gcd(a, n)})
    if op in ("modinv", "inverse"):
        modulus = _as_int(args.get("modulus", args.get("m")), "modulus")
        value = sp.mod_inverse(n % modulus, modulus)
        return _ok({"op": op, "a": n, "modulus": modulus, "inverse": None if value is None else int(value), "exists": value is not None})
    if op == "zeta":
        precision = max(4, min(80, int(args.get("precision", 20))))
        return _ok({"op": op, "s": str(args.get("s", n)), "zeta": sp.sstr(sp.N(sp.zeta(args.get("s", n)), precision))})
    if op == "grh_bound":
        x = max(2, n)
        logx = math.log(x)
        bound = 8 * math.pi ** 2 * math.sqrt(x) * (logx + 1)
        return _ok({
            "op": op, "x": x, "explicit_bound": bound,
            "note": "Illustrative explicit-estimate bound under GRH; not a proof of primality for a specific integer.",
        })
    return _fail(f"unknown number_theory op: {op}", supported=["factor", "isprime", "nextprime", "primepi", "totient", "gcd", "modinv", "zeta", "grh_bound"])


def _stats(args: Dict[str, Any]) -> Dict[str, Any]:
    samples = args.get("samples") or []
    if isinstance(samples, str):
        samples = list(samples)
    if not isinstance(samples, list):
        return _fail("samples must be a list")
    if len(samples) > MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]
    test = str(args.get("test") or "summarize").strip().lower()

    if test in ("entropy", "chi_square", "ioc"):
        items = [str(v) for v in samples]
        if not items:
            return _fail("samples is empty")
        counts = Counter(items)
        total = len(items)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
        top = counts.most_common(12)
        if test == "entropy":
            return _ok({
                "test": test, "n": total, "unique": len(counts),
                "entropy_bits": round(entropy, 6), "max_entropy_bits": round(max_entropy, 6),
                "uniformity": round(entropy / max_entropy, 6) if max_entropy else 1.0,
                "top": [{"value": k[:80], "count": v} for k, v in top],
            })
        if test == "ioc":
            ioc = sum(c * (c - 1) for c in counts.values()) / (total * (total - 1)) if total > 1 else 0.0
            return _ok({"test": test, "n": total, "ioc": round(ioc, 6), "top": [{"value": k[:80], "count": v} for k, v in top]})
        buckets = max(2, min(4096, int(args.get("buckets", 256))))
        observed = Counter()
        for item in items:
            # Python's hash() is randomized per process; SHA-256 keeps bucket
            # assignment deterministic across worker invocations.
            digest = hashlib.sha256(item.encode("utf-8", "replace")).digest()
            bucket = int.from_bytes(digest[:4], "big") % buckets
            observed[bucket] += 1
        expected = total / buckets
        chi2 = sum((observed.get(i, 0) - expected) ** 2 / expected for i in range(buckets))
        return _ok({"test": test, "n": total, "buckets": buckets, "chi_square": round(chi2, 6), "expected_per_bucket": round(expected, 3)})

    numeric: List[float] = []
    for value in samples:
        try:
            numeric.append(float(value))
        except Exception:  # noqa: BLE001
            continue
    if not numeric:
        return _fail("no numeric samples found")
    if test == "summarize":
        mean = statistics.fmean(numeric)
        sd = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
        last = numeric[-1]
        return _ok({
            "test": test, "n": len(numeric), "mean": round(mean, 6),
            "median": round(statistics.median(numeric), 6),
            "stdev": round(sd, 6), "variance": round(sd * sd, 6),
            "min": min(numeric), "max": max(numeric), "last": last,
            "last_zscore": round((last - mean) / sd, 6) if sd else 0.0,
        })
    if test == "runs":
        if len(numeric) < 4:
            return _fail("runs test needs at least 4 samples")
        threshold = float(args.get("threshold", statistics.median(numeric)))
        bits = [1 if value >= threshold else 0 for value in numeric]
        ones = sum(bits)
        zeros = len(bits) - ones
        if not ones or not zeros:
            return _ok({"test": test, "runs": 1, "z": 0.0, "note": "all samples on one side of threshold"})
        runs = 1 + sum(1 for i in range(1, len(bits)) if bits[i] != bits[i - 1])
        n = len(bits)
        expected = 2 * ones * zeros / n + 1
        variance = (2 * ones * zeros * (2 * ones * zeros - n)) / (n * n * (n - 1)) if n > 1 else 0.0
        z = (runs - expected) / math.sqrt(variance) if variance > 0 else 0.0
        return _ok({
            "test": test, "n": n, "threshold": threshold, "runs": runs,
            "expected_runs": round(expected, 6), "z": round(z, 6),
            "two_sided_p": round(math.erfc(abs(z) / math.sqrt(2)), 6),
        })
    if test == "timing":
        other = args.get("samples_b") or []
        if not isinstance(other, list) or not other:
            return _fail("timing test requires samples_b")
        other_numeric = []
        for value in other:
            try:
                other_numeric.append(float(value))
            except Exception:  # noqa: BLE001
                continue
        if not other_numeric:
            return _fail("samples_b has no numeric values")
        mean_a = statistics.fmean(numeric)
        mean_b = statistics.fmean(other_numeric)
        sd_a = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
        sd_b = statistics.pstdev(other_numeric) if len(other_numeric) > 1 else 0.0
        se = math.sqrt(sd_a ** 2 / len(numeric) + sd_b ** 2 / len(other_numeric)) if (numeric and other_numeric) else 0.0
        delta = mean_a - mean_b
        return _ok({
            "test": test, "n_a": len(numeric), "n_b": len(other_numeric),
            "mean_a": round(mean_a, 6), "mean_b": round(mean_b, 6),
            "delta": round(delta, 6), "std_error": round(se, 6),
            "z": round(delta / se, 6) if se else 0.0,
            "note": "Use repeated, controlled samples; timing deltas alone are not a finding.",
        })
    return _fail(f"unknown stats test: {test}", supported=["summarize", "entropy", "runs", "chi_square", "ioc", "timing"])


def _graph(args: Dict[str, Any]) -> Dict[str, Any]:
    nodes = [str(n) for n in (args.get("nodes") or [])][:MAX_NODES]
    edges = args.get("edges") or []
    if not isinstance(edges, list):
        return _fail("edges must be a list of [source, target] pairs")
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for edge in edges[:MAX_NODES * 4]:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        src, dst = str(edge[0]), str(edge[1])
        adj.setdefault(src, [])
        adj.setdefault(dst, [])
        adj[src].append(dst)
        adj[dst].append(src)
    degree = {name: len(neighbours) for name, neighbours in adj.items()}
    ranked = sorted(degree.items(), key=lambda pair: (-pair[1], pair[0]))[:50]
    start = str(args.get("start") or "").strip()
    goal = str(args.get("goal") or "").strip()
    path: List[str] = []
    distance: Optional[int] = None
    if start and goal and start in adj and goal in adj:
        queue: deque[Tuple[str, List[str]]] = deque([(start, [start])])
        seen = {start}
        while queue:
            current, trail = queue.popleft()
            if current == goal:
                path = trail
                distance = len(trail) - 1
                break
            for neighbour in adj[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append((neighbour, trail + [neighbour]))
    return _ok({
        "nodes": len(adj), "edges": sum(len(v) for v in adj.values()) // 2,
        "top_degree": [{"node": k, "degree": v} for k, v in ranked],
        "shortest_path": path, "distance": distance,
        "note": "Graph math ranks observed assets only; it does not create evidence of access.",
    })


def _crypto(args: Dict[str, Any]) -> Dict[str, Any]:
    op = str(args.get("op") or "rsa_check").strip().lower()
    n = _as_int(args.get("n"), "n")
    if op == "modinv":
        modulus = _as_int(args.get("modulus", args.get("m")), "modulus")
        return _number_theory({"op": "modinv", "n": n, "modulus": modulus})
    if op == "gcd":
        a = _as_int(args.get("a"), "a")
        return _number_theory({"op": "gcd", "n": n, "a": a})
    if op == "prime":
        return _number_theory({"op": "isprime", "n": n})
    if op == "factor":
        return _number_theory({"op": "factor", "n": n, "allow_slow": bool(args.get("allow_slow"))})
    if op == "rsa_check":
        e = _as_int(args.get("e", 65537), "e")
        result: Dict[str, Any] = {
            "op": op, "n_bits": n.bit_length(), "e": e,
            "small_e": e < 65537,
            "guardrail": "RH/GRH does not break RSA; it improves conditional bounds and deterministic primality reasoning.",
        }
        if abs(n) <= MAX_FACTOR_N:
            factors = _number_theory({"op": "factor", "n": n})
            result["factors"] = factors.get("factors")
        else:
            result["factors"] = None
            result["note"] = f"n is larger than the {MAX_FACTOR_N} local factor gate; use a dedicated factorization route."
        return _ok(result)
    if op == "prng_bias":
        return _stats({"samples": args.get("samples") or [], "test": args.get("test", "entropy")})
    return _fail(f"unknown crypto op: {op}", supported=["rsa_check", "modinv", "gcd", "prime", "factor", "prng_bias"])


def _verify(args: Dict[str, Any]) -> Dict[str, Any]:
    sp = _sympy()
    expr = str(args.get("expr") or "")
    expected_raw = args.get("expected")
    expected = _parse_expr(str(expected_raw)) if expected_raw is not None else sp.Integer(0)
    parsed = _parse_expr(expr)
    if "=" in expr:
        lhs, rhs = expr.split("=", 1)
        parsed = _parse_expr(lhs) - _parse_expr(rhs)
    residual = sp.simplify(parsed - expected)
    try:
        numeric = float(sp.N(residual, 30))
    except Exception:  # noqa: BLE001
        numeric = None
    tolerance = abs(float(args.get("tolerance", 1e-9)))
    verified = residual == 0 or (numeric is not None and abs(numeric) <= tolerance)
    return _ok({
        "expr": expr, "expected": str(expected_raw), "residual": sp.sstr(residual),
        "numeric_residual": numeric, "tolerance": tolerance, "verified": bool(verified),
        "gate": "verified is a math check only; it is not security proof or target evidence.",
    })


HANDLERS = {
    "symbolic": _symbolic,
    "number_theory": _number_theory,
    "stats": _stats,
    "graph": _graph,
    "crypto": _crypto,
    "verify": _verify,
}


def main() -> None:
    started = time.perf_counter()
    try:
        request = json.loads(sys.stdin.read() or "{}")
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        op = str(request.get("op") or "").strip().lower()
        handler = HANDLERS.get(op)
        if handler is None:
            response = _fail(f"unknown math op: {op}", supported=sorted(HANDLERS))
        else:
            args = request.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError("args must be a JSON object")
            response = handler(args)
    except Exception as exc:  # noqa: BLE001
        response = _fail(exc)
    response["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
    sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
