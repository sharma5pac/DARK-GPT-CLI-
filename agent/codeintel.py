"""Lightweight code intelligence: symbol outlines, references and safe edits.

Uses the Python standard library AST for Python and a conservative regex
outline for JS/TS-family files. Reference search uses ripgrep when present,
with a grep fallback. No third-party index or daemon is required.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

MAX_RESULTS = 120
CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java", ".c", ".h", ".cpp", ".rb"}


def safe_path(cwd: str, value: str) -> Path:
    base = Path(cwd).resolve()
    p = (base / (value or ".")).resolve()
    if p != base and base not in p.parents:
        raise ValueError("path escapes the working directory")
    return p


def symbols(path: str, cwd: str = "") -> Dict[str, Any]:
    p = safe_path(cwd or os.getcwd(), path)
    if not p.is_file():
        return {"error": f"not found: {p}"}
    text = p.read_text(errors="replace")
    suffix = p.suffix.lower()
    found: List[Dict[str, Any]] = []
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return {"error": f"syntax error: {exc}"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                args = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in (node.args.posonlyargs + node.args.args)]
                found.append({
                    "kind": kind,
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "signature": f"{node.name}({', '.join(args)})",
                })
    else:
        pattern = re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
            r"(?:function|class|interface|type|const|let|var|def|fn|pub\s+fn)\s+([A-Za-z_$][\w$]*)",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            found.append({
                "kind": "symbol",
                "name": match.group(1),
                "line": text.count("\n", 0, match.start()) + 1,
                "end_line": 0,
                "signature": match.group(0).strip(),
            })
    found.sort(key=lambda row: row["line"])
    return {"path": str(p), "language": suffix or "text", "symbols": found[:MAX_RESULTS]}


def _search(cwd: str, pattern: str, path: str = "", limit: int = MAX_RESULTS) -> List[str]:
    target = str(safe_path(cwd, path or "."))
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--color", "never", pattern, target]
    else:
        cmd = ["grep", "-RIn", pattern, target]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        lines = (proc.stdout or "").splitlines()
    except Exception:
        lines = []
    return [line[:400] for line in lines if line.strip()][:limit]


def find_refs(symbol: str, cwd: str = "", path: str = "") -> Dict[str, Any]:
    if not symbol or not re.fullmatch(r"[A-Za-z_$][\w$]*", symbol):
        return {"error": "invalid symbol name"}
    lines = _search(cwd or os.getcwd(), rf"\b{re.escape(symbol)}\b", path)
    return {"symbol": symbol, "count": len(lines), "matches": lines}


def rename(symbol: str, new_name: str, cwd: str = "", path: str = "", dry_run: bool = False) -> Dict[str, Any]:
    """Rename a whole word across code files in the working directory."""
    cwd = cwd or os.getcwd()
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", symbol) or not re.fullmatch(r"[A-Za-z_$][\w$]*", new_name):
        return {"error": "invalid symbol name"}
    refs = find_refs(symbol, cwd, path)
    files: Dict[str, int] = {}
    for line in refs.get("matches", []):
        filename = line.split(":", 1)[0]
        files[filename] = files.get(filename, 0) + 1
    changed: List[Dict[str, Any]] = []
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    for filename in sorted(files)[:80]:
        target = Path(filename)
        if not target.is_file() or target.suffix.lower() not in CODE_SUFFIXES:
            continue
        try:
            text = target.read_text(errors="replace")
        except Exception:
            continue
        updated = pattern.sub(new_name, text)
        if updated == text:
            continue
        preview = "".join(difflib.unified_diff(
            text.splitlines(True), updated.splitlines(True),
            fromfile=str(target), tofile=str(target), n=2,
        ))
        item: Dict[str, Any] = {
            "path": str(target),
            "replacements": len(pattern.findall(text)),
            "diff": preview[:4000],
        }
        if not dry_run:
            target.write_text(updated)
            item["ok"] = True
        changed.append(item)
    return {"from": symbol, "to": new_name, "files": changed, "dry_run": bool(dry_run)}


def patch(path: str, hunks: List[Dict[str, str]], cwd: str = "") -> Dict[str, Any]:
    """Apply several exact-text replacements atomically, with rollback."""
    p = safe_path(cwd or os.getcwd(), path)
    if not p.is_file():
        return {"error": f"not found: {p}"}
    original = p.read_text(errors="replace")
    current = original
    for hunk in hunks or []:
        old = str(hunk.get("old") or "")
        new = str(hunk.get("new") or "")
        if not old or old not in current:
            return {"error": f"old text not found: {old[:80]!r}", "applied": 0}
        current = current.replace(old, new, 1)
    diff = "".join(difflib.unified_diff(
        original.splitlines(True), current.splitlines(True),
        fromfile=str(p), tofile=str(p), n=3,
    ))
    if current != original:
        p.write_text(current)
    return {"path": str(p), "ok": True, "hunks": len(hunks or []), "diff": diff[:4000]}
