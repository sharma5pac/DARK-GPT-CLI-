"""Cline-compatible built-in tool implementations.

Cline 3.x exposes a consolidated tool catalog: ``read_files``,
``search_codebase``, ``run_commands``, ``fetch_web_content``, ``apply_patch``,
``editor``, ``skills``, ``ask_question`` and ``submit_and_exit``.  This module
contains the filesystem/network primitives behind those tools.  Permission
gating, hooks, undo snapshots and streaming stay in ``agent.core.ToolRegistry``
so every Cline tool inherits the same safety path as DarkGPT's native tools.

All functions are deliberately synchronous and side-effect conscious: batch
tools return structured per-item results, and ``apply_patch`` validates every
file before writing anything so a malformed patch cannot leave a half-edited
workspace.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

MAX_ITEM_OUTPUT = 8000
HTTP_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; DarkGPTClineTools/1.0; +https://openrouter.ai)"


def _path(cwd: str, value: str) -> Path:
    p = Path(str(value or "."))
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def _clamp(text: str, limit: int = MAX_ITEM_OUTPUT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


# ---------------------------------------------------------------- read_files
def normalise_read_requests(raw: Any) -> List[Dict[str, Any]]:
    """Accept every input shape Cline's ReadFilesInputUnionSchema allows."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"path": raw, "start_line": None, "end_line": None}]
    if isinstance(raw, list):
        out: List[Dict[str, Any]] = []
        for item in raw:
            out.extend(normalise_read_requests(item))
        return out
    if isinstance(raw, dict):
        for key in ("files", "paths", "file_paths"):
            if key in raw:
                return normalise_read_requests(raw[key])
        path = raw.get("path", raw.get("file_path", raw.get("filePath")))
        if path is None:
            return []
        try:
            start = int(raw["start_line"]) if raw.get("start_line") is not None else None
        except (TypeError, ValueError):
            start = None
        try:
            end = int(raw["end_line"]) if raw.get("end_line") is not None else None
        except (TypeError, ValueError):
            end = None
        return [{"path": str(path), "start_line": start, "end_line": end}]
    return []


def read_files(raw: Any, cwd: str) -> Dict[str, Any]:
    requests_ = normalise_read_requests(raw)
    results: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for req in requests_[:100]:
        pth = _path(cwd, req["path"])
        item: Dict[str, Any] = {"path": str(pth)}
        if not pth.exists():
            item.update(ok=False, error=f"not found: {pth}")
        elif pth.is_dir():
            item.update(ok=False, error=f"is a directory: {pth}")
        else:
            lines = pth.read_text(errors="replace").splitlines()
            start = req.get("start_line") or 1
            end = req.get("end_line") or len(lines)
            start = max(1, int(start))
            end = min(len(lines), max(start, int(end)))
            chunk = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
            content = _clamp("\n".join(chunk))
            item.update(
                ok=True,
                content=content,
                start_line=start,
                end_line=end,
                total_lines=len(lines),
            )
            rendered.append(f"--- {pth} [{start}-{end} of {len(lines)}] ---\n{content}")
        results.append(item)
    return {
        "ok": all(item.get("ok") for item in results) if results else False,
        "files": results,
        "formatted": "\n\n".join(rendered) or "(no files read)",
    }


# ------------------------------------------------------------ search_codebase
def normalise_queries(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(q) for q in raw if str(q).strip()]
    if isinstance(raw, dict):
        return normalise_queries(raw.get("queries", raw.get("query")))
    return []


def search_codebase(raw: Any, cwd: str) -> Dict[str, Any]:
    queries = normalise_queries(raw)[:25]
    results: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for query in queries:
        cmd = [
            "grep", "-RInE", "--exclude-dir=.git", "--exclude-dir=node_modules",
            "--exclude-dir=.venv", "--exclude-dir=__pycache__", "--", query, ".",
        ]
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=45)
            output = _clamp((proc.stdout or proc.stderr or "").strip() or "(no matches)")
            error = None if proc.returncode in (0, 1) else f"grep exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            output, error = "(timed out after 45s)", "timeout"
        except Exception as exc:  # noqa: BLE001
            output, error = "", str(exc)
        count = len([line for line in output.splitlines() if line and not line.startswith("...[truncated")])
        results.append({"query": query, "output": output, "matches": count, "error": error})
        rendered.append(f"--- query: {query} ({count} lines) ---\n{output}")
    return {
        "ok": bool(queries) and all(r.get("error") is None for r in results),
        "results": results,
        "formatted": "\n\n".join(rendered) or "(no queries)",
    }


# --------------------------------------------------------------- run_commands
def normalise_commands(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            out.extend(normalise_commands(item))
        return out
    if isinstance(raw, dict):
        if "commands" in raw:
            return normalise_commands(raw["commands"])
        command = raw.get("command", raw.get("cmd"))
        if command is None:
            return []
        args = raw.get("args") or []
        if args:
            return [shlex.join([str(command)] + [str(a) for a in _list(args)])]
        return [str(command)]
    return []


def run_commands(raw: Any, cwd: str) -> Dict[str, Any]:
    commands = normalise_commands(raw)[:20]
    results: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for command in commands:
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command, shell=True, cwd=cwd, capture_output=True, text=True,
                timeout=300, executable="/bin/bash",
            )
            output = _clamp(((proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")).strip())
            exit_code = proc.returncode
            error = None
        except subprocess.TimeoutExpired:
            output, exit_code, error = "(timed out after 300s)", -1, "timeout"
        except Exception as exc:  # noqa: BLE001
            output, exit_code, error = str(exc), -1, str(exc)
        duration = max(0.0, time.perf_counter() - started)
        results.append({
            "command": command, "exit_code": exit_code,
            "output": output or "(no output)", "error": error,
            "duration": round(duration, 3),
        })
        rendered.append(f"--- $ {command} (exit {exit_code}, {duration:.2f}s) ---\n{output or '(no output)'}")
    return {
        "ok": bool(commands) and all(r["exit_code"] == 0 for r in results),
        "commands": results,
        "formatted": "\n\n".join(rendered) or "(no commands)",
    }


# ---------------------------------------------------------- fetch_web_content
def normalise_web_requests(raw: Any) -> List[Dict[str, str]]:
    if isinstance(raw, str):
        text = raw.strip()
        # Models frequently double-encode this argument: the JSON array/object
        # arrives as a *string* (e.g. '[{"url": "https://..."}]'). Without this
        # decode the whole blob is treated as a URL and every fetch fails with
        # "only http:// and https:// URLs are supported" - a mistake the agent
        # then repeats for the rest of the turn.
        if text.startswith(("[", "{")):
            try:
                decoded = json.loads(text)
                if isinstance(decoded, (list, dict)):
                    return normalise_web_requests(decoded)
            except (ValueError, TypeError):
                pass
        return [{"url": text, "prompt": ""}]
    if isinstance(raw, list):
        out: List[Dict[str, str]] = []
        for item in raw:
            out.extend(normalise_web_requests(item))
        return out
    if isinstance(raw, dict):
        if "requests" in raw:
            return normalise_web_requests(raw["requests"])
        if raw.get("url"):
            return [{"url": str(raw["url"]), "prompt": str(raw.get("prompt") or "")}]
    return []


def _html_to_text(markup: str) -> str:
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def fetch_web_content(raw: Any) -> Dict[str, Any]:
    """Fetch URLs and return cleaned text plus the analysis prompt.

    The calling agent receives the prompt alongside the content, so it can
    perform the extraction/analysis itself without a nested model call.
    """
    # De-duplicate identical URLs in a single batch - the model sometimes asks
    # for the same page twice with different prompts.
    seen: set[str] = set()
    unique: List[Dict[str, str]] = []
    for req in normalise_web_requests(raw):
        key = req["url"].strip().rstrip("/")
        if key and key not in seen:
            seen.add(key)
            unique.append(req)
    web_requests = unique[:10]
    results: List[Dict[str, Any]] = []
    rendered: List[str] = []
    for req in web_requests:
        url = req["url"]
        prompt = req.get("prompt") or ""
        item: Dict[str, Any] = {"url": url, "prompt": prompt}
        if not re.match(r"^https?://", url, re.IGNORECASE):
            item.update(ok=False, error="only http:// and https:// URLs are supported")
        else:
            try:
                response = requests.get(
                    url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT},
                    allow_redirects=True,
                )
                content_type = response.headers.get("content-type", "")
                if "html" in content_type.lower() or "<html" in response.text[:2000].lower():
                    text = _html_to_text(response.text)
                else:
                    text = response.text
                item.update(
                    ok=response.ok,
                    status_code=response.status_code,
                    final_url=response.url,
                    content_type=content_type,
                    content=_clamp(text),
                    bytes=len(response.content),
                )
                if not response.ok:
                    item["error"] = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001
                item.update(ok=False, error=str(exc))
        if item.get("ok"):
            rendered.append(
                f"--- {url} ---\nAnalysis prompt: {prompt or '(summarise the page)'}\n\n{item['content']}"
            )
        else:
            item["hint"] = (
                "Verify the URL, try another source, use web_search, or use shell with curl. "
                "Do not repeat the identical fetch."
            )
            rendered.append(f"--- {url} ---\nERROR: {item.get('error')}\n{item['hint']}")
        results.append(item)
    failed = [r for r in results if not r.get("ok")]
    payload: Dict[str, Any] = {
        "ok": bool(web_requests) and not failed,
        "requests": results,
        "formatted": "\n\n".join(rendered) or "(no URLs)",
    }
    if failed:
        payload["hint"] = (
            f"{len(failed)} fetch(es) failed. Change the URL(s) or use web_search / shell curl "
            "instead of retrying the identical request."
        )
    return payload


# ------------------------------------------------------------------ editor
def editor(
    path: str,
    new_text: str = "",
    old_text: Optional[str] = None,
    insert_line: Optional[int] = None,
    cwd: str = "",
) -> Dict[str, Any]:
    cwd = cwd or os.getcwd()
    pth = _path(cwd, path)
    if not pth.exists():
        return {"ok": False, "error": f"not found: {pth}", "path": str(pth)}
    before = pth.read_text(errors="replace")
    if old_text is not None:
        if not old_text:
            return {"ok": False, "error": "old_text is empty; use insert_line to insert", "path": str(pth)}
        if old_text not in before:
            return {"ok": False, "error": "old_text not found in file", "path": str(pth)}
        after = before.replace(old_text, new_text or "", 1)
        operation = "replace"
    elif insert_line is not None:
        lines = before.splitlines(keepends=True)
        index = max(0, min(len(lines), int(insert_line) - 1))
        insertion = new_text if new_text.endswith("\n") or not lines else new_text + "\n"
        lines.insert(index, insertion)
        after = "".join(lines)
        operation = f"insert@{insert_line}"
    else:
        return {"ok": False, "error": "provide old_text (replace) or insert_line (insert)", "path": str(pth)}
    pth.write_text(after)
    diff = "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=str(pth), tofile=str(pth), n=3,
    ))
    return {"ok": True, "path": str(pth), "operation": operation, "diff": _clamp(diff)}


# --------------------------------------------------------------- apply_patch
def _parse_patch(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if line.strip() == "*** Begin Patch":
            start = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "*** End Patch":
            end = i
            break
    ops: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in lines[start:end]:
        stripped = line.strip()
        for prefix, action in (
            ("*** Update File:", "update"),
            ("*** Add File:", "add"),
            ("*** Delete File:", "delete"),
        ):
            if stripped.startswith(prefix):
                current = {"action": action, "path": stripped[len(prefix):].strip(), "hunks": [[]], "move_to": None}
                ops.append(current)
                break
        else:
            if stripped.startswith("*** Move to:"):
                if current is not None:
                    current["move_to"] = stripped[len("*** Move to:"):].strip()
            elif stripped.startswith("@@") or stripped == "*** End of File":
                if current is not None:
                    current["hunks"].append([])
            elif current is not None:
                if not current["hunks"]:
                    current["hunks"].append([])
                current["hunks"][-1].append(line)
    return [op for op in ops if op["path"]]


def _apply_hunks(original: str, hunks: List[List[str]], path: str) -> str:
    result = original
    for hunk in hunks:
        old_lines: List[str] = []
        new_lines: List[str] = []
        for line in hunk:
            if line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                old_lines.append(line[1:])
            else:
                content = line[1:] if line.startswith(" ") else line
                old_lines.append(content)
                new_lines.append(content)
        old_block = "\n".join(old_lines)
        new_block = "\n".join(new_lines)
        if old_block:
            if old_block not in result:
                raise ValueError(f"patch context not found in {path}: {old_block[:120]!r}")
            result = result.replace(old_block, new_block, 1)
        elif new_block:
            result = result + ("\n" if result and not result.endswith("\n") else "") + new_block + "\n"
    return result


def apply_patch(text: str, cwd: str = "") -> Dict[str, Any]:
    """Parse and atomically apply a Codex/Cline style ``*** Begin Patch`` block."""
    cwd = cwd or os.getcwd()
    try:
        ops = _parse_patch(text)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"invalid patch: {exc}", "changes": []}
    if not ops:
        return {"ok": False, "error": "no file operations found (expected *** Begin Patch)", "changes": []}

    pending: List[Dict[str, Any]] = []
    for op in ops:
        pth = _path(cwd, op["path"])
        action = op["action"]
        if action == "add":
            if pth.exists():
                return {"ok": False, "error": f"cannot add existing file: {pth}", "changes": []}
            added: List[str] = []
            for hunk in op["hunks"]:
                for line in hunk:
                    if line.startswith("+"):
                        added.append(line[1:])
                    elif line.strip() and not line.startswith("@@"):
                        return {"ok": False, "error": f"Add File only accepts + lines: {pth}", "changes": []}
            pending.append({"path": pth, "action": "add", "before": "", "after": "\n".join(added) + ("\n" if added else ""), "move_to": None})
        elif action == "delete":
            if not pth.exists():
                return {"ok": False, "error": f"cannot delete missing file: {pth}", "changes": []}
            pending.append({"path": pth, "action": "delete", "before": pth.read_text(errors="replace"), "after": "", "move_to": None})
        else:
            if not pth.exists():
                return {"ok": False, "error": f"cannot update missing file: {pth}", "changes": []}
            before = pth.read_text(errors="replace")
            try:
                after = _apply_hunks(before, op["hunks"], str(pth))
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "changes": []}
            move_to = _path(cwd, op["move_to"]) if op.get("move_to") else None
            pending.append({"path": pth, "action": "update", "before": before, "after": after, "move_to": move_to})

    changes: List[Dict[str, Any]] = []
    for item in pending:  # all validation passed; now write
        pth: Path = item["path"]
        if item["action"] == "delete":
            pth.unlink(missing_ok=True)
        else:
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(item["after"])
        target = item.get("move_to")
        if target is not None and target != pth:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["after"])
            pth.unlink(missing_ok=True)
        diff = "".join(difflib.unified_diff(
            item["before"].splitlines(True), item["after"].splitlines(True),
            fromfile=str(pth), tofile=str(target or pth), n=3,
        ))
        changes.append({
            "path": str(target or pth),
            "action": item["action"] + (" + move" if target is not None else ""),
            "diff": _clamp(diff),
        })
    return {
        "ok": True,
        "changes": changes,
        "formatted": "\n\n".join(f"--- {c['path']} ({c['action']}) ---\n{c['diff']}" for c in changes),
    }
