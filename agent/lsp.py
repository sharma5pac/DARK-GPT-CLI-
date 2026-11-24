"""Language Server Protocol integration for precision code intelligence.

Talks real LSP JSON-RPC over stdio to pyright / typescript-language-server /
rust-analyzer / gopls / clangd, giving definitions, references, diagnostics and
semantic-safe rename. Everything is bounded by hard timeouts and *always*
degrades to the dependency-free regex/AST fallback in ``agent.codeintel`` - a
missing or broken server can never crash the agent.

Design notes
------------
* A single client instance handles one request "session": spawn -> initialize
  -> didOpen -> request(s) -> shutdown. This keeps state isolated and avoids
  leaking long-lived server processes.
* A reader thread parses Content-Length framed messages into a queue, so LSP
  notifications (publishDiagnostics, logMessage) are collected while waiting
  for a response and never deadlock the request path.
* Server selection is detection-based: install a server later and it is picked
  up automatically, no code change needed.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_RESULTS = 80
DEFAULT_TIMEOUT = 6.0

# Extension -> ordered list of (command argv, language id). First available wins.
SERVERS: Dict[str, List[Tuple[List[str], str]]] = {
    ".py": [
        (["pyright-langserver", "--stdio"], "python"),
        (["basedpyright-langserver", "--stdio"], "python"),
        (["jedi-language-server"], "python"),
        (["pylsp"], "python"),
    ],
    ".ts": [(["typescript-language-server", "--stdio"], "typescript")],
    ".tsx": [(["typescript-language-server", "--stdio"], "typescriptreact")],
    ".js": [(["typescript-language-server", "--stdio"], "javascript")],
    ".jsx": [(["typescript-language-server", "--stdio"], "javascriptreact")],
    ".mjs": [(["typescript-language-server", "--stdio"], "javascript")],
    ".cjs": [(["typescript-language-server", "--stdio"], "javascript")],
    ".rs": [(["rust-analyzer"], "rust")],
    ".go": [(["gopls"], "go")],
    ".c": [(["clangd"], "c")],
    ".h": [(["clangd"], "c")],
    ".cpp": [(["clangd"], "cpp")],
    ".cc": [(["clangd"], "cpp")],
    ".hpp": [(["clangd"], "cpp")],
}

# CLI diagnostics used when there is no language server, or as a second source.
CLI_DIAGNOSTICS: Dict[str, List[List[str]]] = {
    ".py": [["pyright", "--outputjson"]],
    ".ts": [["tsc", "--noEmit", "--pretty", "false"]],
    ".tsx": [["tsc", "--noEmit", "--pretty", "false"]],
    ".rs": [["cargo", "check", "--message-format", "short"]],
    ".go": [["go", "vet", "./..."]],
}


class LSPError(Exception):
    """Raised for transport/protocol problems; callers fall back, never crash."""


def _uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def server_for(path: str) -> Optional[Tuple[List[str], str]]:
    """Return the first installed language server for ``path``'s extension."""
    suffix = Path(path).suffix.lower()
    for cmd, lang in SERVERS.get(suffix, []):
        if shutil.which(cmd[0]):
            return cmd, lang
    return None


def has_server(path: str) -> bool:
    return server_for(path) is not None


def _read_message(stream: Any) -> Optional[Dict[str, Any]]:
    headers: Dict[bytes, bytes] = {}
    while True:
        line = stream.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        if b":" in line:
            key, value = line.split(b":", 1)
            headers[key.strip().lower()] = value.strip()
    try:
        length = int(headers.get(b"content-length", b"0"))
    except ValueError:
        return None
    if length <= 0 or length > 16 * 1024 * 1024:
        return None
    body = stream.read(length)
    if not body:
        return None
    try:
        value = json.loads(body.decode("utf-8", "replace"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


class LSPClient:
    """Minimal synchronous LSP client. Use as a context manager."""

    def __init__(self, command: List[str], cwd: str, language_id: str, timeout: float = DEFAULT_TIMEOUT):
        self.command = command
        self.cwd = cwd
        self.language_id = language_id
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._inbox: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self.notifications: List[Dict[str, Any]] = []
        self._reader: Optional[threading.Thread] = None

    def __enter__(self) -> "LSPClient":
        try:
            self.proc = subprocess.Popen(
                self.command,
                cwd=self.cwd or os.getcwd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:  # noqa: BLE001
            raise LSPError(f"cannot start {self.command[0]}: {exc}") from exc
        self._reader = threading.Thread(target=self._pump, daemon=True, name="darkgpt-lsp")
        self._reader.start()
        self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": _uri(self.cwd or os.getcwd()),
            "capabilities": {},
        })
        self.notify("initialized", {})
        return self

    def __exit__(self, *_exc: Any) -> None:
        try:
            self.request("shutdown", None, timeout=2.0)
        except Exception:
            pass
        self.notify("exit", None)
        proc = self.proc
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        while True:
            try:
                message = _read_message(self.proc.stdout)
            except Exception:
                message = None
            if message is None:
                self._inbox.put(None)
                return
            self._inbox.put(message)

    def _send(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise LSPError("server not running")
        raw = json.dumps(payload).encode("utf-8")
        try:
            self.proc.stdin.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
            self.proc.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            raise LSPError(f"write failed: {exc}") from exc

    def notify(self, method: str, params: Any) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Any, timeout: Optional[float] = None) -> Dict[str, Any]:
        self._id += 1
        request_id = self._id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LSPError(f"{method} timed out after {timeout or self.timeout}s")
            try:
                message = self._inbox.get(timeout=remaining)
            except queue.Empty:
                raise LSPError(f"{method} timed out after {timeout or self.timeout}s") from None
            if message is None:
                raise LSPError(f"server closed during {method}")
            if "id" not in message and "method" in message:
                self.notifications.append(message)
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise LSPError(f"{method}: {message['error']}")
                return message.get("result") or {}

    def open_document(self, path: str, text: str) -> str:
        uri = _uri(path)
        self.notify("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": self.language_id, "version": 1, "text": text},
        })
        return uri

    @staticmethod
    def _locations(result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            result = [result]
        out: List[Dict[str, Any]] = []
        for item in (result or [])[:MAX_RESULTS]:
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or item.get("targetUri") or "")
            rng = item.get("range") or item.get("targetSelectionRange") or {}
            start = rng.get("start") or {}
            out.append({
                "uri": uri,
                "path": _path_from_uri(uri),
                "line": int(start.get("line", 0)) + 1,
                "character": int(start.get("character", 0)),
            })
        return out

    def definition(self, uri: str, line: int, character: int) -> List[Dict[str, Any]]:
        result = self.request("textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": max(0, line - 1), "character": max(0, character)},
        })
        return self._locations(result)

    def references(self, uri: str, line: int, character: int) -> List[Dict[str, Any]]:
        result = self.request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": max(0, line - 1), "character": max(0, character)},
            "context": {"includeDeclaration": True},
        })
        return self._locations(result)

    def rename(self, uri: str, line: int, character: int, new_name: str) -> Dict[str, Any]:
        return self.request("textDocument/rename", {
            "textDocument": {"uri": uri},
            "position": {"line": max(0, line - 1), "character": max(0, character)},
            "newName": new_name,
        }) or {}

    def diagnostics(self, uri: str, wait: float = 0.8) -> List[Dict[str, Any]]:
        """Collect publishDiagnostics notifications (and LSP 3.17 pull diagnostics)."""
        end = time.monotonic() + wait
        while time.monotonic() < end:
            try:
                msg = self._inbox.get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                break
            if msg is None:
                break
            if "method" in msg:
                self.notifications.append(msg)
        found: List[Dict[str, Any]] = []
        for msg in self.notifications:
            if msg.get("method") != "textDocument/publishDiagnostics":
                continue
            params = msg.get("params") or {}
            if params.get("uri") != uri:
                continue
            for item in (params.get("diagnostics") or [])[:MAX_RESULTS]:
                rng = item.get("range") or {}
                start = rng.get("start") or {}
                found.append({
                    "severity": int(item.get("severity", 3)),
                    "message": str(item.get("message") or "")[:600],
                    "line": int(start.get("line", 0)) + 1,
                    "character": int(start.get("character", 0)),
                    "code": item.get("code"),
                })
        return found


def _path_from_uri(uri: str) -> str:
    try:
        from urllib.parse import unquote, urlparse

        parsed = urlparse(uri)
        if parsed.scheme == "file":
            return unquote(parsed.path)
    except Exception:
        pass
    return uri


def _position(text: str, symbol: str) -> Tuple[int, int]:
    """Find a 1-based line / 0-based character for a symbol, preferring its definition."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        col = line.find(symbol)
        if col < 0:
            continue
        # Skip pure import lines so we land on a real declaration/reference.
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "use ", "pub use ")) and idx < len(lines) - 1:
            continue
        return idx + 1, col
    return 1, 0


def _with_client(path: str, cwd: str, operation, timeout: float = DEFAULT_TIMEOUT):
    """Run ``operation(client, uri, text)`` or raise LSPError."""
    info = server_for(path)
    if not info:
        raise LSPError(f"no language server for {Path(path).suffix or path}")
    command, language_id = info
    target = str(Path(cwd) / path) if not os.path.isabs(path) else path
    if not Path(target).is_file():
        raise LSPError(f"not found: {target}")
    try:
        text = Path(target).read_text(errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise LSPError(f"cannot read {target}: {exc}") from exc
    with LSPClient(command, cwd, language_id, timeout=timeout) as client:
        uri = client.open_document(target, text)
        return operation(client, uri, text)


def goto_definition(path: str, symbol: str, cwd: str = "", line: int = 0, character: int = 0) -> Dict[str, Any]:
    cwd = cwd or os.getcwd()
    def op(client, uri, text):
        ln, ch = (line, character) if line else _position(text, symbol)
        return {"symbol": symbol, "definitions": client.definition(uri, ln, ch)}
    try:
        result = _with_client(path, cwd, op)
        if not result.get("definitions"):
            raise LSPError("no definitions returned")
        return result
    except Exception as exc:  # noqa: BLE001
        from agent import codeintel

        refs = codeintel.find_refs(symbol, cwd, path)
        return {"symbol": symbol, "definitions": refs.get("matches", [])[:MAX_RESULTS], "fallback": True, "note": str(exc)[:200]}


def find_references(path: str, symbol: str, cwd: str = "", line: int = 0, character: int = 0) -> Dict[str, Any]:
    cwd = cwd or os.getcwd()
    def op(client, uri, text):
        ln, ch = (line, character) if line else _position(text, symbol)
        return {"symbol": symbol, "references": client.references(uri, ln, ch)}
    try:
        return _with_client(path, cwd, op)
    except Exception as exc:  # noqa: BLE001
        from agent import codeintel

        refs = codeintel.find_refs(symbol, cwd, path)
        return {"symbol": symbol, "references": refs.get("matches", [])[:MAX_RESULTS], "fallback": True, "note": str(exc)[:200]}


def diagnostics(path: str, cwd: str = "") -> Dict[str, Any]:
    cwd = cwd or os.getcwd()
    target = str(Path(cwd) / path) if not os.path.isabs(path) else path
    suffix = Path(target).suffix.lower()

    def op(client, uri, _text):
        items = client.diagnostics(uri)
        return {"path": target, "diagnostics": items, "source": client.command[0]}

    try:
        result = _with_client(path, cwd, op)
        if result.get("diagnostics"):
            return result
    except Exception:
        pass
    # CLI diagnostics (pyright --outputjson, tsc, cargo check, go vet).
    for cmd in CLI_DIAGNOSTICS.get(suffix, []):
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=45, errors="replace")
            output = ((proc.stdout or "") + (proc.stderr or "")).strip()
            if output:
                return {"path": target, "diagnostics": [{"severity": 1 if proc.returncode else 3, "message": output[:6000], "line": 0, "character": 0}], "source": cmd[0]}
        except Exception:
            continue
    return {"path": target, "diagnostics": [], "source": "none", "note": "no diagnostics source available"}


def rename_symbol(path: str, symbol: str, new_name: str, cwd: str = "", line: int = 0, character: int = 0, apply: bool = False) -> Dict[str, Any]:
    """Semantic rename. With ``apply=False`` returns the planned edits only."""
    cwd = cwd or os.getcwd()
    def op(client, uri, text):
        ln, ch = (line, character) if line else _position(text, symbol)
        edit = client.rename(uri, ln, ch, new_name)
        changes = edit.get("changes") or {}
        document_changes = edit.get("documentChanges") or []
        for change in document_changes:
            if not isinstance(change, dict):
                continue
            doc = change.get("textDocument") or {}
            uri2 = str(doc.get("uri") or "")
            edits = change.get("edits") or []
            mapped = changes.setdefault(uri2, [])
            for item in edits:
                rng = item.get("range") or {}
                mapped.append({"range": rng, "newText": item.get("newText", "")})
        applied: List[Dict[str, Any]] = []
        if apply:
            for uri2, edits in changes.items():
                dest = _path_from_uri(uri2)
                if not dest or not Path(dest).is_file():
                    continue
                source = Path(dest).read_text(errors="replace")
                lines = source.splitlines(True)
                # Apply edits from the end so offsets stay valid.
                ordered = sorted(edits, key=lambda e: (e.get("range", {}).get("start", {}).get("line", 0), e.get("range", {}).get("start", {}).get("character", 0)), reverse=True)
                for item in ordered:
                    rng = item.get("range") or {}
                    s = rng.get("start") or {}
                    e = rng.get("end") or {}
                    sl, sc = int(s.get("line", 0)), int(s.get("character", 0))
                    el, ec = int(e.get("line", sl)), int(e.get("character", sc))
                    if sl >= len(lines) or el >= len(lines):
                        continue
                    if sl == el:
                        row = lines[sl]
                        lines[sl] = row[:sc] + str(item.get("newText") or "") + row[ec:]
                    else:
                        lines[sl] = lines[sl][:sc] + str(item.get("newText") or "") + "\n"
                        del lines[sl + 1:el + 1]
                Path(dest).write_text("".join(lines))
                applied.append({"path": dest, "edits": len(edits)})
        return {"from": symbol, "to": new_name, "changes": changes, "applied": applied, "apply": bool(apply), "source": client.command[0]}
    try:
        return _with_client(path, cwd, op)
    except Exception as exc:  # noqa: BLE001
        from agent import codeintel

        preview = codeintel.rename(symbol, new_name, cwd, path, dry_run=not apply)
        preview["fallback"] = True
        preview["note"] = str(exc)[:200]
        return preview
