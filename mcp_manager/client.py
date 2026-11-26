"""Async MCP host/client manager.

Owns the real lifecycle of every configured MCP server: spawn (stdio) or
connect (SSE / streamable HTTP), initialize the session, discover tools,
resources and prompts, route ``tools/call`` requests, capture stderr, and
support stop / restart / add / remove from the TUI.
"""

from __future__ import annotations

import asyncio
import json
import os
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python < 3.11
    BaseExceptionGroup  # type: ignore[used-before-def]
except NameError:  # pragma: no cover
    from exceptiongroup import BaseExceptionGroup  # type: ignore[no-redef]

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils import agentic_guard, appconfig

try:
    from mcp.client.sse import sse_client
except Exception:  # pragma: no cover
    sse_client = None

try:
    from mcp.client.streamable_http import (
        create_mcp_http_client,
        streamable_http_client,
    )
except Exception:  # pragma: no cover
    streamable_http_client = None
    create_mcp_http_client = None

LOG_DIR = Path("/tmp")


@dataclass
class MCPTool:
    server: str
    name: str
    description: str
    input_schema: Dict[str, Any]

    @property
    def qualified(self) -> str:
        return f"mcp__{self.server}__{self.name}"

    @property
    def display(self) -> str:
        return f"{self.server}:{self.name}"


@dataclass
class MCPServerState:
    name: str
    config: Dict[str, Any]
    status: str = "disconnected"
    error: str = ""
    transport: str = "stdio"
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    guard_findings: List[Dict[str, Any]] = field(default_factory=list)
    quarantined: List[str] = field(default_factory=list)
    session: Optional[ClientSession] = None
    log_path: str = ""
    _task: Optional[asyncio.Task] = None
    _ready: asyncio.Event = field(default_factory=asyncio.Event)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _call_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    @property
    def tool_count(self) -> int:
        return len(self.tools)


class MCPManager:
    def __init__(self) -> None:
        self.servers: Dict[str, MCPServerState] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- lifecycle
    async def start_from_config(self, config: Dict[str, Any]) -> None:
        from .config import servers

        for name, cfg in servers(config).items():
            if cfg.get("disabled"):
                self.servers[name] = MCPServerState(name=name, config=cfg, status="disabled")
                continue
            await self.start_server(name, cfg)

    async def start_server(self, name: str, cfg: Dict[str, Any]) -> MCPServerState:
        await self.stop_server(name, quiet=True)
        async with self._lock:
            state = MCPServerState(
                name=name,
                config=cfg,
                status="connecting",
                transport=self._transport_of(cfg),
                log_path=str(LOG_DIR / f"darkgpt-mcp-{name}.log"),
            )
            self.servers[name] = state
            state._task = asyncio.create_task(self._run_server(state), name=f"mcp:{name}")
            return state

    async def stop_server(self, name: str, quiet: bool = False) -> None:
        state = self.servers.get(name)
        if not state:
            return
        state._stop.set()
        task = state._task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        state.session = None
        state.tools = []
        state.resources = []
        state.prompts = []
        state.guard_findings = []
        state.quarantined = []
        if not quiet:
            state.status = "disconnected"
            state.error = ""

    async def restart_server(self, name: str) -> None:
        state = self.servers.get(name)
        if not state:
            from .config import get_server

            cfg = get_server(name)
            if not cfg:
                raise KeyError(name)
            await self.start_server(name, cfg)
            return
        await self.start_server(name, dict(state.config))

    async def stop_all(self) -> None:
        for name in list(self.servers):
            await self.stop_server(name, quiet=True)

    async def add_server(self, name: str, cfg: Dict[str, Any], scope: str = "user") -> None:
        from .config import upsert_server

        upsert_server(name, cfg, scope=scope)
        await self.start_server(name, cfg)

    async def remove_server(self, name: str, scope: str = "user") -> None:
        from .config import remove_server

        await self.stop_server(name)
        self.servers.pop(name, None)
        remove_server(name, scope=scope)

    # ---------------------------------------------------------------- runners
    @staticmethod
    def _transport_of(cfg: Dict[str, Any]) -> str:
        if cfg.get("url"):
            return "sse" if cfg.get("transport") == "sse" else "http"
        return "stdio"

    async def _run_server(self, state: MCPServerState) -> None:
        cfg = state.config
        try:
            if cfg.get("command"):
                await self._run_stdio(state)
            elif cfg.get("url"):
                await self._run_http(state)
            else:
                state.status = "error"
                state.error = "missing command/url"
        except asyncio.CancelledError:
            state.status = "stopped"
            raise
        except Exception as e:  # noqa: BLE001
            state.status = "error"
            state.error = self._clean_error(e)
        finally:
            state._ready.set()

    @staticmethod
    def _clean_error(exc: BaseException) -> str:
        # mcp 2.x wraps failures in (nested) ExceptionGroups; surface the leaf.
        while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]
        text = str(exc) or exc.__class__.__name__
        if "ECONNREFUSED" in text or "Connection refused" in text:
            return "connection refused (is the server running?)"
        if "No such file" in text:
            return f"command not found: {text}"
        return text[:300]

    async def _run_stdio(self, state: MCPServerState) -> None:
        cfg = state.config
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in (cfg.get("env") or {}).items()})
        params = StdioServerParameters(
            command=cfg["command"],
            args=[str(a) for a in (cfg.get("args") or [])],
            env=env,
        )
        log_path = Path(state.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        try:
            async with stdio_client(params, errlog=log_file) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    state.session = session
                    await self._refresh_meta(state)
                    state.status = "connected"
                    state.error = ""
                    state._ready.set()
                    await self._serve_calls(state)
        finally:
            with contextlib.suppress(Exception):
                log_file.close()

    async def _run_http(self, state: MCPServerState) -> None:
        cfg = state.config
        url = cfg["url"]
        headers = cfg.get("headers") or {}
        if streamable_http_client is not None and cfg.get("transport") != "sse":
            http_client = None
            if headers and create_mcp_http_client is not None:
                http_client = create_mcp_http_client(headers=headers)
            try:
                async with streamable_http_client(url, http_client=http_client) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        state.session = session
                        await self._refresh_meta(state)
                        state.status = "connected"
                        state.error = ""
                        state._ready.set()
                        await self._serve_calls(state)
            finally:
                if http_client is not None:
                    with contextlib.suppress(Exception):
                        await http_client.aclose()
            return
        if sse_client is None:
            raise RuntimeError("SSE transport unavailable in mcp sdk")
        async with sse_client(url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                state.session = session
                await self._refresh_meta(state)
                state.status = "connected"
                state.error = ""
                state._ready.set()
                await self._serve_calls(state)

    async def _serve_calls(self, state: MCPServerState) -> None:
        while not state._stop.is_set():
            try:
                fut, name, arguments = await asyncio.wait_for(state._call_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                assert state.session is not None
                result = await state.session.call_tool(name, arguments)
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:  # noqa: BLE001
                if not fut.done():
                    fut.set_exception(e)

    async def _refresh_meta(self, state: MCPServerState) -> None:
        assert state.session is not None
        tools = await state.session.list_tools()
        state.tools = []
        state.guard_findings = []
        state.quarantined = []
        quarantine = bool(appconfig.get("mcp_poison_quarantine", True))
        for tool in tools.tools:
            raw_schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {"type": "object", "properties": {}}
            )
            # MCP spec alignment: every OpenAI-style tool schema must be an
            # object with a properties map, even for zero-argument tools.
            schema = dict(raw_schema) if isinstance(raw_schema, dict) else {}
            if schema.get("type") != "object":
                schema["type"] = "object"
            if not isinstance(schema.get("properties"), dict):
                schema["properties"] = {}
            item: MCPTool = MCPTool(
                server=state.name,
                name=str(tool.name or ""),
                description=str(tool.description or ""),
                input_schema=schema,
            )
            findings = agentic_guard.scan_tool_spec(
                {"type": "function", "function": {
                    "name": item.name,
                    "description": item.description,
                    "parameters": item.input_schema,
                }},
                source=state.name,
            )
            if findings:
                state.guard_findings.extend(findings)
                if quarantine and agentic_guard.is_poisoned(findings):
                    state.quarantined.append(item.name)
                    continue
            state.tools.append(item)
        state.resources = []
        state.prompts = []
        with contextlib.suppress(Exception):
            resources = await state.session.list_resources()
            state.resources = [str(getattr(r, "uri", r)) for r in resources.resources]
        with contextlib.suppress(Exception):
            prompts = await state.session.list_prompts()
            state.prompts = [p.name for p in prompts.prompts]

    # ------------------------------------------------------------------ query
    def list_tools(
        self,
        disabled: Optional[set[str]] = None,
        disabled_servers: Optional[set[str]] = None,
    ) -> List[MCPTool]:
        disabled = disabled or set()
        disabled_servers = disabled_servers or set()
        out: List[MCPTool] = []
        for state in self.servers.values():
            if state.name in disabled_servers:
                continue
            for tool in state.tools:
                if tool.name in state.quarantined:
                    continue
                if tool.qualified in disabled or tool.display in disabled:
                    continue
                out.append(tool)
        return out

    def openai_tools(
        self,
        disabled: Optional[set[str]] = None,
        disabled_servers: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.qualified,
                    "description": f"[MCP {tool.server}] {tool.description}",
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.list_tools(disabled, disabled_servers)
        ]

    def connected_count(self) -> int:
        return sum(1 for s in self.servers.values() if s.status == "connected")

    def total_count(self) -> int:
        return len(self.servers)

    def status_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for name, state in sorted(self.servers.items()):
            rows.append(
                {
                    "name": name,
                    "status": state.status,
                    "transport": state.transport,
                    "tools": state.tool_count,
                    "resources": len(state.resources),
                    "prompts": len(state.prompts),
                    "quarantined": list(state.quarantined),
                    "guard_findings": len(state.guard_findings),
                    "error": state.error,
                    "log": state.log_path,
                }
            )
        return rows

    def guard_summary(self) -> Dict[str, Any]:
        findings: List[Dict[str, Any]] = []
        quarantined: Dict[str, List[str]] = {}
        for name, state in self.servers.items():
            findings.extend(state.guard_findings)
            if state.quarantined:
                quarantined[name] = list(state.quarantined)
        return {"findings": findings, "quarantined": quarantined}

    def status_lines(self) -> List[str]:
        lines = []
        for row in self.status_rows():
            extra = f" {row['tools']} tools" if row["status"] == "connected" else (
                f" {row['error']}" if row["error"] else ""
            )
            if row["quarantined"]:
                extra += f" | quarantined: {len(row['quarantined'])}"
            lines.append(f"{row['name']}: {row['status']}{extra}")
        return lines or ["no mcp servers configured"]

    async def call_tool(self, server: str, tool: str, arguments: Dict[str, Any], timeout: float = 120.0) -> Any:
        state = self.servers.get(server)
        if not state:
            raise RuntimeError(f"unknown mcp server: {server}")
        if state.status != "connected":
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(state._ready.wait(), timeout=8)
        if state.status != "connected":
            raise RuntimeError(f"mcp server not connected: {server} ({state.status}) {state.error}")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await state._call_queue.put((fut, tool, arguments))
        return await asyncio.wait_for(fut, timeout=timeout)

    def log_tail(self, name: str, lines: int = 120) -> str:
        state = self.servers.get(name)
        path = Path(state.log_path) if state and state.log_path else LOG_DIR / f"darkgpt-mcp-{name}.log"
        try:
            content = path.read_text(errors="replace").splitlines()
            return "\n".join(content[-lines:]) or "(log is empty)"
        except FileNotFoundError:
            return f"(no log yet at {path})"
        except Exception as exc:  # noqa: BLE001
            return f"(could not read log: {exc})"

    def schema_text(self, server: str) -> str:
        state = self.servers.get(server)
        if not state:
            return f"unknown server: {server}"
        if not state.tools:
            return f"{server}: no tools discovered (status={state.status})"
        parts = []
        for tool in state.tools:
            parts.append(f"{tool.name}\n  {tool.description or '(no description)'}")
            parts.append("  " + json.dumps(tool.input_schema, indent=2).replace("\n", "\n  "))
        return "\n".join(parts)
