from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Markdown, OptionList, Static, TextArea

# Ensure markdown responses are selectable/copyable in the TUI.
try:
    Markdown.ALLOW_SELECT = True
except Exception:
    pass
from textual.widgets.option_list import Option

from agent import tokens as token_util
from agent import jobs
from agent.core import DEFAULT_MODEL, SYSTEM, ToolRegistry, run_agent
from agent.skills import SkillRegistry
from mcp_manager.client import MCPManager
from mcp_manager.config import ensure_default_config, load_config
from utils import appconfig, clipboard, hooks, permissions, scripting, undo
from agent import context as context_util, lsp as lsp_util, memory as memory_util, routing, rules as rules_util, session as session_util
from utils import history as db
from utils.get_response import (
    PROJECT_ROOT,
    PROVIDERS,
    current_provider,
    get_api_key,
    get_catalog,
    provider_config,
)
from utils.local_models import discover_local_models

APP_VERSION = "2.2"

COMMANDS: List[tuple[str, str]] = [
    ("/model", "Choose a model for the active provider (/provider first)"),
    ("/provider", "Switch chat provider: /provider openrouter|nvidia"),
    ("/mcp", "Manage MCP servers, tools and logs"),
    ("/skills", "Toggle, create or edit skills"),
    ("/tools", "Enable or disable built-in and MCP tools"),
    ("/new", "Start a new conversation"),
    ("/clear", "Clear the current conversation"),
    ("/copy", "Copy the last assistant answer"),
    ("/copy-all", "Copy the entire conversation including tool findings"),
    ("/retry", "Re-run the last prompt (or /retry force)"),
    ("/pending", "Show or clear prompts queued while a turn was running"),
    ("/status", "Show model, MCP, tokens and 402 cooldown"),
    ("/jobs", "List background tasks/jobs (/jobs <id> for log, /jobs kill <id>)"),
    ("/budget", "Show/set tool/step budget (/budget tools 60, /budget steps 40)"),
    ("/steps", "Alias for /budget steps <n>"),
    ("/proxy", "Proxy control (/proxy status|start|stop|import)"),
    ("/verify-findings", "Replay finding repros and mark fixed/still_vulnerable"),
    ("/bb", "Blackboard summary/post (/bb, /bb post intent|fact|hint <text>)"),
    ("/attack-paths", "Rank multi-hop attack paths from knowledge graph"),
    ("/mcp-verbose", "Toggle background MCP display (shown/hidden cards)"),
    ("/bg", "Show background tool findings (/bg <n> for full result of one call)"),
    ("/clipboard", "Show clipboard backends and QTerminal/X11/Wayland notes"),
    ("/diff", "Show the last edit diff"),
    ("/undo", "Restore files from the last edit snapshot"),
    ("/verify", "Toggle automatic post-edit tests/lint"),
    ("/plan", "Toggle cheap-model planning before the turn"),
    ("/review", "Toggle independent diff review"),
    ("/worktree", "Toggle isolated worktree checks during review"),
    ("/lsp", "Show/toggle LSP servers (definitions, references, diagnostics, rename)"),
    ("/subagents", "Toggle isolated subagents (explore/review/test workers)"),
    ("/delegates", "Show subagent reports from the last turn"),
    ("/rules", "Show project rules (/rules add <text> to append)"),
    ("/compact", "Compact older turns into a summary now"),
    ("/permissions", "Show/set permission mode (full|ask|strict)"),
    ("/hooks", "Show configured pre/post tool hooks"),
    ("/react", "Show/set ReAct fallback (/react on|off) — native tools are retried first"),
    ("/scripts", "List self-written helper scripts (/scripts show <path>)"),
    ("/checkpoint", "Save a session checkpoint (/checkpoint <label>)"),
    ("/resume", "Resume the latest or a chosen session (/resume <id>)"),
    ("/sessions", "List saved session checkpoints"),
    ("/memory", "Toggle/show project memory"),
    ("/lessons", "Show self-learned lessons (/lessons clear to forget them)"),
    ("/patterns", "Show recurring patterns the agent has learned (/patterns clear)"),
    ("/learn", "Toggle self-learning (lessons + pattern capture)"),
    ("/target", "Set/show active target for knowledge graph + evidence (/target <name>)"),
    ("/targets", "List known targets"),
    ("/kg", "Knowledge graph: /kg [hosts|services|endpoints|findings|query <t>|chain <t>|export json|dot]"),
    ("/evidence", "Show evidence ledger summary for active target"),
    ("/report", "Generate report: /report md|html|json|csv [out]"),
    ("/recon", "Run parallel recon orchestration for active target (/recon [focus])"),
    ("/route", "Set or clear the cheap routing model (/route model/id)"),
    ("/power", "Show all coding-power settings"),
    ("/cwd", "Show the working directory"),
    ("/exit", "Quit DarkGPT"),
]

PHASE_LABELS = {
    "connecting": "Connecting / waiting for first token",
    "thinking": "Thinking",
    "writing": "Writing response",
    "tool": "Tool",
}


# --------------------------------------------------------------------------- widgets


class PhaseTimeline(Static):
    ALLOW_SELECT = True

    """Live phase timeline driven by perf_counter events from the agent."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: List[Dict[str, Any]] = []
        self._started = time.perf_counter()

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self.tick)  # keep a strong ref (WeakSet in Textual)
        self.render_timeline()

    def start(self, phase: str, label: Optional[str] = None) -> int:
        idx = len(self.steps)
        self.steps.append(
            {
                "phase": phase,
                "label": label or PHASE_LABELS.get(phase, phase),
                "status": "active",
                "started": time.perf_counter(),
                "elapsed": 0.0,
            }
        )
        self.render_timeline()
        return idx

    def end(self, phase: str, duration: Optional[float] = None, ok: bool = True, label: Optional[str] = None) -> None:
        for step in reversed(self.steps):
            if step["phase"] == phase and step["status"] == "active":
                step["status"] = "done" if ok else "error"
                step["elapsed"] = float(duration) if duration is not None else (time.perf_counter() - step["started"])
                if label:
                    step["label"] = label
                break
        self.render_timeline()

    def finish_all(self, ok: bool = True) -> None:
        for step in self.steps:
            if step["status"] == "active":
                step["status"] = "done" if ok else "error"
                step["elapsed"] = time.perf_counter() - step["started"]
        self.render_timeline()

    def tick(self) -> None:
        if any(step["status"] == "active" for step in self.steps):
            self.render_timeline()

    def render_timeline(self) -> None:
        total = time.perf_counter() - self._started
        text = Text()
        text.append(f"Activity  {total:.1f}s\n", style="bold #8b949e")
        for step in self.steps[-12:]:
            elapsed = step["elapsed"] if step["status"] != "active" else (time.perf_counter() - step["started"])
            if step["status"] == "done":
                mark, style = "✓", "#6abf69"
            elif step["status"] == "error":
                mark, style = "✗", "#e06c75"
            else:
                mark, style = "●", "#d4a359"
            text.append(f" {mark} ", style=style)
            text.append(f"{step['label']}", style="#e5e5e5" if step["status"] == "active" else "#c9d1d9")
            text.append(f"  {elapsed:.1f}s\n", style="#777777")
        self.update(text)


class ThoughtBlock(Static):
    ALLOW_SELECT = True

    can_focus = True
    BINDINGS = [Binding("enter", "toggle", "Toggle thought", show=False)]

    def __init__(self, text: str = "", seconds: float = 0.0) -> None:
        super().__init__()
        self.thought = text
        self.seconds = seconds
        self.expanded = False

    def on_mount(self) -> None:
        self.render_block()

    def set_thought(self, text: str, seconds: float) -> None:
        self.thought = text
        self.seconds = seconds
        self.render_block()

    def render_block(self) -> None:
        summary = (self.thought.strip().splitlines() or ["reasoning"])[0][:90]
        head = f"+ Thought: {summary} · {self.seconds:.1f}s"
        if self.expanded and self.thought.strip():
            self.update(Text(head + "\n\n" + self.thought.strip(), style="#d4a359"))
        else:
            self.update(Text(head, style="#d4a359"))

    def action_toggle(self) -> None:
        self.expanded = not self.expanded
        self.render_block()

    def on_click(self, event=None) -> None:
        self.action_toggle()


def clean_result(result: Any) -> str:
    """Turn a raw tool result into readable text, unwrapping common MCP shapes."""
    text = str(result or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    if isinstance(parsed, dict):
        for key in ("output", "findings", "result", "stdout", "text", "message", "error", "hint"):
            value = parsed.get(key)
            if value not in (None, "", [], {}):
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    if isinstance(parsed, list):
        return "\n".join(str(item) for item in parsed[:100])
    return text


def result_preview(result: Any, lines: int = 4, limit: int = 360) -> str:
    """Short, non-JSON-blob preview of a tool result for inline display."""
    clean = "\n".join(row.rstrip() for row in clean_result(result).splitlines() if row.strip())
    if not clean:
        return ""
    parts = clean.splitlines()
    if len(parts) > lines:
        clean = "\n".join(parts[:lines]) + f"\n… +{len(parts) - lines} more lines"
    if len(clean) > limit:
        clean = clean[:limit] + "…"
    return clean


class ToolCard(Static):
    ALLOW_SELECT = True

    can_focus = True
    BINDINGS = [Binding("enter", "toggle", "Toggle output", show=False)]

    def __init__(self, name: str, display: str, args: Dict[str, Any]) -> None:
        super().__init__()
        self.tool_name = name
        self.display_name = display or name
        self.args = args
        self.result = ""
        self.duration = 0.0
        self.done = False
        self.ok = True
        self.output_lines: List[str] = []
        self.expanded = False

    def on_mount(self) -> None:
        self.refresh_view()

    def append_output(self, stream: str, text: str) -> None:
        for line in text.splitlines():
            prefix = "!" if stream == "stderr" else ""
            self.output_lines.append(f"{prefix}{line}")
        if len(self.output_lines) > 400:
            self.output_lines = self.output_lines[-400:]
        self.refresh_view()

    def set_result(self, result: str, duration: float = 0.0, ok: bool = True) -> None:
        self.result = result
        self.duration = duration
        self.done = True
        self.ok = ok
        self.refresh_view()

    def _result_output(self) -> str:
        if self.result:
            try:
                data = json.loads(self.result)
                if isinstance(data, dict):
                    if data.get("output") is not None:
                        return str(data.get("output"))
                    if data.get("error"):
                        return f"error: {data['error']}"
            except Exception:
                pass
        return self.result

    def refresh_view(self) -> None:
        if self.tool_name == "shell":
            head = f"$ {self.args.get('command', '')}"
        else:
            arg = json.dumps(self.args, ensure_ascii=False)
            if len(arg) > 120:
                arg = arg[:120] + "..."
            head = f"⚙ {self.display_name}({arg})"
        body = "\n".join(self.output_lines)
        if not body and self.result:
            body = self._result_output()
        lines = body.splitlines()
        if not self.expanded and len(lines) > 8:
            body = f"({len(lines)-8} earlier lines)\n" + "\n".join(lines[-8:])
        if not self.done:
            status, style = "active", "#d4a359"
        elif self.ok:
            status, style = f"✓ done in {self.duration:.2f}s", "#6abf69"
        else:
            status, style = "✗ failed", "#e06c75"
        text = Text()
        text.append(head + "\n", style="bold #e5e5e5")
        if body:
            text.append(body + "\n", style="#a8a8a8")
        text.append(status, style=style)
        self.update(text)

    def action_toggle(self) -> None:
        self.expanded = not self.expanded
        self.refresh_view()

    def on_click(self, event=None) -> None:
        self.action_toggle()


class DiffCard(Static):
    """Compact unified diff for a file edit, with the undo id visible."""

    def __init__(self, diff: str, undo_id: str = "") -> None:
        super().__init__()
        self.diff = diff or ""
        self.undo_id = undo_id or ""

    def on_mount(self) -> None:
        lines = self.diff.splitlines()
        body = "\n".join(lines[:60])
        if len(lines) > 60:
            body += f"\n… {len(lines) - 60} more lines (/diff for the full diff)"
        text = Text()
        text.append("Edit preview\n", style="bold #8ab4f8")
        text.append(body or "(no diff)", style="#a8a8a8")
        if self.undo_id:
            text.append(f"\nundo: /undo {self.undo_id}", style="#6a6a6a")
        self.update(text)


class SubagentCard(Static):
    ALLOW_SELECT = True

    """Live status for one isolated subagent worker."""

    def __init__(self, role: str, task: str) -> None:
        super().__init__()
        self.agent_role = role  # Widget already owns .task / .role
        self.subtask = task

    def on_mount(self) -> None:
        text = Text()
        text.append(f"◇ subagent:{self.agent_role}", style="bold #c58af9")
        text.append("  " + self.subtask[:90], style="#a8a8a8")
        text.append("\n  working…", style="#6a6a6a")
        self.update(text)

    def finish(self, ok: bool, preview: str, duration: float) -> None:
        text = Text()
        text.append(f"{'◆' if ok else '✗'} subagent:{self.agent_role}", style="bold #c58af9" if ok else "bold #ff6b6b")
        text.append("  " + self.subtask[:90], style="#a8a8a8")
        text.append(f"\n  {duration:.1f}s · " + (preview[:180] if ok else f"FAILED: {preview[:160]}"), style="#8ab4f8" if ok else "#ff9d9d")
        self.update(text)


class BackgroundIndicator(Static):
    ALLOW_SELECT = True

    """One-line summary for tool calls that run without chat cards."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[Dict[str, Any]] = []
        self.running = 0

    def on_mount(self) -> None:
        self.refresh_line()

    def start(self, display: str) -> None:
        self.running += 1
        self.refresh_line()

    def finish(self, display: str, duration: float, ok: bool) -> None:
        self.running = max(0, self.running - 1)
        self.calls.append({"name": display, "duration": duration, "ok": ok})
        self.refresh_line()

    def refresh_line(self) -> None:
        n = len(self.calls)
        if self.running:
            text = Text(f"⚙ mcp: {n} background call{'s' if n != 1 else ''} · running…", style="#d4a359")
        else:
            text = Text(f"⚙ mcp: {n} background call{'s' if n != 1 else ''} · done", style="#8a8a8a")
        self.update(text)
        self.set_class(self.running > 0, "running")


class ActivitySpinner(Static):
    """Rotating braille spinner with the current activity and live elapsed time.

    Driven by a local timer so the TUI visibly moves while the model thinks,
    connects or runs tools — not a static line waiting for the next event.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self) -> None:
        super().__init__()
        self.frame = 0
        self.phase = "Connecting"
        self.detail = ""
        self.started = time.perf_counter()
        self.finished: Optional[bool] = None
        self.final_elapsed = 0.0

    def on_mount(self) -> None:
        self._paint()
        self._timer = self.set_interval(0.09, self._spin)  # strong ref: timers are weak-held

    def set_activity(self, phase: str, detail: str = "") -> None:
        if self.finished is not None:
            return
        self.phase = phase
        self.detail = detail
        self._paint()

    def finish(self, ok: bool) -> None:
        if self.finished is not None:
            return
        self.finished = bool(ok)
        self.final_elapsed = time.perf_counter() - self.started
        self._paint()

    def _spin(self) -> None:
        if self.finished is not None:
            return
        self.frame = (self.frame + 1) % len(self.FRAMES)
        self._paint()

    def _paint(self) -> None:
        text = Text()
        if self.finished is None:
            dots = "." * (self.frame % 4)
            label = (self.detail or self.phase).strip()
            if len(label) > 62:
                label = label[:59] + "..."
            text.append(f"{self.FRAMES[self.frame]} ", style="bold #d4a359")
            text.append(f"{label}{dots}", style="#e5e5e5")
            text.append(f"  {time.perf_counter() - self.started:.1f}s", style="#6a6a6a")
        else:
            ok = bool(self.finished)
            mark = "✓" if ok else "✗"
            style = "bold #6abf69" if ok else "bold #ff6b6b"
            text.append(f"{mark} ", style=style)
            text.append("Response ready" if ok else "Turn stopped", style=style)
            text.append(f"  {self.final_elapsed:.1f}s", style="#6a6a6a")
        self.update(text)


class SectionLabel(Static):
    ALLOW_SELECT = True

    """Divider that separates tool findings from the model response."""

    def __init__(self, title: str, style: str) -> None:
        super().__init__()
        self.section_title = title
        self.section_style = style

    def on_mount(self) -> None:
        self.update(Text(f"── {self.section_title} ──", style=self.section_style))


class FindingCard(Static):
    ALLOW_SELECT = True

    """Readable preview of a background MCP/search result.

    Raw output stays out of the chat by default; the model still receives the
    full result. Click/Enter to expand. ``/bg <n>`` prints the complete result.
    """

    can_focus = True
    BINDINGS = [Binding("enter", "toggle", "Expand/collapse", show=False)]

    def __init__(
        self,
        name: str,
        display: str,
        result: str,
        duration: float,
        ok: bool,
        hint: str = "",
    ) -> None:
        super().__init__()
        self.tool_name = name
        self.display_name = display or name
        self.full_result = clean_result(result)
        self.duration = float(duration or 0.0)
        self.ok = bool(ok)
        self.hint = str(hint or "")
        self.expanded = False

    def on_mount(self) -> None:
        self.refresh_view()

    def refresh_view(self) -> None:
        body = (
            self.full_result[:4000] if self.expanded else result_preview(self.full_result)
        )
        if not self.ok:
            body = "FAILED: " + (body or self.hint[:200] or "unknown error")
        elif not body:
            body = "(no output)"
        text = Text()
        mark = "◆" if self.ok else "✗"
        text.append(
            f"{mark} finding: {self.display_name}",
            style="bold #6abf69" if self.ok else "bold #ff6b6b",
        )
        text.append(f"  · {self.duration:.2f}s", style="#6a6a6a")
        text.append("\n" + body, style="#a8a8a8" if self.ok else "#ff9d9d")
        text.append("\n  Enter/click to expand · /bg <n> for full result", style="#5a5a5a")
        self.update(text)

    def action_toggle(self) -> None:
        self.expanded = not self.expanded
        self.refresh_view()

    def on_click(self, event=None) -> None:
        self.action_toggle()


class ErrorCard(Static):
    ALLOW_SELECT = True

    """Structured provider error, rendered instead of a raw traceback."""

    def __init__(self, card: Dict[str, Any], elapsed: float = 0.0) -> None:
        super().__init__()
        self.card = card or {}
        self.elapsed = float(elapsed or 0.0)

    def on_mount(self) -> None:
        self.render_card()

    def render_card(self) -> None:
        c = self.card
        status = c.get("status_code") or 402
        reason = c.get("reason") or "request failed"
        daily = bool(c.get("daily_limit"))
        text = Text()
        text.append(f"HTTP {status} · {reason}\n", style="bold #e06c75")
        message = (c.get("message") or "").strip()
        if message:
            text.append(message + "\n", style="#f0c0c0")
        reset_at = c.get("reset_at")
        if reset_at:
            try:
                when = time.strftime("%H:%M:%S", time.localtime(float(reset_at)))
                left = max(0, int(float(reset_at) - time.time()))
                text.append(
                    f"Daily limit resets at {when} (~{left // 60}m {left % 60}s)\n",
                    style="#d4a359",
                )
            except Exception:
                pass
        elif c.get("retry_after") is not None:
            text.append(
                f"Retry after ~{c['retry_after']}s or once in-flight requests settle\n",
                style="#d4a359",
            )
        if c.get("limit_source"):
            text.append(f"limit: {c['limit_source']}\n", style="#888888")
        hint = (c.get("remedy_hint") or "").strip()
        if hint:
            text.append(hint + "\n", style="#9a9a9a")
        text.append(f"elapsed: {self.elapsed:.1f}s\n", style="#777777")
        if daily:
            text.append(
                "Stop hitting this cap: /provider nvidia (add NVIDIA_API_KEY to .env), "
                "run a local model, or add $10 OpenRouter credits for 1000/day.",
                style="#8ab4f8",
            )
        else:
            text.append("Recover: /retry · /model · /status", style="#8ab4f8")
        self.update(text)


class ChatMessage(Vertical):
    ALLOW_SELECT = True

    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self.role = role
        self.raw_content = content
        self._md: Markdown | None = None
        self._user_text: Static | None = None

    def on_mount(self) -> None:
        self.render_msg()

    def set_content(self, content: str) -> None:
        self.raw_content = content
        self.render_msg()

    def render_msg(self) -> None:
        if self.role == "user":
            self.remove_class("msg-assistant")
            self.add_class("msg-user")
            if self._md is not None:
                self._md.remove()
                self._md = None
            if self._user_text is None:
                self._user_text = Static(Text(self.raw_content, style="#f0f0f0"), markup=False)
                self._user_text.ALLOW_SELECT = True
                self.mount(self._user_text)
            else:
                self._user_text.update(Text(self.raw_content, style="#f0f0f0"))
        else:
            self.remove_class("msg-user")
            self.add_class("msg-assistant")
            if self._user_text is not None:
                self._user_text.remove()
                self._user_text = None
            content = self.raw_content or " "
            if self._md is None:
                self._md = Markdown(content)
                self._md.ALLOW_SELECT = True
                self.mount(self._md)
            else:
                try:
                    self.run_worker(self._md.update(content), exclusive=False)
                except Exception:
                    try:
                        import asyncio
                        asyncio.create_task(self._md.update(content))
                    except Exception:
                        pass


class PromptArea(TextArea):
    """TextArea that owns prompt shortcuts and forwards palette navigation."""

    async def _on_key(self, event) -> None:  # type: ignore[override]
        key = getattr(event, "key", "")
        shift = bool(getattr(event, "shift", False))
        app = self.app  # type: ignore[assignment]

        if getattr(app, "palette_visible", False) and key in ("up", "down", "enter", "escape"):
            event.stop()
            event.prevent_default()
            if key == "up":
                app.palette_move(-1)
            elif key == "down":
                app.palette_move(1)
            elif key == "enter":
                app.palette_select()
            else:
                app.hide_palette()
            return

        if key == "shift+enter" or (key == "enter" and shift):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if key in ("enter", "ctrl+enter"):
            event.stop()
            event.prevent_default()
            await app.submit_prompt()
            return
        # Tab is handled by the app-level binding so it works from any focus,
        # not only when the prompt happens to be focused.
        if key == "ctrl+c":
            event.stop()
            event.prevent_default()
            await app.handle_ctrl_c()
            return
        if key == "ctrl+shift+v":
            event.stop()
            event.prevent_default()
            app.action_paste_clipboard()
            return
        if key == "ctrl+shift+c":
            event.stop()
            event.prevent_default()
            app.action_copy_selection()
            return
        await super()._on_key(event)


# --------------------------------------------------------------------------- modals


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal small"):
            yield Static(self.message, classes="modal-title")
            with Horizontal(classes="button-row"):
                yield Button("Cancel", id="confirm-no")
                yield Button("Confirm", id="confirm-yes", variant="error")

    @on(Button.Pressed, "#confirm-yes")
    def _yes(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _no(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class PermissionScreen(ModalScreen[str]):
    """Three-way confirmation for a potentially unsafe tool call."""

    BINDINGS = [Binding("escape", "deny", "Deny")]

    def __init__(self, message: str, risk: str) -> None:
        super().__init__()
        self.message = message
        self.risk = risk

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal small"):
            yield Static(f"Permission · {self.risk}\n\n{self.message}", classes="modal-title")
            with Horizontal(classes="button-row"):
                yield Button("Deny", id="perm-deny", variant="error")
                yield Button("Allow once", id="perm-once", variant="warning")
                yield Button("Allow session", id="perm-session", variant="success")

    @on(Button.Pressed, "#perm-once")
    def _once(self) -> None:
        self.dismiss("once")

    @on(Button.Pressed, "#perm-session")
    def _session(self) -> None:
        self.dismiss("session")

    @on(Button.Pressed, "#perm-deny")
    def _deny(self) -> None:
        self.dismiss("deny")

    def action_deny(self) -> None:
        self.dismiss("deny")


class AskQuestionScreen(ModalScreen[Optional[str]]):
    """Cline-style follow-up question: choose an option or type a custom answer."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, question: str, options: List[str]) -> None:
        super().__init__()
        self.question = question
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal small"):
            yield Static(f"Question\n\n{self.question}", classes="modal-title")
            yield OptionList(id="question-options")
            yield Input(placeholder="Or type your own answer…", id="question-custom")
            with Horizontal(classes="button-row"):
                yield Button("Answer", id="question-answer", variant="success")
                yield Button("Cancel", id="question-cancel")
            yield Static("↑↓ choose · Enter answer · Esc cancel", classes="modal-help")

    def on_mount(self) -> None:
        options = self.query_one("#question-options", OptionList)
        for index, option in enumerate(self.options):
            options.add_option(Option(option, id=f"q-opt-{index}"))
        if self.options:
            options.highlighted = 0

    def _answer(self) -> None:
        custom = self.query_one("#question-custom", Input).value.strip()
        if custom:
            self.dismiss(custom)
            return
        option = self.query_one("#question-options", OptionList).highlighted_option
        if option is not None:
            self.dismiss(str(option.prompt))

    @on(OptionList.OptionSelected, "#question-options")
    def _picked(self) -> None:
        self._answer()

    @on(Button.Pressed, "#question-answer")
    def _submit(self) -> None:
        self._answer()

    @on(Button.Pressed, "#question-cancel")
    def _cancel(self) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


def _dedupe_models(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate model ids while keeping the first (richest) entry."""
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        model_id = str(entry.get("id") or "").strip()
        if model_id:
            merged.setdefault(model_id, entry)
    return list(merged.values())


class ModelSearchInput(Input):
    """Search box that forwards list navigation keys to the model list.

    Textual's Input consumes printable characters and binds Enter to `submit`.
    Up/Down are forwarded here so the user can filter and then arrow through
    the results without leaving the search box; Enter selects the highlight.
    """

    def on_key(self, event: events.Key) -> None:
        if event.key in ("up", "down"):
            option_list = self.screen.query_one("#model-list", OptionList)
            if event.key == "up":
                option_list.action_cursor_up()
            else:
                option_list.action_cursor_down()
            event.stop()
            event.prevent_default()

    async def action_submit(self) -> None:
        # Enter while typing selects the highlighted model instead of posting
        # Input.Submitted (which has no meaning in this modal).
        self.screen.query_one("#model-list", OptionList).action_select()


class ModelPickerScreen(ModalScreen[Optional[Any]]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, current: str, active_provider: Optional[str] = None) -> None:
        super().__init__()
        self.current = current
        self.active_provider = active_provider or current_provider()
        # (provider key, display label, models) — headers are disabled rows.
        self.groups: List[tuple[str, str, List[Dict[str, Any]]]] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Static("Select a model · OpenRouter + NVIDIA NIM", classes="modal-title")
            yield ModelSearchInput(placeholder="Search all models…", id="model-search")
            yield OptionList(id="model-list")
            yield Static("↑/↓ move · Enter select · Esc close · type to filter", classes="modal-help")

    def on_mount(self) -> None:
        self.run_worker(self._load(), exclusive=False)

    async def _load(self) -> None:
        async def fetch(provider: str) -> List[Dict[str, Any]]:
            try:
                catalog = await asyncio.to_thread(get_catalog, provider=provider)
                entries = list(catalog.get("free") or []) + list(catalog.get("uncensored") or [])
            except Exception:
                entries = []
            return _dedupe_models(entries)

        async def local() -> List[Dict[str, Any]]:
            try:
                return await asyncio.to_thread(discover_local_models)
            except Exception:
                return []

        # Load both backends together: a slow or offline one must not hide the
        # other. Local Ollama models live in the OpenRouter group.
        openrouter, nvidia, local_models = await asyncio.gather(
            fetch("openrouter"), fetch("nvidia"), local()
        )
        self.groups = [
            ("openrouter", provider_config("openrouter")["label"],
             _dedupe_models(openrouter + local_models)),
            ("nvidia", provider_config("nvidia")["label"], nvidia),
        ]
        self._populate("")

    def _populate(self, query: str) -> None:
        query = (query or "").lower().strip()
        option_list = self.query_one("#model-list", OptionList)
        option_list.clear_options()
        options: List[Option] = []
        first_enabled: Optional[int] = None
        for provider, label, models in self.groups:
            shown: List[Option] = []
            for model in models:
                haystack = f"{model.get('id','')} {model.get('name','')}".lower()
                if query and query not in haystack:
                    continue
                ctx = int(model.get("context") or 0)
                ctx_text = f"{ctx // 1000}k" if ctx else "?"
                tag = "local" if model.get("local") else ("uncensored" if model.get("uncensored") else "free")
                mark = "*" if (provider == self.active_provider and model["id"] == self.current) else " "
                shown.append(Option(
                    f"{mark} {model['id']}  ·  {ctx_text} ctx  ·  {tag}",
                    id=f"{provider}|{model['id']}",
                ))
            if not shown:
                continue
            options.append(Option(f"── {label} ({len(shown)}) ──", disabled=True))
            if first_enabled is None:
                first_enabled = len(options)
            options.extend(shown)
        option_list.add_options(options)
        # Disabled headers cannot be highlighted, so start on the first real model.
        option_list.highlighted = first_enabled

    @on(Input.Changed, "#model-search")
    def _filter(self, event: Input.Changed) -> None:
        self._populate(event.value)

    @on(OptionList.OptionSelected, "#model-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        option_id = str(event.option.id or "")
        if "|" in option_id:
            provider, model = option_id.split("|", 1)
            self.dismiss((provider, model))
        elif option_id:
            self.dismiss((None, option_id))  # legacy bare id
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AddServerScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Static("Add MCP server", classes="modal-title")
            yield Input(placeholder="name (e.g. kala-mcp-root)", id="srv-name")
            yield Input(placeholder="command (stdio) OR leave blank and use URL", id="srv-command")
            yield Input(placeholder="args (space separated, optional)", id="srv-args")
            yield Input(placeholder="url (sse / streamable http, optional)", id="srv-url")
            yield Input(placeholder="env KEY=VALUE, comma separated (optional)", id="srv-env")
            yield Input(placeholder="scope: user (default) or project", id="srv-scope", value="user")
            with Horizontal(classes="button-row"):
                yield Button("Cancel", id="srv-cancel")
                yield Button("Save & connect", id="srv-save", variant="primary")
            yield Static("", id="srv-error", classes="error-text")

    @on(Button.Pressed, "#srv-cancel")
    def _cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#srv-save")
    async def _save(self) -> None:
        name = self.query_one("#srv-name", Input).value.strip()
        command = self.query_one("#srv-command", Input).value.strip()
        url = self.query_one("#srv-url", Input).value.strip()
        args_raw = self.query_one("#srv-args", Input).value.strip()
        env_raw = self.query_one("#srv-env", Input).value.strip()
        scope = self.query_one("#srv-scope", Input).value.strip() or "user"
        error = self.query_one("#srv-error", Static)
        if not name:
            error.update("name is required")
            return
        if not command and not url:
            error.update("provide a command (stdio) or a URL (sse/http)")
            return
        cfg: Dict[str, Any] = {}
        if command:
            cfg["command"] = command
            cfg["args"] = args_raw.split() if args_raw else []
        else:
            cfg["url"] = url
        env: Dict[str, str] = {}
        if env_raw:
            for item in env_raw.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    env[key.strip()] = value.strip()
        if env:
            cfg["env"] = env
        try:
            await self.app.mcp.add_server(name, cfg, scope=scope)
        except Exception as exc:  # noqa: BLE001
            error.update(f"could not add server: {exc}")
            return
        self.app.refresh_status()
        self.dismiss(True)


class SkillEditorScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, skill: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.skill = skill

    def compose(self) -> ComposeResult:
        editing = self.skill is not None
        with Vertical(classes="modal"):
            yield Static("Edit skill" if editing else "Create skill", classes="modal-title")
            yield Input(value=(self.skill or {}).get("name", ""), placeholder="name", id="skill-name")
            yield Input(value=(self.skill or {}).get("description", ""), placeholder="description", id="skill-description")
            yield Input(value=", ".join((self.skill or {}).get("triggers", [])), placeholder="triggers, comma separated", id="skill-triggers")
            yield Input(value=", ".join((self.skill or {}).get("tools", [])), placeholder="preferred tools, comma separated", id="skill-tools")
            yield TextArea(
                (self.skill or {}).get("instructions", ""),
                id="skill-instructions",
                show_line_numbers=False,
            )
            with Horizontal(classes="button-row"):
                yield Button("Cancel", id="skill-cancel")
                yield Button("Save", id="skill-save", variant="primary")
            yield Static("", id="skill-error", classes="error-text")

    @on(Button.Pressed, "#skill-cancel")
    def _cancel(self) -> None:
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#skill-save")
    def _save(self) -> None:
        name = self.query_one("#skill-name", Input).value.strip()
        description = self.query_one("#skill-description", Input).value.strip()
        triggers = [t.strip() for t in self.query_one("#skill-triggers", Input).value.split(",") if t.strip()]
        tools = [t.strip() for t in self.query_one("#skill-tools", Input).value.split(",") if t.strip()]
        instructions = self.query_one("#skill-instructions", TextArea).text
        error = self.query_one("#skill-error", Static)
        if not name:
            error.update("name is required")
            return
        try:
            if self.skill:
                self.app.skills.update(name, description, triggers, tools, instructions)
            else:
                self.app.skills.create(name, description, triggers, tools, instructions)
        except Exception as exc:  # noqa: BLE001
            error.update(f"could not save skill: {exc}")
            return
        self.app.refresh_status()
        self.app.update_context()
        self.dismiss(True)


class SkillsManagerScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("space", "toggle", "Toggle"),
        Binding("enter", "toggle", "Toggle"),
        Binding("n", "new", "New"),
        Binding("e", "edit", "Edit"),
        Binding("r", "reload", "Reload"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal wide"):
            yield Static("Skills", classes="modal-title")
            yield OptionList(id="skill-list")
            with VerticalScroll(classes="details"):
                yield Static("", id="skill-details")
            yield Static("Space/Enter toggle · n new · e edit · r reload · Esc close", classes="modal-help")

    def on_mount(self) -> None:
        self.refresh_list()

    def refresh_list(self) -> None:
        option_list = self.query_one("#skill-list", OptionList)
        current = option_list.highlighted_option.id if option_list.highlighted_option else None
        option_list.clear_options()
        rows = self.app.skills.rows()
        for row in rows:
            mark = "[x]" if row["active"] else "[ ]"
            option_list.add_option(Option(f"{mark} {row['name']}  —  {row['description']}", id=row["name"]))
        if current:
            for index, option in enumerate(option_list.options):
                if option.id == current:
                    option_list.highlighted = index
                    break
        if not rows:
            self._details("No skills yet. Press n to create one.")
        else:
            self._show_selected()

    def _details(self, text: str) -> None:
        self.query_one("#skill-details", Static).update(Text(text))

    def _selected_name(self) -> Optional[str]:
        option = self.query_one("#skill-list", OptionList).highlighted_option
        return option.id if option else None

    def _show_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        row = next((r for r in self.app.skills.rows() if r["name"] == name), None)
        if not row:
            return
        skill = self.app.skills.get(name)
        lines = [
            f"{row['name']}  ({'active' if row['active'] else 'inactive'})",
            f"path: {row['path']}",
            f"triggers: {', '.join(row['triggers']) or '(none)'}",
            f"preferred tools: {', '.join(row['tools']) or '(none)'}",
            "",
            (skill.instructions if skill else ""),
        ]
        self._details("\n".join(lines))

    @on(OptionList.OptionHighlighted, "#skill-list")
    def _highlighted(self) -> None:
        self._show_selected()

    @on(OptionList.OptionSelected, "#skill-list")
    def _selected(self) -> None:
        self.action_toggle()

    def action_toggle(self) -> None:
        name = self._selected_name()
        if not name:
            return
        self.app.skills.toggle(name)
        self.refresh_list()
        self._show_selected()
        self.app.refresh_status()
        self.app.update_context()

    def action_new(self) -> None:
        self.app.push_screen(SkillEditorScreen(None), lambda _: self.refresh_list())

    def action_edit(self) -> None:
        name = self._selected_name()
        if not name:
            return
        row = next((r for r in self.app.skills.rows() if r["name"] == name), None)
        if row:
            self.app.push_screen(SkillEditorScreen(row), lambda _: self.refresh_list())

    def action_reload(self) -> None:
        self.app.skills.reload()
        self.refresh_list()
        self.app.refresh_status()
        self.app.update_context()

    def action_close(self) -> None:
        self.dismiss(None)


class ToolsManagerScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("space", "toggle", "Toggle"),
        Binding("enter", "toggle", "Toggle"),
        Binding("i", "inspect", "Inspect"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal wide"):
            yield Static("Tools", classes="modal-title")
            yield OptionList(id="tool-list")
            with VerticalScroll(classes="details"):
                yield Static("", id="tool-details")
            yield Static("Space/Enter toggle · i inspect · Esc close", classes="modal-help")

    def on_mount(self) -> None:
        self.refresh_list()

    def _tools(self) -> List[Dict[str, Any]]:
        registry = ToolRegistry(self.app.mcp, disabled=set(), model=str(getattr(self.app, "model", "") or ""))
        rows = []
        for spec in registry.local_specs():
            fn = spec["function"]
            rows.append(
                {
                    "id": fn["name"],
                    "label": fn["name"],
                    "description": fn.get("description", ""),
                    "schema": fn.get("parameters", {}),
                    "kind": "built-in",
                }
            )
        for tool in self.app.mcp.list_tools():
            rows.append(
                {
                    "id": tool.qualified,
                    "label": tool.display,
                    "description": tool.description,
                    "schema": tool.input_schema,
                    "kind": "mcp",
                }
            )
        return rows

    def refresh_list(self) -> None:
        option_list = self.query_one("#tool-list", OptionList)
        current = option_list.highlighted_option.id if option_list.highlighted_option else None
        option_list.clear_options()
        for row in self._tools():
            mark = "[ ]" if row["id"] in self.app.disabled_tools else "[x]"
            option_list.add_option(Option(f"{mark} {row['label']}  ·  {row['kind']}", id=row["id"]))
        if current:
            for index, option in enumerate(option_list.options):
                if option.id == current:
                    option_list.highlighted = index
                    break
        if not self._tools():
            self.query_one("#tool-details", Static).update(Text("No tools available."))
        else:
            self._inspect()

    def _selected(self) -> Optional[Dict[str, Any]]:
        option = self.query_one("#tool-list", OptionList).highlighted_option
        if not option:
            return None
        return next((r for r in self._tools() if r["id"] == option.id), None)

    def _inspect(self) -> None:
        row = self._selected()
        if not row:
            return
        state = "disabled" if row["id"] in self.app.disabled_tools else "enabled"
        text = (
            f"{row['label']}  ({row['kind']}, {state})\n\n"
            f"{row['description']}\n\n"
            + json.dumps(row["schema"], indent=2)
        )
        self.query_one("#tool-details", Static).update(Text(text))

    @on(OptionList.OptionHighlighted, "#tool-list")
    def _highlighted(self) -> None:
        self._inspect()

    @on(OptionList.OptionSelected, "#tool-list")
    def _selected_event(self) -> None:
        self.action_toggle()

    def action_toggle(self) -> None:
        row = self._selected()
        if not row:
            return
        tool_id = row["id"]
        if tool_id in self.app.disabled_tools:
            self.app.disabled_tools.discard(tool_id)
        else:
            self.app.disabled_tools.add(tool_id)
        appconfig.update(disabled_tools=sorted(self.app.disabled_tools))
        self.refresh_list()
        self.app.refresh_status()
        self.app.update_context()

    def action_inspect(self) -> None:
        self._inspect()

    def action_close(self) -> None:
        self.dismiss(None)


class MCPManagerScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("c", "toggle", "Connect/Disconnect"),
        Binding("r", "restart", "Restart"),
        Binding("i", "inspect", "Inspect"),
        Binding("l", "logs", "Logs"),
        Binding("a", "add", "Add"),
        Binding("d", "remove", "Remove"),
        Binding("t", "toggle_tools", "Tools on/off"),
        Binding("b", "toggle_display", "MCP display"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal wide"):
            yield Static("MCP servers", classes="modal-title")
            yield OptionList(id="mcp-list")
            with VerticalScroll(classes="details"):
                yield Static("", id="mcp-details")
            yield Static("c connect · r restart · t tools on/off · b background/verbose · i inspect · l logs · a add · d remove · Esc close", classes="modal-help")

    def on_mount(self) -> None:
        self.refresh_list()
        self._timer = self.set_interval(0.8, self.refresh_list)  # strong ref: timers are weak-held

    def refresh_list(self) -> None:
        option_list = self.query_one("#mcp-list", OptionList)
        current = option_list.highlighted_option.id if option_list.highlighted_option else None
        option_list.clear_options()
        rows = self.app.mcp.status_rows()
        for row in rows:
            mark = "●" if row["status"] == "connected" else ("○" if row["status"] in ("disconnected", "disabled", "stopped") else "✗")
            tools_state = "tools OFF" if row["name"] in self.app.disabled_servers else f"{row['tools']} tools"
            label = f"{mark} {row['name']}  [{row['transport']}]  {tools_state}"
            if row["error"]:
                label += f"  — {row['error'][:60]}"
            option_list.add_option(Option(label, id=row["name"]))
        if current:
            for index, option in enumerate(option_list.options):
                if option.id == current:
                    option_list.highlighted = index
                    break
        if not rows:
            self._details("No MCP servers configured. Press a to add one.")
        else:
            self._show_selected()
        self.app.refresh_status()

    def _details(self, text: str) -> None:
        self.query_one("#mcp-details", Static).update(Text(text))

    def _selected_name(self) -> Optional[str]:
        option = self.query_one("#mcp-list", OptionList).highlighted_option
        return option.id if option else None

    def _show_selected(self) -> None:
        name = self._selected_name()
        if not name:
            return
        state = self.app.mcp.servers.get(name)
        if not state:
            return
        text = (
            f"{name}\n"
            f"status: {state.status}\n"
            f"transport: {state.transport}\n"
            f"tools: {len(state.tools)}  resources: {len(state.resources)}  prompts: {len(state.prompts)}\n"
            f"tool injection: {'OFF' if name in self.app.disabled_servers else 'ON'}\n"
            f"log: {state.log_path}\n"
        )
        if state.error:
            text += f"error: {state.error}\n"
        if state.tools:
            text += "\n" + ", ".join(t.name for t in state.tools)
        self._details(text)

    @on(OptionList.OptionHighlighted, "#mcp-list")
    def _highlighted(self) -> None:
        self._show_selected()

    async def action_toggle(self) -> None:
        name = self._selected_name()
        if not name:
            return
        state = self.app.mcp.servers.get(name)
        if not state:
            return
        if state.status == "connected":
            await self.app.mcp.stop_server(name)
        else:
            await self.app.mcp.start_server(name, dict(state.config))
        self.refresh_list()
        self.app.refresh_status()

    def action_toggle_display(self) -> None:
        self.app.action_toggle_mcp_display()
        self.refresh_list()

    def action_toggle_tools(self) -> None:
        name = self._selected_name()
        if not name:
            return
        if name in self.app.disabled_servers:
            self.app.disabled_servers.discard(name)
            note = f"{name}: tools ON"
        else:
            self.app.disabled_servers.add(name)
            note = f"{name}: tools OFF"
        appconfig.update(disabled_servers=sorted(self.app.disabled_servers))
        self.refresh_list()
        self._show_selected()
        self.app.refresh_status()
        self.app.update_context()
        self._details(note + "\n\n" + self._current_details())

    def _current_details(self) -> str:
        name = self._selected_name()
        if not name:
            return ""
        state = self.app.mcp.servers.get(name)
        if not state:
            return ""
        return (
            f"{name}\nstatus: {state.status}\ntransport: {state.transport}\n"
            f"tools: {len(state.tools)}  resources: {len(state.resources)}  prompts: {len(state.prompts)}\n"
            f"tool injection: {'OFF' if name in self.app.disabled_servers else 'ON'}\n"
            f"log: {state.log_path}\n"
            + (f"error: {state.error}\n" if state.error else "")
            + ("\n" + ", ".join(t.name for t in state.tools) if state.tools else "")
        )

    async def action_restart(self) -> None:
        name = self._selected_name()
        if not name:
            return
        try:
            await self.app.mcp.restart_server(name)
        except Exception as exc:  # noqa: BLE001
            self._details(f"restart failed: {exc}")
            return
        self.refresh_list()

    def action_inspect(self) -> None:
        name = self._selected_name()
        if name:
            self._details(self.app.mcp.schema_text(name))

    def action_logs(self) -> None:
        name = self._selected_name()
        if name:
            self._details(self.app.mcp.log_tail(name))

    def action_add(self) -> None:
        self.app.push_screen(AddServerScreen(), lambda _: self.refresh_list())

    def action_remove(self) -> None:
        name = self._selected_name()
        if not name:
            return

        async def _remove() -> None:
            await self.app.mcp.remove_server(name)
            self.refresh_list()
            self.app.refresh_status()

        self.app.push_screen(
            ConfirmScreen(f"Remove MCP server '{name}' from mcp.json and stop it?"),
            lambda confirmed: self.run_worker(_remove()) if confirmed else None,
        )

    def action_close(self) -> None:
        self.dismiss(None)


class SessionPickerScreen(ModalScreen[Optional[str]]):
    """OpenCode-style session picker: recent chats from the DB and checkpoints."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal wide"):
            yield Static("Recent sessions", classes="modal-title")
            yield OptionList(id="session-list")
            yield Static(
                "Enter resume · Esc close · sessions are kept on disk, nothing is deleted",
                classes="modal-help",
            )

    def on_mount(self) -> None:
        self._populate()

    def _populate(self) -> None:
        options: List[Option] = []
        # Database conversations (full text history with tools).
        try:
            for conv in db.list_conversations():
                title = str(conv.get("title") or "(untitled)").replace("\n", " ")[:64]
                updated = str(conv.get("updated_at") or "")[:16]
                model = str(conv.get("model") or "").split("/")[-1]
                options.append(
                    Option(
                        f"chat  {title:<64}  {updated}  {model}",
                        id=f"db:{conv['id']}",
                    )
                )
        except Exception:
            pass
        # Checkpoint sessions (restored text-only).
        try:
            for sess in session_util.list_sessions(os.getcwd(), 12):
                title = str(sess.get("last_user") or "(empty)").replace("\n", " ")[:52]
                sid = str(sess.get("id") or "")
                turns = int(sess.get("turns") or 0)
                options.append(
                    Option(
                        f"save  {title:<52}  {turns} turns  {sid[:12]}",
                        id=f"cp:{sid}",
                    )
                )
        except Exception:
            pass
        option_list = self.query_one("#session-list", OptionList)
        option_list.clear_options()
        if options:
            option_list.add_options(options)
            option_list.highlighted = 0
        else:
            option_list.add_options([Option("No saved sessions yet", id="none")])

    @on(OptionList.OptionSelected, "#session-list")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        sel = event.option.id or ""
        self.dismiss(None if sel == "none" else sel)

    def action_cancel(self) -> None:
        self.dismiss(None)


# --------------------------------------------------------------------------- app


class DarkGPTApp(App):
    CSS_PATH = "theme.tcss"
    BINDINGS = [
        Binding("ctrl+m", "open_mcp", "MCP", show=False),
        Binding("ctrl+g", "open_skills", "Skills", show=False),
        Binding("ctrl+t", "open_tools", "Tools", show=False),
        Binding("ctrl+o", "toggle_mcp_display", "MCP display", show=False),
        Binding("ctrl+l", "open_model", "Model", show=False),
        Binding("ctrl+p", "command_palette", "Commands", show=False),
        Binding("ctrl+y", "copy_last", "Copy answer", show=False),
        Binding("ctrl+shift+a", "copy_all", "Copy everything", show=False),
        Binding("tab", "toggle_mode", "Build/Ask", show=False),
        Binding("ctrl+shift+c", "copy_selection", "Copy selection", show=False),
        Binding("ctrl+shift+v", "paste_clipboard", "Paste", show=False),
        # QTerminal owns Ctrl+Shift+C/V. These alternatives reach DarkGPT
        # itself, so copy/paste works regardless of terminal shortcuts.
        Binding("ctrl+alt+c", "copy_selection", "Copy selection", show=False),
        Binding("ctrl+alt+v", "paste_clipboard", "Paste", show=False),
        Binding("alt+c", "copy_selection", "Copy selection", show=False),
        Binding("alt+v", "paste_clipboard", "Paste", show=False),
        Binding("ctrl+c", "handle_ctrl_c", "Copy/Stop/Quit", show=False),
        Binding("escape", "escape", "Close/Stop", show=False),
        Binding("ctrl+n", "new_chat", "New", show=False),
    ]

    mode = reactive("Build")
    model = reactive(DEFAULT_MODEL)
    busy = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.mcp = MCPManager()
        self.skills = SkillRegistry()
        self.disabled_tools: set[str] = set()
        self.disabled_servers: set[str] = set()
        self.conv_id: Optional[int] = None
        self.messages: List[Dict[str, Any]] = []
        self._worker = None
        self.last_prompt: str = ""
        self.last_answer: str = ""
        self._usage: Dict[str, Any] = {"tokens": 0, "pct": 0.0}
        self._collapsing = False
        self._ctrl_c_once = False
        self._pending: List[str] = []  # prompts submitted during a running turn
        self._last_failed: str = ""
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str = ""
        self.mcp_display: str = "background"
        self._bg_calls: List[Dict[str, Any]] = []
        self._last_diff: str = ""
        self._last_undo_id: str = ""
        self.cheap_model: str = ""
        self.verify_enabled: bool = True
        self.memory_enabled: bool = True
        self.plan_enabled: bool = False
        self.review_enabled: bool = False
        self.worktree_enabled: bool = True
        self.lsp_enabled: bool = True
        self.subagents_enabled: bool = True
        self._delegates: List[Dict[str, Any]] = []
        self._session_approvals: set[str] = set()
        self.permission_mode: str = "full"
        self.hooks_enabled: bool = True
        self.checkpoint_enabled: bool = True
        self.scripts_enabled: bool = True
        self.allow_react_fallback: bool = True
        self._session_id: str = ""
        self._liveness_phase = "Working"
        self._liveness_detail = ""
        self._liveness_frame = 0
        self._liveness_started = 0.0
        self._liveness_timer = None
        self._active_subagents = 0
        self._job_manager = jobs.JobManager()

    # ---------------------------------------------------------------- compose
    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static(self._header_left(), id="header-left")
            with Horizontal(id="session-bar"):
                yield Button("Sessions", id="session-btn", classes="badge")
                yield Static("new chat", id="session-current", classes="session-current")
            with Horizontal(id="header-badges"):
                yield Button("MCP 0/0", id="badge-mcp", classes="badge")
                yield Button("Skills 0", id="badge-skills", classes="badge")
                yield Button("Tools 0", id="badge-tools", classes="badge")
        with Vertical(id="body"):
            yield VerticalScroll(id="conversation")
            yield OptionList(id="palette")
            yield Static("", id="liveness-bar")
            with Vertical(id="prompt-wrap"):
                with Vertical(id="prompt-box"):
                    yield PromptArea("", id="prompt-input", show_line_numbers=False)
                    yield Static("", id="hint-line")
                    with Horizontal(id="prompt-footer"):
                        yield Button("Build", id="mode-btn", classes="footer-btn")
                        yield Button("model", id="model-btn", classes="footer-btn")
                        yield Static("MCP 0/0", id="footer-mcp", classes="footer-info")
                        yield Static("Skills 0", id="footer-skills", classes="footer-info")
                        yield Static("0 (0%)", id="token-label")

    async def on_mount(self) -> None:
        db.init_db()
        ensure_default_config()
        config = appconfig.load_config()
        self.model = config.get("model") or DEFAULT_MODEL
        self.mode = config.get("mode") or "Build"
        self.disabled_tools = set(config.get("disabled_tools") or [])
        self.disabled_servers = set(config.get("disabled_servers") or [])
        self.mcp_display = str(config.get("mcp_display") or "background")
        self.cheap_model = str(config.get("cheap_model") or "")
        self.verify_enabled = bool(config.get("verify_enabled", True))
        self.memory_enabled = bool(config.get("memory_enabled", True))
        self.self_learning_enabled = bool(config.get("self_learning_enabled", True))
        self.active_target = str(config.get("active_target") or "").strip()
        self.plan_enabled = bool(config.get("plan_enabled", False))
        self.review_enabled = bool(config.get("review_enabled", False))
        self.worktree_enabled = bool(config.get("worktree_enabled", True))
        self.lsp_enabled = bool(config.get("lsp_enabled", True))
        self.subagents_enabled = bool(config.get("subagents_enabled", True))
        self.permission_mode = str(config.get("permission_mode") or "full")
        self.hooks_enabled = bool(config.get("hooks_enabled", True))
        self.checkpoint_enabled = bool(config.get("checkpoint_enabled", True))
        self.scripts_enabled = bool(config.get("scripts_enabled", True))
        self.allow_react_fallback = bool(config.get("allow_react_fallback", True))
        if self.mcp_display not in ("background", "verbose"):
            self.mcp_display = "background"
        self._cooldown_until = float(config.get("cooldown_until") or 0.0)
        self._cooldown_reason = str(config.get("cooldown_reason") or "")
        if self._cooldown_until and self._cooldown_until <= time.time():
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            appconfig.update(cooldown_until=0, cooldown_reason="")
        self.skills = SkillRegistry()
        # Textual's built-in Screen binding claims Tab for focus navigation,
        # which used to win whenever the prompt was not focused and made the
        # Build/Ask toggle look broken. Give our toggle priority on the main
        # screen only; modal screens keep their own Tab navigation.
        try:
            tab_bindings = self.screen._bindings.key_to_bindings.setdefault("tab", [])
            tab_bindings.insert(
                0,
                Binding(
                    "tab", "app.toggle_mode", "Build/Ask",
                    show=False, priority=True,
                ),
            )
            self.screen.refresh_bindings()
        except Exception:
            pass
        self.query_one("#palette", OptionList).display = False
        self.query_one("#prompt-input", PromptArea).focus()
        self.run_worker(self._start_mcp(), exclusive=False)
        self.refresh_status()
        self.update_context()
        self._system_note(
            f"DarkGPT TUI v{APP_VERSION} · Enter send · Shift+Enter newline · "
            "Tab Build/Ask (any focus) · Ctrl+P commands · Ctrl+M MCP · Ctrl+G skills · Ctrl+T tools · Ctrl+L model · "
            "Copy: Ctrl+Alt+C / Alt+C · Paste: Ctrl+Alt+V / Alt+V · "
            "Ctrl+Shift+C/V also work where the terminal passes them through · "
            "Ctrl+O MCP display · Ctrl+C stop/quit\n"
            "WEB: built-in browser is ON (Playwright → Selenium auto-fallback, saved sessions). "
            "The agent tries fetch → search → browser → MCP before reporting failure.\n"
            "STATE: /provider and /status show live provider/usage · /pending lists or clears queued prompts "
            "(prompts sent while busy are queued, never dropped) · /sessions and /resume for checkpoints\n"
            "LEARN: self-learning ON — mistakes become lessons, repeated signals become patterns. "
            "/lessons and /patterns show them · /learn toggles capture · hack skill active (HackSkills router)"
        )

    async def on_unmount(self) -> None:
        try:
            await self.mcp.stop_all()
        except Exception:
            pass

    async def _run_recon_cmd(self, target: str, focus: str = "") -> None:
        from agent import evidence as evidence_util, knowledge_graph as kg_util
        from agent import subagents
        try:
            pool = subagents.SubagentPool(self.model, os.getcwd(), max_concurrent=2, max_total=4)
            workers = ["subdomains", "ports", "tech", "endpoints"]
            prompts = {
                "subdomains": f"Enumerate subdomains and related hosts for target '{target}'. Use read-only tools (shell curl/dig/host/nslookup, web_search patterns). Return lines: HOST <host> <notes>.",
                "ports": f"Identify open ports/services for '{target}' from passive sources and safe probes (curl -I, nc -z if present, nmap if present). Return lines: SERVICE <host> <port> <proto> <name>.",
                "tech": f"Fingerprint web technologies/endpoints for '{target}' (headers, HTML clues, common paths). Return lines: ENDPOINT <url> <notes> and FINDING info <title> :: <details>.",
                "endpoints": f"Discover interesting endpoints/paths for '{target}' (robots, sitemap, js links, common API paths). Return lines: ENDPOINT <url> <notes>.",
            }
            tasks = [("explore", prompts[w] + (f"\nFocus: {focus}" if focus else ""), f"Target: {target}") for w in workers]
            results = await pool.run_many(tasks)
            imported = {"hosts": 0, "services": 0, "endpoints": 0, "findings": 0}
            for r in results:
                for line in (r.text or "").splitlines():
                    line = line.strip()
                    if line.startswith("HOST "):
                        parts = line.split(None, 2)
                        if len(parts) >= 2:
                            kg_util.add_host(target, parts[1], {"notes": parts[2] if len(parts) > 2 else ""}, source="recon")
                            imported["hosts"] += 1
                    elif line.startswith("SERVICE "):
                        parts = line.split(None, 4)
                        if len(parts) >= 4:
                            try:
                                kg_util.add_service(target, parts[1], int(parts[2]), parts[3], {"name": parts[4] if len(parts) > 4 else ""}, source="recon")
                                imported["services"] += 1
                            except Exception:
                                pass
                    elif line.startswith("ENDPOINT "):
                        parts = line.split(None, 2)
                        if len(parts) >= 2:
                            kg_util.add_endpoint(target, parts[1], {"notes": parts[2] if len(parts) > 2 else ""}, source="recon")
                            imported["endpoints"] += 1
                    elif line.startswith("FINDING "):
                        rest = line[len("FINDING "):]
                        sev, _, tail = rest.partition(" ")
                        title, _, details = tail.partition("::")
                        kg_util.add_finding(target, title.strip() or "Recon finding", sev.strip().lower() or "info", details.strip(), source="recon")
                        imported["findings"] += 1
                evidence_util.record(target, tool=f"recon:{r.role}", args={"task": r.task[:200]}, result=r.text or r.error, ok=r.ok, source="recon", tags=["recon", r.role])
            g = kg_util.summary(target)
            c = g.get("counts") or {}
            self._system_note(
                f"Recon complete for {target}. Imported {imported}. "
                f"Graph now: hosts={c.get('host',0)} services={c.get('service',0)} endpoints={c.get('endpoint',0)} findings={c.get('finding',0)}"
            )
        except Exception as e:
            self._system_note(f"Recon failed: {e}")

    async def _start_mcp(self) -> None:
        try:
            await self.mcp.start_from_config(load_config())
        except Exception as exc:  # noqa: BLE001
            self._system_note(f"MCP startup error: {exc}")
        # start_server returns as soon as the task is spawned; wait for every
        # server to settle so the badges, tool list and token meter are real.
        for _ in range(300):
            if not any(s.status == "connecting" for s in self.mcp.servers.values()):
                break
            await asyncio.sleep(0.1)
        self.refresh_status()
        self.update_context()

    # ---------------------------------------------------------------- status
    def _header_left(self) -> str:
        cwd = os.getcwd().replace(str(Path.home()), "~")
        state = "running" if self.busy else "ready"
        return f" {cwd}   DarkGPT v{APP_VERSION} · TUI   {state}"

    def refresh_status(self) -> None:
        try:
            self.query_one("#header-left", Static).update(self._header_left())
            self.query_one("#badge-mcp", Button).label = f"MCP {self.mcp.connected_count()}/{self.mcp.total_count()}"
            self.query_one("#badge-skills", Button).label = f"Skills {self.skills.active_count()}"
            self.query_one("#badge-tools", Button).label = f"Tools {self._enabled_tool_count()}"
            mode_button = self.query_one("#mode-btn", Button)
            mode_button.label = self.mode
            mode_button.set_class(self.mode == "Build", "mode-build")
            mode_button.set_class(self.mode == "Ask", "mode-ask")
            self.query_one("#hint-line", Static).update(
                f" Tab: {self.mode}  ·  Copy: Ctrl+Alt+C/Alt+C  ·  Paste: Ctrl+Alt+V/Alt+V"
                f"  ·  Browser: ON  ·  Learn: {'ON' if self.self_learning_enabled else 'OFF'}"
            )
            self.query_one("#model-btn", Button).label = self.model.split("/")[-1]
            self.query_one("#footer-mcp", Static).update(
                f"MCP {self.mcp.connected_count()}/{self.mcp.total_count()}"
            )
            self.query_one("#footer-skills", Static).update(f"Skills {self.skills.active_count()}")
            turns = len([m for m in self.messages if m.get("role") in ("user", "assistant")])
            label = f"chat #{self.conv_id}" if self.conv_id else "new chat"
            if self._session_id:
                label += f" · save {self._session_id[:8]}"
            self.query_one("#session-current", Static).update(f"{label} · {turns} msg")
        except Exception:
            pass

    # ---------------------------------------------------------- header/footer buttons
    @on(Button.Pressed, "#mode-btn")
    async def _on_mode_button(self) -> None:
        await self.action_toggle_mode()

    @on(Button.Pressed, "#model-btn")
    def _on_model_button(self) -> None:
        self.action_open_model()

    @on(Button.Pressed, "#badge-mcp")
    def _on_mcp_button(self) -> None:
        self.action_open_mcp()

    @on(Button.Pressed, "#badge-skills")
    def _on_skills_button(self) -> None:
        self.action_open_skills()

    @on(Button.Pressed, "#badge-tools")
    def _on_tools_button(self) -> None:
        self.action_open_tools()

    @on(Button.Pressed, "#session-btn")
    def _on_session_button(self) -> None:
        self.action_open_sessions()

    def _enabled_tool_count(self) -> int:
        local = len([s for s in ToolRegistry(self.mcp, disabled=set(), model=self.model).local_specs() if s["function"]["name"] not in self.disabled_tools])
        mcp = len(
            self.mcp.list_tools(
                disabled=self.disabled_tools,
                disabled_servers=self.disabled_servers,
            )
        )
        return local + mcp

    def update_context(self) -> None:
        try:
            tools = ToolRegistry(
                self.mcp,
                disabled=self.disabled_tools,
                disabled_servers=self.disabled_servers,
            ).specs()
            usage = token_util.usage(
                model=self.model,
                system=SYSTEM.format(cwd=os.getcwd()),
                skills=self.skills.active_instructions(),
                tools=tools,
                history=self.messages,
                input_text=self.query_one("#prompt-input", PromptArea).text,
            )
            self._usage = usage
            self.query_one("#token-label", Static).update(token_util.format_usage(usage))
        except Exception:
            pass

    def _system_note(self, text: str) -> None:
        conv = self.query_one("#conversation", VerticalScroll)
        conv.mount(Static(Text(text, style="#888888")))
        conv.scroll_end(animate=False)

    # --------------------------------------------------------------- palette
    @property
    def palette_visible(self) -> bool:
        try:
            return bool(self.query_one("#palette", OptionList).display)
        except Exception:
            return False

    def hide_palette(self) -> None:
        try:
            self.query_one("#palette", OptionList).display = False
        except Exception:
            pass

    def update_palette(self, text: str) -> None:
        if not text.startswith("/") or "\n" in text:
            self.hide_palette()
            return
        query = text[1:].split()[0].lower() if len(text) > 1 else ""
        items = [(cmd, desc) for cmd, desc in COMMANDS if cmd[1:].startswith(query)]
        palette = self.query_one("#palette", OptionList)
        palette.clear_options()
        if not items:
            self.hide_palette()
            return
        palette.add_options([Option(f"{cmd}   {desc}", id=cmd) for cmd, desc in items])
        palette.display = True
        palette.highlighted = 0

    def palette_move(self, delta: int) -> None:
        palette = self.query_one("#palette", OptionList)
        if not palette.options:
            return
        current = palette.highlighted or 0
        palette.highlighted = max(0, min(len(palette.options) - 1, current + delta))

    def palette_select(self) -> None:
        palette = self.query_one("#palette", OptionList)
        option = palette.highlighted_option
        if option:
            self.run_worker(self.execute_command(option.id or ""), exclusive=False)

    @on(OptionList.OptionSelected, "#palette")
    async def _palette_selected(self, event: OptionList.OptionSelected) -> None:
        await self.execute_command(event.option.id or "")

    # ---------------------------------------------------------------- commands
    async def execute_command(self, command: str) -> None:
        self.hide_palette()
        self.query_one("#prompt-input", PromptArea).load_text("")
        parts = (command or "").strip().split()
        name = parts[0] if parts else ""
        arg = parts[1].lower() if len(parts) > 1 else ""
        if name in ("/new", "/clear"):
            await self.action_new_chat()
        elif name == "/model":
            self.open_model()
        elif name == "/provider":
            self.set_provider(arg)
        elif name == "/mcp":
            self.open_mcp()
        elif name == "/skills":
            self.open_skills()
        elif name == "/tools":
            self.open_tools()
        elif name == "/cwd":
            self._system_note(os.getcwd())
        elif name == "/copy":
            self.action_copy_last()
        elif name == "/copy-all":
            self.action_copy_all()
        elif name == "/retry":
            await self.action_retry(force=arg == "force")
        elif name == "/status":
            self._show_status()
        elif name == "/jobs":
            await self._handle_jobs_command(parts)
        elif name in ("/budget", "/steps"):
            await self._handle_budget_command(name, parts)
        elif name == "/proxy":
            await self._handle_proxy_command(parts)
        elif name == "/verify-findings":
            await self._handle_verify_findings_command(parts)
        elif name == "/bb":
            await self._handle_blackboard_command(parts)
        elif name == "/attack-paths":
            await self._handle_attack_paths_command(parts)
        elif name == "/mcp-verbose":
            self.action_toggle_mcp_display()
        elif name == "/bg":
            try:
                bg_index = int(parts[1]) if len(parts) > 1 else None
            except ValueError:
                bg_index = None
            self._show_bg_summary(bg_index)
        elif name == "/clipboard":
            self._system_note(clipboard.diagnostics())
        elif name == "/diff":
            self._system_note(self._last_diff or "No edit in this session yet.")
        elif name == "/undo":
            result = undo.undo(arg or None)
            if result.get("ok"):
                self._system_note(
                    f"Undid {result['id']}: restored " + ", ".join(result.get("restored") or [])
                )
            else:
                self._system_note(f"Undo failed: {result.get('error') or 'nothing to undo'}")
        elif name == "/verify":
            self.verify_enabled = not self.verify_enabled
            appconfig.update(verify_enabled=self.verify_enabled)
            self.notify(f"Auto-verify: {'on' if self.verify_enabled else 'off'}", timeout=2)
        elif name == "/plan":
            self.plan_enabled = not self.plan_enabled
            appconfig.update(plan_enabled=self.plan_enabled)
            self.notify(f"Cheap-model planning: {'on' if self.plan_enabled else 'off'}", timeout=2)
        elif name == "/review":
            self.review_enabled = not self.review_enabled
            appconfig.update(review_enabled=self.review_enabled)
            self.notify(f"Independent review: {'on' if self.review_enabled else 'off'}", timeout=2)
        elif name == "/worktree":
            self.worktree_enabled = not self.worktree_enabled
            appconfig.update(worktree_enabled=self.worktree_enabled)
            self.notify(f"Isolated worktree checks: {'on' if self.worktree_enabled else 'off'}", timeout=2)
        elif name == "/lsp":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() in ("on", "off"):
                self.lsp_enabled = parts[1].lower() == "on"
                appconfig.update(lsp_enabled=self.lsp_enabled)
            py = lsp_util.server_for("probe.py")
            rs = lsp_util.server_for("probe.rs")
            ts = lsp_util.server_for("probe.ts")
            def label(info):
                return info[0][0] if info else "none (regex fallback)"
            self._system_note(
                "LSP integration: " + ("ON" if self.lsp_enabled else "OFF") + "\n"
                f"  Python:  {label(py)}\n"
                f"  TS/JS:   {label(ts)}\n"
                f"  Rust:    {label(rs)}\n"
                "  Tools: lsp_definition · lsp_references · lsp_diagnostics · lsp_rename\n"
                "  Toggle with /lsp on|off. Install pyright / typescript-language-server for full coverage."
            )
        elif name == "/subagents":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() in ("on", "off"):
                self.subagents_enabled = parts[1].lower() == "on"
                appconfig.update(subagents_enabled=self.subagents_enabled)
            self._system_note(
                "Isolated subagents: " + ("ON" if self.subagents_enabled else "OFF")
                + f" · concurrency {appconfig.get('subagent_concurrency', 2)} · max 3/turn\n"
                "  roles: explore · review · test (read-only, isolated context)\n"
                "  Toggle with /subagents on|off. The model delegates via the delegate tool."
            )
        elif name == "/delegates":
            if not self._delegates:
                self._system_note("No subagents ran in the last turn.")
            else:
                self._system_note(
                    "Subagent reports:\n"
                    + "\n".join(
                        f"- [{d.get('role')}] {d.get('task', '')[:70]} — {d.get('preview', '')[:140]}"
                        for d in self._delegates
                    )
                )
        elif name == "/permissions":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() in ("full", "ask", "strict"):
                self.permission_mode = parts[1].lower()
                appconfig.update(permission_mode=self.permission_mode)
                self._session_approvals.clear()
            self._system_note(
                f"Permission mode: {self.permission_mode} · hooks: {'on' if self.hooks_enabled else 'off'}\n"
                "  full: no prompts · ask: edits/shell/MCP · strict: all non-read tools\n"
                "  Switch with /permissions full|ask|strict. Ask mode in the footer forces strict."
            )
        elif name == "/hooks":
            loaded = hooks.load(os.getcwd())
            active = {event: entries for event, entries in loaded.items() if entries}
            if not active:
                self._system_note("No hooks configured. Add ~/.config/darkgpt/hooks.json or <repo>/.darkgpt/hooks.json.\nEvents: " + ", ".join(sorted(hooks.EVENTS)))
            else:
                self._system_note(
                    "Configured hooks:\n"
                    + "\n".join(f"- {event}: {len(entries)}" for event, entries in sorted(active.items()))
                )
        elif name == "/pending":
            parts = (command or "").strip().split()
            if len(parts) > 1 and parts[1].lower() in ("clear", "cancel", "drop"):
                count = len(self._pending)
                self._pending.clear()
                self._system_note(f"Cleared {count} queued prompt(s).")
            elif self._pending:
                lines = [f"{i + 1}. {p}" for i, p in enumerate(self._pending)]
                self._system_note("Queued prompts:\n" + "\n".join(lines))
            else:
                self._system_note("No queued prompts.")
        elif name == "/rules":
            parts = (command or "").strip().split(maxsplit=2)
            if len(parts) >= 3 and parts[1].lower() == "add":
                target = rules_util.add_rule(parts[2], os.getcwd())
                self._system_note(f"Rule appended to {target}")
            else:
                self._system_note(rules_util.show(os.getcwd()))
        elif name == "/compact":
            before = len(self.messages)
            self.messages = context_util.compact(self.messages, keep_recent=6)
            self.update_context()
            self._system_note(f"Compacted {before} messages → {len(self.messages)} (older turns summarised).")
        elif name == "/memory":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() == "off":
                self.memory_enabled = False
                appconfig.update(memory_enabled=False)
                self._system_note("Project memory: off")
            else:
                entries = memory_util.recent(os.getcwd(), 8)
                self._system_note(
                    "Project memory:\n"
                    + ("\n".join(f"- {str(e.get('text'))[:200]}" for e in entries) or "(empty)")
                )
        elif name == "/lessons":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() in ("clear", "forget", "reset"):
                removed = memory_util.forget_kinds(
                    os.getcwd(), {"lesson", "failed-approach", "correction"}
                )
                self._system_note(f"Cleared {removed} learned lesson(s).")
            else:
                entries = memory_util.by_kind(
                    os.getcwd(), {"lesson", "failed-approach", "correction"}, k=20
                )
                self._system_note(
                    memory_util.format_lessons(entries)
                    or "No lessons learned yet. They accumulate as tools fail or are blocked."
                )
        elif name == "/patterns":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() in ("clear", "forget", "reset"):
                removed = memory_util.forget_kinds(os.getcwd(), {"pattern"})
                self._system_note(f"Cleared {removed} learned pattern(s).")
            else:
                entries = memory_util.by_kind(os.getcwd(), {"pattern"}, k=20)
                self._system_note(
                    memory_util.format_patterns(entries)
                    or "No patterns learned yet. Repeated failure signatures become patterns."
                )
        elif name == "/learn":
            parts = (command or "").strip().split(maxsplit=1)
            value = parts[1].strip().lower() if len(parts) > 1 else ""
            if value in ("on", "off"):
                enabled = value == "on"
            else:
                enabled = not bool(getattr(self, "self_learning_enabled", True))
            self.self_learning_enabled = enabled
            appconfig.update(self_learning_enabled=enabled)
            self._system_note(f"Self-learning: {'on' if enabled else 'off'}")
        elif name == "/target":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1:
                self.active_target = parts[1].strip()
                appconfig.update(active_target=self.active_target)
                self._system_note(f"Active target: {self.active_target}")
            else:
                self._system_note(f"Active target: {self.active_target or '(none)'}\nUse /target <name-or-domain> to set it.")
        elif name == "/targets":
            from agent import knowledge_graph as kg_util
            items = kg_util.list_targets(30)
            if not items:
                self._system_note("No targets yet. Set one with /target <name> and run recon.")
            else:
                lines = ["Known targets:"]
                for t in items:
                    c = t.get("counts") or {}
                    lines.append(f"- {t.get('target')} · hosts={c.get('host',0)} services={c.get('service',0)} endpoints={c.get('endpoint',0)} findings={c.get('finding',0)}")
                self._system_note("\n".join(lines))
        elif name == "/kg":
            from agent import knowledge_graph as kg_util
            target = self.active_target
            if not target:
                self._system_note("No active target. Use /target <name> first.")
            else:
                raw = (command or "").split(maxsplit=2)
                arg = raw[1].strip().lower() if len(raw) > 1 else "summary"
                rest = raw[2].strip() if len(raw) > 2 else ""
                if arg in ("hosts", "services", "endpoints", "findings", "credentials", "auth_states", "params", "fingerprints"):
                    kind = {"endpoints": "endpoint", "credentials": "credential", "auth_states": "auth_state", "params": "param", "fingerprints": "fingerprint"}.get(arg, arg[:-1] if arg.endswith("s") else arg)
                    nodes = kg_util.query(target, kind=kind, limit=50)
                    if not nodes:
                        self._system_note(f"No {arg} for {target}.")
                    else:
                        lines = [f"{arg} for {target}:"]
                        for n in nodes[:50]:
                            a = n.get("attrs") or {}
                            label = a.get("host") or a.get("url") or a.get("title") or a.get("name") or a.get("fingerprint") or n.get("id")
                            extra = ""
                            if n.get("kind") == "service": extra = f" ({a.get('name') or a.get('product') or ''})"
                            if n.get("kind") == "finding": extra = f" [{a.get('severity','info')}/{a.get('status','attempted')}]"
                            if n.get("kind") == "auth_state": extra = f" [{a.get('state','')}]"
                            lines.append(f"- {label}{extra}")
                        self._system_note("\n".join(lines))
                elif arg == "query":
                    nodes = kg_util.query(target, text=rest, limit=30)
                    if not nodes:
                        self._system_note(f"No KG nodes matching '{rest}'.")
                    else:
                        lines = [f"KG query '{rest}':"]
                        for n in nodes:
                            a = n.get("attrs") or {}
                            label = a.get("host") or a.get("url") or a.get("title") or a.get("name") or n.get("id")
                            lines.append(f"- [{n.get('kind')}] {label}")
                        self._system_note("\n".join(lines))
                elif arg == "export":
                    fmt = (rest or "json").lower().strip()
                    try:
                        res = kg_util.export_graph(target, fmt)
                        self._system_note(f"KG exported ({res.get('format')}): {res.get('path')}")
                    except Exception as e:
                        self._system_note(f"KG export failed: {e}")
                elif arg == "chain":
                    nodes = kg_util.query(target, kind="finding", text=rest, limit=10)
                    if not nodes:
                        self._system_note(f"No findings matching '{rest}'.")
                    else:
                        g = kg_util._load(target)
                        lines = [f"Finding chains for '{rest}':"]
                        for n in nodes:
                            a = n.get("attrs") or {}
                            lines.append(f"- [{a.get('severity','info')}/{a.get('status','attempted')}] {a.get('title') or n.get('id')}")
                            for e in g.get("edges") or []:
                                if e.get("src") == n.get("key") or e.get("dst") == n.get("key"):
                                    other = e.get("dst") if e.get("src") == n.get("key") else e.get("src")
                                    on = (g.get("nodes") or {}).get(other) or {}
                                    oa = on.get("attrs") or {}
                                    olabel = oa.get("host") or oa.get("url") or oa.get("title") or oa.get("name") or on.get("id")
                                    lines.append(f"  - {e.get('rel')} -> [{on.get('kind')}] {olabel}")
                        self._system_note("\n".join(lines))
                else:
                    s = kg_util.summary(target)
                    c = s.get("counts") or {}
                    self._system_note(
                        f"KG {s.get('target')}\n"
                        f"hosts={c.get('host',0)} services={c.get('service',0)} endpoints={c.get('endpoint',0)} findings={c.get('finding',0)} edges={s.get('edges',0)}\n"
                        f"graph: {s.get('path')}\nUse /kg hosts|services|endpoints|findings|query <text>|chain <text>|export json|dot"
                    )
        elif name == "/evidence":
            from agent import evidence as evidence_util
            target = self.active_target
            if not target:
                self._system_note("No active target. Use /target <name> first.")
            else:
                s = evidence_util.summary(target)
                lines = [f"Evidence for {target}: {s.get('count',0)} entries (ok={s.get('ok',0)} failed={s.get('failed',0)})", f"ledger: {s.get('path')}"]
                for tool, count in list((s.get("by_tool") or {}).items())[:15]:
                    lines.append(f"- {tool}: {count}")
                self._system_note("\n".join(lines))
        elif name == "/report":
            from agent import reporting as reporting_util
            target = self.active_target
            if not target:
                self._system_note("No active target. Use /target <name> first.")
            else:
                parts = (command or "").strip().split()
                fmt = parts[1].lower() if len(parts) > 1 and parts[1].lower() in ("md", "markdown", "html", "htm", "json", "csv") else "markdown"
                out = parts[2] if len(parts) > 2 else ""
                try:
                    res = reporting_util.write_report(target, fmt=fmt, out=out)
                    self._system_note(f"Report written ({res.get('format')}): {res.get('path')} ({res.get('bytes',0)} bytes)")
                except Exception as e:
                    self._system_note(f"Report failed: {e}")
        elif name == "/recon":
            target = self.active_target
            if not target:
                self._system_note("No active target. Use /target <name> first.")
            else:
                focus = (command or "").strip().split(maxsplit=1)[1] if len((command or "").strip().split(maxsplit=1)) > 1 else ""
                self._system_note(f"Recon orchestration started for {target}…")
                self.run_worker(self._run_recon_cmd(target, focus), exclusive=False)
        elif name == "/route":
            parts = (command or "").strip().split(maxsplit=1)
            self.cheap_model = parts[1].strip() if len(parts) > 1 else ""
            appconfig.update(cheap_model=self.cheap_model)
            resolved = routing.pick_cheap_model(self.model, self.cheap_model)
            self._system_note(f"Cheap routing model: {resolved}")
        elif name == "/power":
            resolved = routing.pick_cheap_model(self.model, self.cheap_model)
            self._system_note(
                "Coding power settings:\n"
                f"  verify: {'on' if self.verify_enabled else 'off'}\n"
                f"  plan:   {'on' if self.plan_enabled else 'off'}\n"
                f"  review: {'on' if self.review_enabled else 'off'}\n"
                f"  memory: {'on' if self.memory_enabled else 'off'}\n"
                f"  worktree: {'on' if self.worktree_enabled else 'off'}\n"
                f"  lsp:       {'on' if self.lsp_enabled else 'off'}\n"
                f"  subagents: {'on' if self.subagents_enabled else 'off'}\n"
                f"  permissions: {self.permission_mode}"
                f"{' (Ask mode forces strict)' if self.mode == 'Ask' else ''}\n"
                f"  hooks:      {'on' if self.hooks_enabled else 'off'}\n"
                f"  checkpoint: {'on' if self.checkpoint_enabled else 'off'}\n"
                f"  scripts:    {'on' if self.scripts_enabled else 'off'}\n"
                f"  react:      {'on' if self.allow_react_fallback else 'off (native only)'}\n"
                f"  cheap model: {resolved}"
            )
        elif name == "/react":
            parts = (command or "").strip().split(maxsplit=1)
            if len(parts) > 1 and parts[1].lower() in ("on", "off"):
                self.allow_react_fallback = parts[1].lower() == "on"
                appconfig.update(allow_react_fallback=self.allow_react_fallback)
            self._system_note(
                "ReAct text-tool fallback: " + ("ON" if self.allow_react_fallback else "OFF")
                + "\n  Native tool calling is always attempted first, then retried with relaxed"
                "\n  provider routing and a minimal tool set before any fallback."
                "\n  Switch with /react on|off. Native-only mode fails clearly instead of downgrading."
            )
        elif name == "/scripts":
            parts = (command or "").strip().split(maxsplit=2)
            if len(parts) >= 3 and parts[1].lower() == "show":
                self._system_note(scripting.show_script(parts[2]))
            else:
                scripts = scripting.recent(12)
                if not scripts:
                    self._system_note(
                        "No self-written scripts yet. The model creates them with the script tool "
                        "for bulk edits, generation and automation. Scripts are kept under "
                        + str(scripting.SCRIPT_DIR)
                    )
                else:
                    self._system_note(
                        "Self-written scripts (kept as evidence):\n"
                        + "\n".join(
                            f"- {s['language']:6} {s['created']}  {s['bytes']:>7}B  {s['path']}"
                            for s in scripts
                        )
                        + "\nInspect one with /scripts show <path>."
                    )
        elif name == "/checkpoint":
            entry = session_util.create(
                os.getcwd(), self.messages, model=self.model,
                label=(command or "").strip().split(maxsplit=1)[1] if len((command or "").strip().split(maxsplit=1)) > 1 else "",
                last_prompt=self.last_prompt,
            )
            if entry:
                self._session_id = entry["id"]
                self._system_note(f"Checkpoint saved: {entry['id']} ({entry['turns']} turns)")
            else:
                self._system_note("Checkpoint failed (no messages to save?).")
        elif name == "/sessions":
            sessions = session_util.list_sessions(os.getcwd(), 10)
            if not sessions:
                self._system_note("No session checkpoints yet. Use /checkpoint or let a successful turn auto-save.")
            else:
                self._system_note(
                    "Session checkpoints:\n"
                    + "\n".join(
                        f"- {s['id']} · {s['turns']} turns · {s['last_user'][:60]}"
                        for s in sessions
                    )
                    + "\nRestore with /resume <id> or /resume latest."
                )
        elif name == "/resume":
            parts = (command or "").strip().split(maxsplit=1)
            selector = parts[1] if len(parts) > 1 else "latest"
            entry = session_util.resolve(selector, os.getcwd())
            if not entry:
                self._system_note(f"No session found for '{selector}'. Use /sessions to list.")
            else:
                restored = session_util.replay_messages(entry)
                self.messages = restored
                self.last_prompt = str(entry.get("last_prompt") or "")
                self.last_answer = str((entry.get("checkpoint") or {}).get("last_assistant") or "")
                self.conv_id = None
                conv = self.query_one("#conversation", VerticalScroll)
                await conv.remove_children()
                for message in restored:
                    await conv.mount(ChatMessage(message["role"], message["content"][:6000]))
                self._session_id = str(entry.get("id") or "")
                self.update_context()
                self._system_note(
                    session_util.summary_text(entry)
                    + "\n\nRestored text history only — tools are NOT re-executed."
                    + f"\nCurrent model unchanged: {self.model}"
                )
        elif name == "/exit":
            self.exit()
        else:
            self._system_note(f"unknown command: {command}")

    # Textual resolves BINDINGS to action_<name>; keep aliases so both the
    # keyboard shortcuts and the slash commands / button presses reach these.
    def action_open_model(self) -> None:
        self.open_model()

    def action_open_mcp(self) -> None:
        self.open_mcp()

    def action_open_skills(self) -> None:
        self.open_skills()

    def action_open_tools(self) -> None:
        self.open_tools()

    def action_open_sessions(self) -> None:
        self.open_sessions()

    async def action_handle_ctrl_c(self) -> None:
        await self.handle_ctrl_c()

    def open_model(self) -> None:
        self.push_screen(
            ModelPickerScreen(self.model, active_provider=current_provider()),
            self._model_picked,
        )

    def set_provider(self, name: str) -> None:
        """Switch chat backend and align the model with it."""
        name = (name or "").strip().lower()
        if not name:
            cfg = provider_config()
            lines = [f"active provider: {cfg['label']} ({current_provider()})",
                     "usage: /provider openrouter | nvidia"]
            for key, meta in PROVIDERS.items():
                have = "set" if get_api_key(provider=key) else "MISSING"
                lines.append(f"  {key:11} {meta['label']:12} key: {have} · {meta['env_key']}")
            lines.append(f"add keys to {os.path.join(PROJECT_ROOT, '.env')}, then /provider <name>")
            self._system_note("\n".join(lines))
            return
        if name not in PROVIDERS:
            self._system_note(
                f"unknown provider: {name}. Use /provider openrouter or /provider nvidia."
            )
            return
        previous = current_provider()
        cfg = provider_config(name)
        # Remember the model per provider: switching back restores what the
        # user chose there instead of silently replacing it with a default.
        appconfig.update(provider=name, **{f"model_{previous}": self.model})
        if name != previous:
            saved = str(appconfig.get(f"model_{name}") or "").strip()
            target = saved or cfg["default_model"]
            self.model = target
            appconfig.update(model=self.model)
        if self._cooldown_until:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            appconfig.update(cooldown_until=0, cooldown_reason="")
        self.refresh_status()
        self.update_context()
        lines = [f"provider: {cfg['label']}", f"model: {self.model}"]
        if get_api_key(provider=name):
            lines.append("api key: set")
        else:
            lines.append(
                f"api key: MISSING — add {cfg['env_key']}=... to "
                f"{os.path.join(PROJECT_ROOT, '.env')} (key: {cfg['signup']})"
            )
        lines.append("use /model to pick a different model for this provider")
        self._system_note("\n".join(lines))

    def _model_picked(self, selected: Optional[Any]) -> None:
        if not selected:
            return
        if isinstance(selected, tuple) and len(selected) == 2:
            provider, model = selected
        else:
            provider, model = None, selected  # legacy bare model id
        if not model:
            return
        previous = current_provider()
        if provider in PROVIDERS and provider != previous:
            # The selected model belongs to the other backend: move to it and
            # remember the model we came from so /provider restores it.
            appconfig.update(provider=provider, **{f"model_{previous}": self.model})
        self.model = model
        if provider in PROVIDERS:
            appconfig.update(**{f"model_{provider}": model})
        appconfig.update(model=model)
        if self._cooldown_until:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            appconfig.update(cooldown_until=0, cooldown_reason="")
        self.refresh_status()
        self.update_context()
        note = f"model: {model}"
        if provider in PROVIDERS and provider != previous:
            note += f" · provider: {provider_config(provider)['label']}"
        self._system_note(note)

    def open_mcp(self) -> None:
        self.push_screen(MCPManagerScreen(), lambda _: self._after_manager())

    def open_skills(self) -> None:
        self.push_screen(SkillsManagerScreen(), lambda _: self._after_manager())

    def open_tools(self) -> None:
        self.push_screen(ToolsManagerScreen(), lambda _: self._after_manager())

    def open_sessions(self) -> None:
        self.push_screen(SessionPickerScreen(), self._session_picked)

    def _session_picked(self, selector: Optional[str]) -> None:
        if not selector:
            return
        self.run_worker(self._resume_selector(selector), exclusive=False)

    async def _resume_selector(self, selector: str) -> None:
        """Restore a DB conversation or checkpoint chosen from the session bar."""
        try:
            kind, ident = selector.split(":", 1)
        except ValueError:
            return
        if kind == "db":
            try:
                conv_id = int(ident)
            except ValueError:
                return
            rows = await asyncio.to_thread(db.get_messages, conv_id)
            restored = [
                {"role": str(r.get("role") or "user"), "content": str(r.get("content") or "")}
                for r in rows
            ]
            if not restored:
                self._system_note("That chat has no messages.")
                return
            self.messages = restored
            self.conv_id = conv_id
            self.last_prompt = ""
            self.last_answer = ""
            for message in reversed(restored):
                if message["role"] == "assistant" and message["content"]:
                    self.last_answer = message["content"]
                    break
            for message in reversed(restored):
                if message["role"] == "user" and message["content"]:
                    self.last_prompt = message["content"]
                    break
            conv = self.query_one("#conversation", VerticalScroll)
            await conv.remove_children()
            for message in restored:
                await conv.mount(ChatMessage(message["role"], message["content"][:6000]))
            self._session_id = ""
            self.refresh_status()
            self.update_context()
            self._system_note(f"Restored chat #{conv_id} ({len(restored)} messages). Model unchanged: {self.model}")
        elif kind == "cp":
            await self.execute_command(f"/resume {ident}")

    def _after_manager(self, _result=None) -> None:
        self.refresh_status()
        self.update_context()

    @work(exclusive=False)
    async def action_command_palette(self) -> None:
        # Ctrl+P opens the same filterable command list as a modal-free overlay.
        self.query_one("#prompt-input", PromptArea).load_text("/")
        self.update_palette("/")
        self.query_one("#prompt-input", PromptArea).focus()

    # --------------------------------------------------------------- input keys
    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        # Always display the real text. An older optimisation replaced pastes
        # of 5+ lines with a "[Pasted ~N lines]" placeholder, which made paste
        # look broken and corrupted multi-line input.
        area = event.text_area
        area._collapsed = False  # type: ignore[attr-defined]
        area._full_text = ""  # type: ignore[attr-defined]
        area._paste_placeholder = ""  # type: ignore[attr-defined]
        self.update_palette(area.text)

    async def submit_prompt(self) -> None:
        area = self.query_one("#prompt-input", PromptArea)
        text = getattr(area, "_full_text", None) or area.text
        text = text.strip()
        area._collapsed = False  # type: ignore[attr-defined]
        area._full_text = ""  # type: ignore[attr-defined]
        area._paste_placeholder = ""  # type: ignore[attr-defined]
        area.load_text("")
        self.hide_palette()
        if not text:
            return
        if self.busy:
            # Never drop input typed during a running turn. Remember it and
            # replay it in order as soon as the current turn finishes.
            self._pending.append(text)
            self._system_note(
                f"Queued ({len(self._pending)}): {text[:90]}\n"
                "  Runs automatically when this turn finishes. /pending to view or clear."
            )
            self.refresh_status()
            return
        if text.startswith("/"):
            await self.execute_command(text)
            return
        self.last_prompt = text
        self._worker = self.run_prompt(text)

    async def _drain_pending(self) -> None:
        """Run prompts submitted while the previous turn was still active."""
        if self.busy or not self._pending:
            return
        text = self._pending.pop(0)
        self._system_note(f"Running queued: {text[:90]}")
        if text.startswith("/"):
            await self.execute_command(text)
            # A command such as /retry may have started a new worker. If it
            # did, that worker's completion will drain the rest.
            if not self.busy and self._pending:
                self.call_later(self._drain_pending)
            return
        self.last_prompt = text
        self._worker = self.run_prompt(text)

    async def action_toggle_mode(self) -> None:
        # While a modal is open, Tab must keep navigating the modal instead of
        # silently flipping the agent mode behind it.
        if len(getattr(self, "screen_stack", ())) > 1:
            return
        new_mode = "Ask" if self.mode == "Build" else "Build"
        self.mode = new_mode
        appconfig.update(mode=new_mode)
        self.refresh_status()
        self._system_note(
            f"Mode: {new_mode}"
            + (" (read-only, no shell/write tools)" if new_mode == "Ask" else " (full tools)")
        )

    def watch_mode(self, mode: str) -> None:
        """Keep the footer button in sync even when mode is set outside the toggle."""
        try:
            button = self.query_one("#mode-btn", Button)
            button.label = mode
            button.set_class(mode == "Build", "mode-build")
            button.set_class(mode == "Ask", "mode-ask")
            self.refresh_status()
        except Exception:
            pass

    async def action_new_chat(self) -> None:
        # Preserve the thread before clearing — nothing is ever silently lost.
        if self.messages:
            entry = session_util.create(
                os.getcwd(), self.messages, model=self.model,
                label="before /new", last_prompt=self.last_prompt,
            )
            if entry:
                self._system_note(f"Saved current chat as {entry['id']} before starting a new one.")
        self.conv_id = None
        self.messages = []
        self.last_answer = ""
        self.last_prompt = ""
        self._session_id = ""
        conv = self.query_one("#conversation", VerticalScroll)
        await conv.remove_children()
        self._system_note("New chat · open Sessions at the top to restore any earlier chat")
        self.refresh_status()
        self.update_context()

    def _cooldown_remaining(self) -> int:
        if not self._cooldown_until:
            return 0
        remaining = int(round(self._cooldown_until - time.time()))
        if remaining <= 0:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            appconfig.update(cooldown_until=0, cooldown_reason="")
            return 0
        return remaining

    def _playwright_status(self) -> str:
        cached = getattr(self, "_playwright_status_cache", None)
        now = time.time()
        if isinstance(cached, tuple) and now - cached[0] < 60:
            return cached[1]
        status = "unknown"
        try:
            import subprocess, json as _json
            code = (
                "from playwright.sync_api import sync_playwright;"
                "p=sync_playwright().start(); b=p.chromium.launch(headless=True); b.close(); p.stop(); print('ok')"
            )
            proc = subprocess.run(["/usr/bin/python3", "-c", code], capture_output=True, text=True, timeout=20)
            if proc.returncode == 0 and "ok" in proc.stdout:
                status = "OK"
            else:
                err = (proc.stderr or proc.stdout or "unavailable").strip().splitlines()[-1][:120]
                status = f"DEGRADED (fallback selenium) · {err}"
        except Exception as e:
            status = f"DEGRADED (fallback selenium) · {type(e).__name__}"
        self._playwright_status_cache = (now, status)
        return status

    async def _handle_budget_command(self, name: str, parts: List[str]) -> None:
        from agent.core import DEFAULT_TOOLS_PER_TURN, DEFAULT_MAX_STEPS
        # /steps 40, /budget steps 40, /budget tools 60, /budget
        if name == "/steps":
            args = parts[1:]
            key = "max_steps"
        else:
            args = parts[1:]
            key = None
            if args and args[0].lower() in ("tools", "tools_per_turn", "tool"):
                key, args = "tools_per_turn", args[1:]
            elif args and args[0].lower() in ("steps", "max_steps", "step"):
                key, args = "max_steps", args[1:]
        tools_per_turn = int(appconfig.get("tools_per_turn", DEFAULT_TOOLS_PER_TURN) or DEFAULT_TOOLS_PER_TURN)
        max_steps = int(appconfig.get("max_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS)
        if not args:
            self._system_note(
                f"Budget: tools_per_turn={tools_per_turn} (10-200), max_steps={max_steps} (4-200)\n"
                "Usage: /budget tools <n> | /budget steps <n> | /steps <n>"
            )
            return
        try:
            value = int(args[0])
        except Exception:
            self._system_note("Budget value must be an integer.")
            return
        if key == "tools_per_turn" or (key is None and args and name == "/budget" and len(parts) > 1 and parts[1].isdigit()):
            key = "tools_per_turn"
        if key is None:
            self._system_note("Use /budget tools <n> or /budget steps <n>.")
            return
        if key == "tools_per_turn":
            value = max(10, min(value, 200))
            appconfig.update(tools_per_turn=value)
            self._system_note(f"tools_per_turn set to {value}. Soft budget: agent summarizes instead of hard-failing.")
        else:
            value = max(4, min(value, 200))
            appconfig.update(max_steps=value)
            self._system_note(f"max_steps set to {value}. Restart not required; takes effect next turn.")

    async def _handle_jobs_command(self, parts: List[str]) -> None:
        # /jobs, /jobs <id>, /jobs kill <id>, /jobs log <id>
        if not hasattr(self, "_job_manager") or self._job_manager is None:
            self._job_manager = jobs.JobManager()
        args = parts[1:]
        if not args:
            data = self._job_manager.list(limit=30)
            rows = data.get("jobs", [])
            if not rows:
                self._system_note("No background jobs. Agent can start them with task_start; jobs persist across TUI restarts under ~/.cache/darkgpt/jobs.")
                return
            lines = ["Background jobs (most recent):"]
            for j in rows[:30]:
                dur = j.get("duration")
                dur_s = f"{dur:.1f}s" if isinstance(dur, (int,float)) else "?"
                lines.append(f"- {j.get('id')} [{j.get('status')}] exit={j.get('exit_code')} {dur_s} kind={j.get('kind')} :: {str(j.get('label',''))[:80]}")
            lines.append("Use /jobs <id> for tail log, /jobs kill <id> to stop.")
            self._system_note("\n".join(lines))
            return
        if args[0].lower() == "kill" and len(args) > 1:
            res = await self._job_manager.kill(args[1])
            if not res.get("ok"):
                self._system_note(f"Kill failed: {res.get('error')}")
            else:
                j = res.get("job", {})
                self._system_note(f"Killed {j.get('id')} status={j.get('status')} exit={j.get('exit_code')}")
            return
        if args[0].lower() == "log" and len(args) > 1:
            jid = args[1]
        else:
            jid = args[0]
        res = self._job_manager.log(jid, tail=6000)
        if not res.get("ok"):
            self._system_note(f"Job log failed: {res.get('error')}")
            return
        j = res.get("job", {})
        out = str(j.get("output") or "(no output)")
        self._system_note(f"Job {j.get('id')} [{j.get('status')}] exit={j.get('exit_code')}\n{out[-6000:]}")

    async def _handle_proxy_command(self, parts: List[str]) -> None:
        from agent import proxy as proxy_util, knowledge_graph as kg_util
        if not hasattr(self, "_proxy_manager") or self._proxy_manager is None:
            self._proxy_manager = proxy_util.ProxyManager()
        args = parts[1:]
        action = (args[0].lower() if args else "status")
        target = str(appconfig.get("active_target") or "default")
        if action == "start":
            port = int(args[1]) if len(args) > 1 and args[1].isdigit() else 8080
            res = await self._proxy_manager.start(target, port=port)
        elif action == "stop":
            res = await self._proxy_manager.stop()
        elif action == "import":
            res = self._proxy_manager.export_to_graph(target=target, limit=int(args[1]) if len(args) > 1 and args[1].isdigit() else 500)
        else:
            res = self._proxy_manager.status()
        self._system_note(json.dumps(res, indent=2, ensure_ascii=False)[:6000])

    async def _handle_verify_findings_command(self, parts: List[str]) -> None:
        target = str(appconfig.get("active_target") or "").strip()
        if not target:
            self._system_note("No active target. Use /target <name> first.")
            return
        from agent import knowledge_graph as kg_util, evidence as ev, reporting as rep
        args = parts[1:]
        action = args[0].lower() if args else "list"
        findings = kg_util.query(target, kind="finding", limit=200)
        if action == "list":
            rows = []
            for f in findings[:50]:
                a = f.get("attrs") or {}
                rows.append(f"- {a.get('title')} [{a.get('status')}/{a.get('severity')}] repro={bool(a.get('repro_script') or a.get('repro_command'))}")
            self._system_note("Findings:\n" + ("\n".join(rows) if rows else "(none)"))
            return
        # replay/replay_all: minimal local execution
        selected = findings if action == "replay_all" else findings[:1]
        results = []
        for f in selected:
            a = dict(f.get("attrs") or {})
            cmd = a.get("repro_command")
            script = a.get("repro_script")
            if not cmd and not script:
                results.append(f"{a.get('title')}: skipped (no repro)")
                continue
            if script:
                p = Path(script)
                if not p.is_absolute(): p = kg_util.target_dir(target) / script
                shell_cmd = f"bash {p}" if p.suffix == ".sh" else f"python3 {p}"
            else:
                shell_cmd = str(cmd)
            proc = await asyncio.create_subprocess_shell(shell_cmd, cwd=os.getcwd(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await proc.communicate()
            verdict = "still_vulnerable" if proc.returncode == 0 else "fixed_or_not_reproduced"
            a["last_replay_verdict"] = verdict
            a["last_replay_exit"] = proc.returncode
            a["status"] = "proved" if verdict == "still_vulnerable" else a.get("status", "attempted")
            kg_util.upsert_node(target, "finding", str(a.get("title") or f.get("id")), a, source="verify_findings")
            ev.record(target, tool="verify_findings", args={"finding": a.get("title")}, result=out.decode("utf-8",errors="replace")[-4000:], ok=proc.returncode==0, source="verify_findings", tags=["replay", verdict])
            results.append(f"{a.get('title')}: {verdict} (exit {proc.returncode})")
        self._system_note("\n".join(results) or "No findings selected.")

    async def _handle_blackboard_command(self, parts: List[str]) -> None:
        target = str(appconfig.get("active_target") or "").strip()
        if not target:
            self._system_note("No active target. Use /target <name> first.")
            return
        from agent import blackboard as bb
        args = parts[1:]
        if not args:
            self._system_note(json.dumps(bb.summary(target), indent=2, ensure_ascii=False)[:6000])
            return
        if args[0].lower() == "post" and len(args) >= 3:
            kind = args[1].lower(); text = " ".join(args[2:])
            item = bb.post(target, kind, text, source="operator")
            self._system_note(json.dumps(item, indent=2, ensure_ascii=False))
            return
        self._system_note("Usage: /bb | /bb post <fact|intent|hint> <text>")

    async def _handle_attack_paths_command(self, parts: List[str]) -> None:
        target = str(appconfig.get("active_target") or "").strip()
        if not target:
            self._system_note("No active target. Use /target <name> first.")
            return
        from agent import knowledge_graph as kg_util
        paths = kg_util.rank_attack_paths(target, max_paths=10, max_depth=5)
        if not paths:
            self._system_note("No attack paths ranked yet.")
            return
        lines = [f"Attack paths for {target}:"]
        for p in paths[:10]:
            seq = " -> ".join(f"{n.get('kind')}:{n.get('label')}" for n in p.get("nodes", []))
            lines.append(f"- score={p.get('score')} :: {seq}")
        self._system_note("\n".join(lines))

    def _show_status(self) -> None:
        remaining = self._cooldown_remaining()
        label = provider_config()["label"]
        try:
            key_state = "set" if get_api_key() else "MISSING (see /provider)"
        except Exception:
            key_state = "unknown"
        try:
            from agent.core import DEFAULT_TOOLS_PER_TURN, DEFAULT_MAX_STEPS
        except Exception:
            DEFAULT_TOOLS_PER_TURN, DEFAULT_MAX_STEPS = 40, 28
        tools_per_turn = int(appconfig.get("tools_per_turn", DEFAULT_TOOLS_PER_TURN) or DEFAULT_TOOLS_PER_TURN)
        max_steps = int(appconfig.get("max_steps", DEFAULT_MAX_STEPS) or DEFAULT_MAX_STEPS)
        pw = self._playwright_status()
        bg_jobs = self._job_manager.list(limit=200).get("jobs", []) if hasattr(self, "_job_manager") else []
        running_jobs = len([j for j in bg_jobs if j.get("status") == "running"])
        lines = [
            f"model: {self.model}",
            f"provider: {label} · key: {key_state}",
            f"mode: {self.mode}",
            f"budget: tools_per_turn={tools_per_turn} · max_steps={max_steps} (/budget tools N, /steps N)",
            f"mcp: {self.mcp.connected_count()}/{self.mcp.total_count()} connected · "
            f"tools: {self._enabled_tool_count()}",
            f"skills: {self.skills.active_count()} active",
            f"browser: selenium=OK · playwright={pw}",
            f"jobs: {running_jobs} running / {len(bg_jobs)} total (/jobs)",
            f"rules: {len(rules_util.load(os.getcwd()))} file(s) loaded",
            f"sessions: {len(session_util.list_sessions(os.getcwd(), 5))} saved · "
            f"auto-checkpoint: {'on' if self.checkpoint_enabled else 'off'}",
            f"scripts: {len(scripting.recent(5))} recent · "
            f"react fallback: {'on' if self.allow_react_fallback else 'off (native only)'}",
            f"context: {token_util.format_usage(self._usage)}",
            f"mcp display: {self.mcp_display} (Ctrl+O or /mcp-verbose)",
            f"power: verify={'on' if self.verify_enabled else 'off'} · "
            f"plan={'on' if self.plan_enabled else 'off'} · "
            f"review={'on' if self.review_enabled else 'off'} · "
            f"memory={'on' if self.memory_enabled else 'off'} · "
            f"worktree={'on' if self.worktree_enabled else 'off'} · "
            f"lsp={'on' if self.lsp_enabled else 'off'} · "
            f"agents={'on' if self.subagents_enabled else 'off'} · "
            f"perm={'strict' if self.mode == 'Ask' else self.permission_mode}",
            f"cheap model: {routing.pick_cheap_model(self.model, self.cheap_model)}",
        ]
        if remaining:
            lines.append(
                f"cooldown: {remaining}s left ({self._cooldown_reason or 'rate limit'})"
            )
        if self._last_failed:
            lines.append(f"last failed turn: {self._last_failed[:60]}")
        if self._pending:
            lines.append(f"queued: {len(self._pending)} prompt(s) · /pending to view or clear")
        self._system_note("\n".join(lines))

    async def action_retry(self, force: bool = False) -> None:
        prompt = self.last_prompt or self._last_failed
        if not prompt or self.busy:
            return
        remaining = self._cooldown_remaining()
        if remaining > 0 and not force:
            self._system_note(
                f"Rate-limit cooldown active: {remaining}s left. Use /retry force to override, "
                "/provider nvidia to leave OpenRouter's free cap, or wait."
            )
            return
        if force:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            appconfig.update(cooldown_until=0, cooldown_reason="")
        self._system_note("Retrying last turn…")
        self._worker = self.run_prompt(prompt, echo_user=False)

    async def action_escape(self) -> None:
        if self.palette_visible:
            self.hide_palette()
            return
        if self.busy:
            await self.action_interrupt()

    async def action_interrupt(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.busy = False
            self._system_note("Interrupted")

    async def handle_ctrl_c(self) -> None:
        selected = None
        try:
            selected = self.screen.get_selected_text()
        except Exception:
            selected = None
        if not selected:
            try:
                selected = self.query_one("#prompt-input", PromptArea).selected_text
            except Exception:
                selected = None
        if selected:
            result = clipboard.copy_text(selected, app=self)
            if result.ok:
                self._system_note(f"Copied selection · {result.backend}")
            else:
                self._system_note("Could not access the system clipboard (install xclip/xsel)")
            return
        if self.busy:
            await self.action_interrupt()
            self._system_note("Interrupted. Press Ctrl+C again to exit.")
            self._ctrl_c_once = True
            return
        if getattr(self, "_ctrl_c_once", False):
            self.exit()
        else:
            self._ctrl_c_once = True
            self._system_note("Press Ctrl+C again to exit.")

    @on(events.TextSelected)
    def _auto_copy_selection(self, event: events.TextSelected) -> None:
        """Copy highlighted text immediately, like OpenCode does."""
        text = ""
        try:
            text = self.screen.get_selected_text() or ""
        except Exception:
            text = ""
        if not text:
            try:
                text = self.query_one("#prompt-input", PromptArea).selected_text or ""
            except Exception:
                text = ""
        text = (text or "").strip()
        if not text:
            return
        result = clipboard.copy_text(text, app=self)
        if result.ok:
            self.notify("Copied selection", timeout=1.8)
        else:
            self.notify(f"Copy failed: {result.detail}", severity="warning", timeout=3)

    def action_copy_selection(self) -> None:
        try:
            text = self.screen.get_selected_text()
        except Exception:
            text = None
        if not text:
            try:
                text = self.query_one("#prompt-input", PromptArea).selected_text
            except Exception:
                text = None
        if not text:
            self._system_note("Nothing selected — drag over text, or use /copy for the last answer")
            return
        result = clipboard.copy_text(text, app=self)
        if result.ok:
            self.notify(f"Copied selection · {result.backend}", timeout=2)
        else:
            self.notify(f"Copy failed: {result.detail}", severity="warning", timeout=3)

    def action_copy_last(self) -> None:
        text = self.last_answer
        if not text:
            assistants = [m for m in self.messages if m.get("role") == "assistant" and m.get("content")]
            text = assistants[-1]["content"] if assistants else ""
        if not text:
            self._system_note("Nothing to copy yet")
            return
        result = clipboard.copy_text(text, app=self)
        if result.ok:
            self.notify(f"Copied last answer · {result.backend}", timeout=2)
        else:
            self.notify(f"Copy failed: {result.detail}", severity="warning", timeout=3)

    def action_copy_all(self) -> None:
        """Copy the entire visible turn: user prompts, tool findings and answers."""
        parts: List[str] = []
        conv = None
        try:
            conv = self.query_one("#conversation", VerticalScroll)
        except Exception:
            conv = None
        if conv is not None:
            for child in conv.children:
                if isinstance(child, ChatMessage):
                    role = "You" if child.role == "user" else "Assistant"
                    content = str(getattr(child, "raw_content", "") or "")
                    if content:
                        parts.append(f"[{role}]\n{content}")
                elif isinstance(child, FindingCard):
                    title = str(getattr(child, "display_name", "tool") or "tool")
                    detail = str(getattr(child, "full_result", "") or "")
                    parts.append(f"[Tool finding: {title}]\n{detail}")
                elif isinstance(child, ThoughtBlock):
                    thought = str(getattr(child, "thought", "") or "")
                    if thought:
                        parts.append(f"[Thinking]\n{thought}")
        if not parts:
            # Fall back to structured messages if the widgets are not mounted.
            for message in self.messages:
                role = "You" if message.get("role") == "user" else "Assistant"
                content = str(message.get("content") or "")
                if content:
                    parts.append(f"[{role}]\n{content}")
        if not parts:
            self._system_note("Nothing to copy yet")
            return
        text = "\n\n".join(parts)
        result = clipboard.copy_text(text, app=self)
        if result.ok:
            self.notify(f"Copied full conversation · {result.backend}", timeout=2)
        else:
            self.notify(f"Copy failed: {result.detail}", severity="warning", timeout=3)

    def action_toggle_mcp_display(self) -> None:
        self.mcp_display = "verbose" if self.mcp_display == "background" else "background"
        appconfig.update(mcp_display=self.mcp_display)
        self.notify(f"MCP display: {self.mcp_display}", timeout=2.5)
        self._system_note(
            f"MCP display = {self.mcp_display}. "
            + ("Use /bg to inspect background calls. " if self.mcp_display == "background" else "Tool cards are shown live. ")
            + "Toggle with Ctrl+O or /mcp-verbose."
        )

    def _show_bg_summary(self, index: Optional[int] = None) -> None:
        if not self._bg_calls:
            self._system_note("No background MCP/search calls in the last turn.")
            return
        if index is not None:
            if 1 <= index <= len(self._bg_calls):
                call = self._bg_calls[index - 1]
                self._system_note(
                    f"[{index}] {call.get('name', 'tool')} · "
                    f"{'ok' if call.get('ok') else 'failed'} · "
                    f"{float(call.get('duration') or 0):.2f}s\n"
                    + clean_result(call.get("result") or "")[:8000]
                )
            else:
                self._system_note(f"No background call #{index} (have {len(self._bg_calls)}).")
            return
        lines = [f"Background calls ({len(self._bg_calls)}):"]
        for i, call in enumerate(self._bg_calls, 1):
            mark = "ok" if call.get("ok") else "failed"
            lines.append(
                f"  {i}. {call.get('name', 'tool')} · {mark} · {float(call.get('duration') or 0):.2f}s"
            )
            hint = str(call.get("hint") or "")
            if hint:
                lines.append(f"     {hint[:160]}")
            preview = str(call.get("preview") or "")
            if preview:
                lines.append("     " + preview[:360].replace("\n", "\n     "))
        lines.append("Use /bg <n> for the full result of one call.")
        self._system_note("\n".join(lines))

    def action_paste_clipboard(self) -> None:
        text = clipboard.paste_text(app=self) or ""
        if not text:
            # QTerminal also maintains PRIMARY; accept that as a fallback.
            try:
                import subprocess

                out = subprocess.run(
                    ["xclip", "-selection", "primary", "-o"],
                    capture_output=True,
                    timeout=2,
                )
                if out.returncode == 0:
                    text = out.stdout.decode("utf-8", "replace")
            except Exception:
                pass
        if not text:
            self._system_note(
                "Clipboard is empty or unavailable. Ctrl+Alt+V / Alt+V paste here; "
                "install xclip or xsel if needed."
            )
            return
        area = self.query_one("#prompt-input", PromptArea)
        if getattr(area, "_collapsed", False):
            area.load_text("")
            area._collapsed = False  # type: ignore[attr-defined]
            area._full_text = ""  # type: ignore[attr-defined]
            area._paste_placeholder = ""  # type: ignore[attr-defined]
        area.insert(text)
        area.focus()
        self._system_note(f"Pasted {len(text)} chars ({len(text.splitlines())} lines)")

    # ---------------------------------------------------------- permissions
    async def _ask_callback(self, question: str, options: List[str]) -> Optional[str]:
        """Answer Cline-style ask_question through a real modal."""
        try:
            return await self.push_screen_wait(AskQuestionScreen(question, options))
        except Exception:
            return None

    async def _permission_callback(self, tool: str, args: Dict[str, Any], risk: str) -> bool:
        """Ask the user through a real modal; remember session approvals."""
        sig = permissions.signature(tool, args)
        if sig in self._session_approvals:
            return True
        if self.mode == "Ask" and risk in ("write", "shell", "danger"):
            self._system_note(f"Ask mode blocked {tool} ({risk}). Switch to Build with Tab.")
            return False
        try:
            choice = await self.push_screen_wait(
                PermissionScreen(permissions.describe(tool, args, risk), risk)
            )
        except Exception:
            return False
        if choice == "session":
            self._session_approvals.add(sig)
            return True
        return choice == "once"

    # ---------------------------------------------------------------- liveness
    _LIVENESS_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _liveness_paint(self) -> None:
        try:
            bar = self.query_one("#liveness-bar", Static)
        except Exception:
            return
        elapsed = max(0.0, time.perf_counter() - self._liveness_started) if self._liveness_started else 0.0
        frame = self._LIVENESS_FRAMES[self._liveness_frame % len(self._LIVENESS_FRAMES)]
        label = (self._liveness_detail or self._liveness_phase or "Working").strip()
        if len(label) > 70:
            label = label[:67] + "..."
        extra = f" · {self._active_subagents} subagent{'s' if self._active_subagents != 1 else ''} active" if self._active_subagents else ""
        bar.update(Text(f"{frame} {label}{extra} · {elapsed:.1f}s", style="bold #e5a93c"))

    def _liveness_tick(self) -> None:
        if not self.busy:
            return
        self._liveness_frame = (self._liveness_frame + 1) % len(self._LIVENESS_FRAMES)
        self._liveness_paint()

    def _liveness_start(self, phase: str = "Working", detail: str = "") -> None:
        self._liveness_phase = phase
        self._liveness_detail = detail
        self._liveness_started = time.perf_counter()
        self._liveness_frame = 0
        try:
            bar = self.query_one("#liveness-bar", Static)
            bar.add_class("active")
        except Exception:
            pass
        if self._liveness_timer is None:
            try:
                self._liveness_timer = self.set_interval(0.1, self._liveness_tick)
            except Exception:
                self._liveness_timer = None
        self._liveness_paint()

    def _liveness_set(self, phase: str, detail: str = "") -> None:
        if not self.busy:
            return
        self._liveness_phase = phase
        self._liveness_detail = detail
        self._liveness_paint()

    def _liveness_stop(self, ok: bool = True) -> None:
        try:
            bar = self.query_one("#liveness-bar", Static)
            bar.remove_class("active")
            bar.update("")
        except Exception:
            pass

    # ------------------------------------------------------------------ agent
    @work(exclusive=False)
    async def run_prompt(self, text: str, echo_user: bool = True) -> None:
        self.busy = True
        self._active_subagents = 0
        self.refresh_status()
        self._liveness_start("Connecting")
        conv = self.query_one("#conversation", VerticalScroll)
        if echo_user:
            await conv.mount(ChatMessage("user", text))
            self.messages.append({"role": "user", "content": text})
            if not self.conv_id:
                self.conv_id = db.create_conversation(text[:44], self.model)
            db.add_message(self.conv_id, "user", text)
            conv.scroll_end(animate=False)

        timeline = PhaseTimeline()
        await conv.mount(timeline)
        spinner = ActivitySpinner()
        assistant = ChatMessage("assistant", "")
        await conv.mount(assistant)
        await conv.mount(spinner, before=assistant)

        thought: Optional[ThoughtBlock] = None
        thought_parts: List[str] = []
        content_parts: List[str] = []
        cards: List[ToolCard] = []
        sub_cards: Dict[str, SubagentCard] = {}
        self._delegates = []
        bg: Optional[BackgroundIndicator] = None
        bg_names: Dict[str, bool] = {}
        findings_header: Optional[SectionLabel] = None
        self._bg_calls = []
        started = time.perf_counter()
        failed = False

        prompt = text
        if self.mode == "Ask":
            prompt = "[MODE: ASK/READ-ONLY] Do not edit files or run destructive commands. Explain, analyze, and answer.\n\n" + text

        try:
            async for event in run_agent(
                prompt,
                history=self.messages[:-1],
                model=self.model,
                mcp_manager=self.mcp,
                skills_instructions=self.skills.active_instructions(),
                disabled_tools=self.disabled_tools,
                disabled_servers=self.disabled_servers,
                enable_subagents=self.subagents_enabled,
                enable_self_learning=self.self_learning_enabled,
                permission_mode="strict" if self.mode == "Ask" else self.permission_mode,
                permission_callback=self._permission_callback,
                hooks_enabled=self.hooks_enabled,
                allow_react_fallback=self.allow_react_fallback,
                skills=self.skills,
                ask_callback=self._ask_callback,
            ):
                event_type = event.get("type")
                if event_type == "phase":
                    phase = event.get("phase", "")
                    state = event.get("state", "")
                    if state == "start":
                        timeline.start(phase, event.get("label"))
                        _phase_label = {"connecting": "Connecting", "thinking": "Thinking", "writing": "Writing response", "tool": "Running tool"}.get(phase, phase.title())
                        _detail = str(event.get("label") or "") if phase == "tool" else ""
                        spinner.set_activity(_phase_label, _detail)
                        self._liveness_set(_phase_label, _detail)
                    elif state == "end":
                        timeline.end(phase, event.get("duration"))
                        if phase == "tool":
                            spinner.set_activity("Running tool")
                            self._liveness_set("Running tool")
                    elif state == "error":
                        timeline.end(phase, event.get("duration"), ok=False, label=event.get("label"))
                elif event_type == "thought_delta":
                    thought_parts.append(event.get("text", ""))
                    if thought is None:
                        thought = ThoughtBlock("", 0.0)
                        await conv.mount(thought, before=assistant)
                    joined = "".join(thought_parts)
                    thought.set_thought(joined, time.perf_counter() - started)
                    # Keep the live indicator reactive while reasoning streams.
                    spinner.set_activity("Thinking", f"reasoning · {len(joined)} chars")
                    self._liveness_set("Thinking", f"reasoning · {len(joined)} chars")
                elif event_type == "thought":
                    if thought is None:
                        thought = ThoughtBlock(event.get("text", ""), float(event.get("duration") or 0.0))
                        await conv.mount(thought, before=assistant)
                    else:
                        thought.set_thought(event.get("text", ""), float(event.get("duration") or 0.0))
                elif event_type in ("permission_denied", "hook_blocked"):
                    self._system_note(
                        ("Permission denied" if event_type == "permission_denied" else "Hook blocked")
                        + f": {event.get('tool') or event.get('event') or event.get('reason') or ''}"
                    )
                elif event_type == "subagent_start":
                    role = str(event.get("role") or "explore")
                    task = str(event.get("task") or "")
                    key = f"{role}|{task}|{len(sub_cards)}"
                    sub_card = SubagentCard(role, task)
                    sub_cards[key] = sub_card
                    self._active_subagents = len(sub_cards)
                    self._liveness_set("Subagents", f"{role}: {task[:50]}")
                    await conv.mount(sub_card, before=assistant)
                elif event_type == "subagent_end":
                    role = str(event.get("role") or "explore")
                    task = str(event.get("task") or "")
                    key = f"{role}|{task}"
                    # best-effort match
                    sub_card = sub_cards.pop(key, None)
                    if sub_card is None and sub_cards:
                        # try prefix match
                        for k in list(sub_cards.keys()):
                            if k.startswith(f"{role}|{task}|"):
                                sub_card = sub_cards.pop(k)
                                break
                    if sub_card is not None:
                        sub_card.finish(
                            bool(event.get("ok")), str(event.get("preview") or event.get("error") or ""),
                            float(event.get("duration") or 0.0),
                        )
                    self._active_subagents = len(sub_cards)
                    self._delegates.append({
                        "role": role, "task": task,
                        "preview": str(event.get("preview") or event.get("error") or ""),
                        "ok": bool(event.get("ok")),
                    })
                elif event_type == "tool_start":
                    name = event.get("name", "tool")
                    display = event.get("display") or name
                    hidden = bool(event.get("background")) and self.mcp_display == "background"
                    bg_names[name] = hidden
                    if hidden:
                        if bg is None:
                            bg = BackgroundIndicator()
                            await conv.mount(bg, before=assistant)
                        bg.start(display)
                    else:
                        card = ToolCard(name, display, event.get("args") or {})
                        cards.append(card)
                        await conv.mount(card, before=assistant)
                    timeline.start("tool", display)
                    spinner.set_activity("Running tool", display)
                    self._liveness_set("Running tool", display)
                elif event_type == "tool_output":
                    if bg_names.get(event.get("name"), False):
                        continue
                    for card in reversed(cards):
                        if card.tool_name == event.get("name") and not card.done:
                            card.append_output(event.get("stream", "stdout"), event.get("text", ""))
                            break
                elif event_type == "tool_end":
                    name = event.get("name", "tool")
                    duration = float(event.get("duration") or 0.0)
                    ok = bool(event.get("ok", True))
                    if bg_names.get(name, False):
                        display = event.get("display") or name
                        raw_result = str(event.get("result") or "")
                        if bg is not None:
                            bg.finish(display, duration, ok)
                        hint = ""
                        try:
                            data = json.loads(raw_result)
                            if isinstance(data, dict):
                                hint = str(data.get("hint") or data.get("error") or data.get("message") or "")
                        except Exception:
                            pass
                        if findings_header is None:
                            findings_header = SectionLabel("TOOL FINDINGS", "#6abf69")
                            await conv.mount(findings_header, before=assistant)
                        finding = FindingCard(name, display, raw_result, duration, ok, hint)
                        await conv.mount(finding, before=assistant)
                        self._bg_calls.append({
                            "name": display,
                            "duration": duration,
                            "ok": ok,
                            "hint": hint,
                            "result": raw_result,
                            "preview": result_preview(clean_result(raw_result)),
                        })
                    else:
                        for card in reversed(cards):
                            if card.tool_name == name and not card.done:
                                card.set_result(event.get("result", ""), duration, ok)
                                break
                        if ok and name in ("write", "edit", "patch", "rename"):
                            try:
                                data = json.loads(event.get("result") or "")
                            except Exception:
                                data = {}
                            diff = str(data.get("diff") or "") if isinstance(data, dict) else ""
                            if diff:
                                self._last_diff = diff
                                self._last_undo_id = str(data.get("undo_id") or "")
                                await conv.mount(DiffCard(diff, self._last_undo_id), before=assistant)
                    timeline.end("tool", event.get("duration"), ok=ok, label=event.get("display"))
                elif event_type == "delta":
                    content_parts.append(event.get("text", ""))
                    assistant.set_content("".join(content_parts))
                elif event_type == "notice":
                    self._system_note(str(event.get("message")))
                elif event_type == "error":
                    failed = True
                    self._last_failed = text
                    card = event.get("card")
                    elapsed = float(event.get("elapsed") or 0.0)
                    if card:
                        await conv.mount(ErrorCard(card, elapsed), before=assistant)
                        retry_after = card.get("retry_after")
                        if retry_after:
                            self._cooldown_until = time.time() + float(retry_after)
                            self._cooldown_reason = str(card.get("reason") or "402")
                            appconfig.update(
                                cooldown_until=self._cooldown_until,
                                cooldown_reason=self._cooldown_reason,
                            )
                    timeline.finish_all(ok=False)
                    # The ErrorCard already renders the message, reset time and
                    # recovery options. Repeating them here produced the noisy
                    # double "HTTP 429 ... / Recover with ..." lines.
                    if not card:
                        self._system_note(f"Error: {event.get('message')}")
                        self._system_note("Run /retry to try again.")
                conv.scroll_end(animate=False)
        except asyncio.CancelledError:
            timeline.finish_all(ok=False)
            self._system_note("Interrupted")
        except Exception as exc:  # noqa: BLE001
            timeline.finish_all(ok=False)
            self._system_note(f"Error: {exc}")
        finally:
            spinner.finish(not failed)
            self._liveness_stop(not failed)
            if findings_header is not None:
                await conv.mount(SectionLabel("RESPONSE", "#8ab4f8"), before=assistant)
            self.busy = False
            timeline.finish_all(ok=not failed)
            self.refresh_status()

        answer = "".join(content_parts).strip()
        if answer:
            self.last_answer = answer
            self.messages.append({"role": "assistant", "content": answer})
            db.add_message(self.conv_id, "assistant", answer)
        if not failed and self.checkpoint_enabled and self.messages:
            entry = session_util.create(
                os.getcwd(), self.messages, model=self.model,
                last_prompt=self.last_prompt,
            )
            if entry:
                self._session_id = entry["id"]
        if not failed and self._cooldown_until:
            self._cooldown_until = 0.0
            self._cooldown_reason = ""
            appconfig.update(cooldown_until=0, cooldown_reason="")
        self.update_context()
        # Prompts typed while this turn was running go out now, in order.
        self.call_later(self._drain_pending)


def main() -> None:
    app = DarkGPTApp()
    app.run()


if __name__ == "__main__":
    main()
