#!/usr/bin/env python3
"""DarkGPT terminal agent - OpenCode-style interface."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner

from utils.agent import DEFAULT_AGENT_MODEL, agent_stream  # noqa: E402
from utils import history as db  # noqa: E402

console = Console()
PROMPT_STYLE = Style.from_dict({
    "prompt": "ansigreen bold",
    "dim": "ansibrightblack",
})
HISTORY_FILE = str(Path.home() / ".darkgpt_history")


def ensure_web() -> None:
    """Start the DarkGPT web service if it is not already up."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8501/api/health", timeout=1.5) as r:
            if r.status == 200:
                return
    except Exception:
        pass
    # try systemd first
    try:
        r = subprocess.run(["systemctl", "--user", "start", "darkgpt.service"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for _ in range(20):
                time.sleep(0.4)
                try:
                    import urllib.request
                    with urllib.request.urlopen("http://127.0.0.1:8501/api/health", timeout=1) as resp:
                        if resp.status == 200:
                            return
                except Exception:
                    pass
    except Exception:
        pass
    # fallback direct
    try:
        venv = ROOT / ".venv" / "bin" / "uvicorn"
        subprocess.Popen([str(venv), "server.api:app", "--host", "127.0.0.1", "--port", "8501", "--log-level", "warning"],
                         cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(30):
            time.sleep(0.4)
            try:
                import urllib.request
                with urllib.request.urlopen("http://127.0.0.1:8501/api/health", timeout=1) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
    except Exception:
        pass


def banner(model: str, web_ok: bool) -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold white]DarkGPT[/bold white] [dim]agent[/dim]\n"
        f"[dim]cwd[/dim]   {os.getcwd()}\n"
        f"[dim]model[/dim] {model}\n"
        f"[dim]web[/dim]   {'http://127.0.0.1:8501  [green]ready[/green]' if web_ok else '[yellow]not running[/yellow]'}\n\n"
        f"[dim]commands[/dim]  /new  /model <id>  /web  /cwd  /exit",
        border_style="bright_black",
        padding=(1, 2),
    ))
    console.print()


def render_tool_start(name: str, args: dict) -> None:
    a = json.dumps(args, ensure_ascii=False)
    if len(a) > 140:
        a = a[:140] + "..."
    console.print(f"[yellow][*] {name}[/yellow] [dim]{a}[/dim]")


def render_tool_end(name: str, result: str) -> None:
    out = result.strip()
    if len(out) > 700:
        out = out[:700] + "\n..."
    console.print(Panel(out, title=f"[dim]{name}[/dim]", border_style="bright_black", padding=(0, 1)))


def run_once(prompt: str, model: str, conversation_id: int | None) -> int | None:
    history = db.get_messages(conversation_id) if conversation_id else []
    if not conversation_id:
        conversation_id = db.create_conversation(prompt[:44], model)
    db.add_message(conversation_id, "user", prompt)

    text_parts: list[str] = []
    status = console.status("[dim]thinking...[/dim]", spinner="dots")
    status.start()
    live = None
    try:
        for event in agent_stream(prompt, history=history, model=model):
            t = event.get("type")
            if t == "meta":
                status.update(f"[dim]{event.get('model', model)}[/dim]")
            elif t == "tool_start":
                status.stop()
                if live:
                    live.stop(); live = None
                render_tool_start(event.get("name", "?"), event.get("args") or {})
                status.start()
                status.update(f"[dim]running {event.get('name','tool')}...[/dim]")
            elif t == "tool_end":
                status.stop()
                render_tool_end(event.get("name", "?"), event.get("result") or "")
                status.start()
                status.update("[dim]thinking...[/dim]")
            elif t == "delta":
                status.stop()
                chunk = event.get("text") or ""
                text_parts.append(chunk)
                # simple live markdown render
                if live is None:
                    live = Live(Markdown("".join(text_parts)), console=console, refresh_per_second=8, transient=False)
                    live.start()
                else:
                    live.update(Markdown("".join(text_parts)))
            elif t == "error":
                if live: live.stop(); live = None
                status.stop()
                console.print(f"[red]error:[/red] {event.get('message')}")
                status.start()
            elif t == "done":
                pass
    except KeyboardInterrupt:
        if live: live.stop()
        status.stop()
        console.print("\n[yellow][interrupted][/yellow]")
    finally:
        try:
            if live: live.stop()
        except Exception:
            pass
        try:
            status.stop()
        except Exception:
            pass

    answer = "".join(text_parts).strip()
    if answer:
        db.add_message(conversation_id, "assistant", answer)
    console.print()
    return conversation_id


def main() -> int:
    ap = argparse.ArgumentParser(description="DarkGPT terminal agent")
    ap.add_argument("prompt", nargs="*", help="One-shot prompt. Omit for interactive mode.")
    ap.add_argument("-m", "--model", default=DEFAULT_AGENT_MODEL)
    ap.add_argument("-c", "--continue", dest="cont", action="store_true", help="Continue most recent conversation")
    ap.add_argument("--no-web", action="store_true", help="Do not auto-start web UI")
    args = ap.parse_args()

    db.init_db()

    web_ok = False
    if not args.no_web:
        with console.status("[dim]starting services...[/dim]", spinner="dots"):
            ensure_web()
        try:
            import urllib.request
            with urllib.request.urlopen("http://127.0.0.1:8501/api/health", timeout=1.5) as r:
                web_ok = r.status == 200
        except Exception:
            web_ok = False

    conv_id = None
    if args.cont:
        convs = db.list_conversations()
        conv_id = convs[0]["id"] if convs else None

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        banner(args.model, web_ok)
        run_once(prompt, args.model, conv_id)
        return 0

    banner(args.model, web_ok)
    session = PromptSession(history=FileHistory(HISTORY_FILE))
    while True:
        try:
            line = session.prompt(HTML("<prompt>? </prompt>"), style=PROMPT_STYLE).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return 0
        if line == "/new":
            conv_id = None
            console.print("[dim]new conversation[/dim]\n")
            continue
        if line.startswith("/model "):
            args.model = line.split(None, 1)[1].strip()
            console.print(f"[dim]model: {args.model}[/dim]\n")
            continue
        if line == "/cwd":
            console.print(os.getcwd() + "\n")
            continue
        if line == "/web":
            console.print("[dim]opening http://127.0.0.1:8501[/dim]")
            try:
                subprocess.Popen(["xdg-open", "http://127.0.0.1:8501"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            console.print()
            continue
        conv_id = run_once(line, args.model, conv_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
