"""System clipboard helpers with layered fallbacks.

Terminal apps cannot rely on OSC 52 alone: xfce4-terminal, QTerminal, tmux and
SSH sessions vary in what they forward. This module writes through every backend
it can find and reports which one worked, so the TUI can show an honest status
instead of failing silently.

Order: OSC 52 (Textual) -> wl-copy (Wayland) -> xclip (X11 CLIPBOARD + PRIMARY)
-> xsel -> pyperclip -> tkinter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CopyResult:
    ok: bool
    backend: str = ""
    errors: List[str] = field(default_factory=list)

    @property
    def detail(self) -> str:
        if self.ok:
            return self.backend or "clipboard"
        return "; ".join(self.errors[-3:]) or "no clipboard backend available"


def _run(cmd: List[str], data: Optional[bytes] = None, timeout: float = 2.0) -> Optional[bytes]:
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout
        return None
    except Exception:
        return None


def _write(cmd: List[str], data: bytes, timeout: float = 2.0) -> bool:
    """Write to a clipboard owner without capturing its pipes.

    xclip/xsel/wl-copy fork into the background and keep stdout/stderr open;
    capturing those pipes makes ``subprocess.run`` wait until the timeout even
    though the copy succeeded. Redirecting to DEVNULL avoids that hang.
    """
    try:
        proc = subprocess.run(
            cmd,
            input=data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return proc.returncode == 0
    except Exception:
        return False


def available() -> Dict[str, Any]:
    """What the current session can use, for /clipboard diagnostics."""
    return {
        "display": os.environ.get("DISPLAY") or "",
        "wayland": os.environ.get("WAYLAND_DISPLAY") or "",
        "session": os.environ.get("XDG_SESSION_TYPE") or "",
        "xclip": shutil.which("xclip") or "",
        "xsel": shutil.which("xsel") or "",
        "wl-copy": shutil.which("wl-copy") or "",
        "wl-paste": shutil.which("wl-paste") or "",
        "pyperclip": bool(__import__("importlib").util.find_spec("pyperclip")),
        "tkinter": bool(__import__("importlib").util.find_spec("tkinter")),
    }


def copy_text(text: str, app: Any = None, also_primary: bool = True) -> CopyResult:
    """Copy text through OSC 52 and every available system backend.

    Never raises: missing binaries and dead displays become entries in
    ``errors`` and the caller decides how to surface them.
    """
    if not text:
        return CopyResult(False, "", ["empty selection"])
    data = text.encode("utf-8")
    used: List[str] = []
    errors: List[str] = []

    # 1. Textual / terminal OSC 52 (works over SSH when the terminal allows it)
    if app is not None:
        try:
            app.copy_to_clipboard(text)
            used.append("osc52")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"osc52: {exc}")

    # 2. Wayland
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        if _write(["wl-copy"], data):
            used.append("wl-copy")
        else:
            errors.append("wl-copy failed")

    # 3. X11 clipboard + PRIMARY (middle-click paste)
    if shutil.which("xclip"):
        selections = ["clipboard"] + (["primary"] if also_primary else [])
        for selection in selections:
            if _write(["xclip", "-selection", selection], data):
                used.append(f"xclip:{selection}")
            else:
                errors.append(f"xclip:{selection} failed")

    if shutil.which("xsel"):
        if _write(["xsel", "--clipboard", "--input"], data):
            used.append("xsel:clipboard")
        else:
            errors.append("xsel failed")
        if also_primary and _write(["xsel", "--primary", "--input"], data):
            used.append("xsel:primary")

    # 4. Pure-Python fallbacks
    if not used:
        try:
            import pyperclip

            pyperclip.copy(text)
            used.append("pyperclip")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pyperclip: {exc}")

    if not used:
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
            used.append("tkinter")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"tkinter: {exc}")

    return CopyResult(bool(used), "+".join(used), errors)


def paste_text(app: Any = None) -> str:
    """Read the clipboard, trying the same backends in reverse order."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        out = _run(["wl-paste", "--no-newline"])
        if out is not None:
            return out.decode("utf-8", "replace")
    if shutil.which("xclip"):
        out = _run(["xclip", "-selection", "clipboard", "-o"])
        if out is not None:
            return out.decode("utf-8", "replace")
    if shutil.which("xsel"):
        out = _run(["xsel", "--clipboard", "--output"])
        if out is not None:
            return out.decode("utf-8", "replace")
    try:
        import pyperclip

        value = pyperclip.paste()
        if value:
            return value
    except Exception:
        pass
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        value = root.clipboard_get()
        root.destroy()
        return value
    except Exception:
        return ""


def diagnostics() -> str:
    info = available()
    lines = ["Clipboard backends:"]
    for key, value in info.items():
        if isinstance(value, bool):
            value = "yes" if value else "no"
        lines.append(f"  {key:10} {value or '-'}")
    lines.append("")
    lines.append("QTerminal / X11 notes:")
    lines.append("  QTerminal copy = Ctrl+Shift+C, paste = Ctrl+Shift+V (Preferences > Shortcuts).")
    lines.append("  If mouse tracking eats the drag, hold Shift while selecting.")
    lines.append("  xclip writes both CLIPBOARD and PRIMARY; parcellite/clipman sync them.")
    lines.append("  Wayland: install wl-clipboard (wl-copy/wl-paste). SSH/tmux: enable OSC 52.")
    return "\n".join(lines)
