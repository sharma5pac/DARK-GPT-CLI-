#!/usr/bin/env python3
"""Compatibility shim for the legacy DarkGPT CLI.

The old prompt_toolkit/rich interface was replaced by the Textual TUI in
``tui/app.py``. This entry point now forwards to that TUI so any old launch
path (``python darkgpt_cli.py``, shell history, scripts) gets the new
interface instead of the retired box UI.

The original implementation is preserved unchanged at
``darkgpt_cli_legacy.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from tui.app import main as tui_main

    tui_main()


if __name__ == "__main__":
    main()
