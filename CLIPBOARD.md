# Clipboard & highlight-to-copy in DarkGPT

DarkGPT copies highlighted text automatically (like OpenCode). Drag over text in
the conversation and it is written to every clipboard backend available in the
session, with a brief `Copied selection` toast. No shortcut is required.

## Backends (tried in order)

1. **OSC 52** via `app.copy_to_clipboard()` - works through SSH/tmux when the
   terminal allows it. This is set by Textual and is always attempted first.
2. **wl-copy / wl-paste** - Wayland (`wl-clipboard` package).
3. **xclip -selection clipboard** and **xclip -selection primary** - X11.
   Writing both means Ctrl+V and middle-click paste work.
4. **xsel --clipboard** - X11 fallback.
5. **pyperclip** - optional Python fallback.
6. **tkinter** - last-resort Python fallback.

Missing binaries never raise: the failure is logged in the result and shown as a
warning toast. Run `/clipboard` in the TUI to see exactly what is available.

## QTerminal / X11 settings

- QTerminal **owns** Ctrl+Shift+C / Ctrl+Shift+V, so those keys never reach the
  TUI. DarkGPT also binds **Ctrl+Alt+C / Alt+C** (copy) and **Ctrl+Alt+V /
  Alt+V** (paste) - use these inside the TUI. Where the terminal passes them
  through, Ctrl+Shift+C / Ctrl+Shift+V still work.
- If mouse tracking intercepts a drag, hold **Shift** while selecting; QTerminal
  then lets the terminal handle the selection rather than the TUI.
- `PRIMARY` (middle-click) and `CLIPBOARD` (Ctrl+V) can be synchronised with
  `parcellite` or `clipman`:
  ```bash
  sudo apt install xclip parcellite
  parcellite &
  ```
- Verify from a shell:
  ```bash
  printf 'test' | xclip -selection clipboard
  xclip -selection clipboard -o
  ```
- DarkGPT writes both selections, so a clipboard manager is optional.

## Wayland settings

```bash
sudo apt install wl-clipboard   # wl-copy / wl-paste
```
`wl-copy` is used automatically when `WAYLAND_DISPLAY` is set. Note that
`xclip` will not paste across a native Wayland-only session.

## SSH / tmux

OSC 52 needs the terminal to pass the escape sequence through:

- tmux: `set -g allow-passthrough on`
- screen: `setattr +x paste` (or use tmux)
- QTerminal: enable OSC 52 clipboard writes in preferences if available.

If OSC 52 is blocked but the remote host has X11 forwarding, `xclip` still
works (`ssh -X`).

## Commands in the TUI

| Command / key | Effect |
|---|---|
| drag-select | auto-copy selection, show `Copied selection` |
| `Ctrl+Alt+C` / `Alt+C` | copy selection (terminal-safe) |
| `Ctrl+Alt+V` / `Alt+V` | paste clipboard into the prompt (terminal-safe) |
| `Ctrl+Shift+C` / `Ctrl+Shift+V` | same, only where the terminal passes them through |
| `Tab` | toggle Build / Ask from any focus (not just the prompt) |
| `Ctrl+Y` | copy the last assistant answer |
| `/copy` | copy the last assistant answer |
| `/clipboard` | show detected backends and these notes |
