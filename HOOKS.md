# Hooks, permissions and project rules

## Hooks

Create `~/.config/darkgpt/hooks.json` (global) or `<repo>/.darkgpt/hooks.json` (repo-local):

```json
{
  "pre_edit": [
    "python tools/check_style.py --guard",
    {"command": "test -f .allow-edits", "timeout": 3}
  ],
  "post_edit": ["git diff --stat"],
  "pre_shell": ["python tools/block_dangerous.py"],
  "post_shell": [],
  "pre_tool": [],
  "post_tool": [],
  "user_prompt_submit": [],
  "stop": []
}
```

Rules:

- Every entry is a shell command string or `{"command": "...", "timeout": N}`.
- Timeouts are 1-30s (default 5s). Output is captured and shown to the model.
- A non-zero exit from a `pre_*` hook **blocks** the tool with the output as the reason.
- `post_*` hooks never block; their output is attached to the tool result as `hook_output`.
- Available environment variables: `DARKGPT_EVENT`, `DARKGPT_CWD`, `DARKGPT_TOOL`,
  `DARKGPT_PATH`, `DARKGPT_COMMAND`.
- Use `/hooks` in the TUI to see which hooks are active.

## Permissions

`/permissions full|ask|strict` (persisted in `~/.config/darkgpt/config.json`):

| Mode | Behaviour |
|---|---|
| `full` | No prompts (default - previous behaviour). |
| `ask` | Confirm file edits, shell commands and MCP calls. |
| `strict` | Confirm every non-read-only tool. |

When prompted, choose **Allow once**, **Allow session** (same tool + path for the
rest of the session), or **Deny**. Two consecutive denials halt the turn so the
agent cannot loop asking. The `Ask` mode toggle in the footer forces strict and
auto-denies anything that would write or execute a shell command.

## Self-written scripts

The `script` tool writes a Python or bash program to
`~/.cache/darkgpt/scripts/` and runs it with a bounded timeout (30s default,
120s max). It is classified as `shell` risk - or `danger` when the source
matches the dangerous-command patterns - so `ask`/`strict` modes prompt and
`pre_shell` / `post_shell` hooks fire exactly as they do for `shell`.

Scripts are never deleted. `/scripts` lists them and `/scripts show <path>`
prints the source.

## Native tools and ReAct

`/react on|off` controls the text-based tool transport. Native tool calling is
always attempted first and retried twice (relaxed provider routing, then a
minimal tool set) before ReAct is considered. With `/react off`, a tool-call
rejection is a clear fatal error with the provider's message instead of a
silent downgrade.

## Project rules (CLAUDE.md-style)

DarkGPT loads, in priority order:

1. `<repo>/.darkgpt/rules.md`
2. `<repo>/DARKGPT.md`
3. `<repo>/CLAUDE.md`
4. `<repo>/AGENTS.md`
5. `~/.config/darkgpt/rules.md`

All found files are merged into the system prompt on every turn. Use
`/rules add "Always use pytest fixtures from tests/conftest.py"` to append a
repo rule, and `/rules` to display everything currently loaded.

## Sessions and compaction

- `/checkpoint [label]` saves the conversation, git status and a compact summary
  under `~/.config/darkgpt/sessions/` (append-only).
- Successful turns save a checkpoint automatically (toggle via config
  `checkpoint_enabled`).
- `/sessions` lists them; `/resume <id>` or `/resume latest` restores the text
  history into the conversation. Tools are never re-executed and the model
  setting is left untouched.
- `/compact` replaces older turns with a structured summary in-place.
