# DarkGPT CLI / TUI

DarkGPT is a terminal-first AI coding and security-research agent with a Textual TUI, native tool calling, MCP, skills, browser automation, self-learning, and target-centric recon/reporting.

> Built for authorized security research, CTF, pentest labs, and engineering workflows. Do not use against systems without explicit permission.

## Highlights

- **TUI-first workflow** — fast prompt loop, streaming phases, tool cards, diff cards, sessions, checkpoints, copy/paste, command palette.
- **Native tools + safe fallback** — structured tool calls first; ReAct fallback is controlled (`/react on|off`), not a silent downgrade.
- **Self-learning** — durable lessons/patterns captured from failures and repeated signals; injected into future turns (`/lessons`, `/patterns`, `/learn`).
- **Skill system** — Markdown skills with frontmatter, nested `<skill>/SKILL.md` bundles, active/inactive state. HackSkills router is included and active by default.
- **MCP integration** — stdio/HTTP MCP servers, tool discovery, per-server enable/disable, background/verbose display.
- **Browser automation** — built-in `browser` tool (Playwright with Selenium fallback), session cookies, JS rendering, forms/clicks/extract/screenshot.
- **Coding brain** — read/write/edit/patch, LSP, git, verify/lint/test, review, subagents, hooks, permissions, scripts, checkpoints, memory.

## Core Security / Recon Reasoning

- **Dynamic target state** — per-target knowledge graph: `hosts → services → endpoints → findings`, persisted under `~/.config/darkgpt/targets/<slug>/`.
- **Evidence ledger** — append-only `evidence.jsonl` with SHA-256, timestamps, redacted args/result previews; tool results auto-captured when a target is active.
- **Adaptive hypothesis loop** — failures/blocks are treated as signal; agent updates state, mutates approach, pivots route, and records lessons.
- **Multi-agent orchestration** — `delegate`/`spawn_agent` roles: `explore`, `review`, `test`, `recon`, `vuln_analysis`, `exploit_validation`, `reporting`; `recon_orchestrate` runs parallel recon and merges into graph.
- **Proof-first reporting** — findings track `attempted/proved/false_positive`, reproduction, impact, remediation, evidence refs. `report_generate` outputs Markdown/HTML.

## Install

```bash
git clone https://github.com/sharma5pac/DARK-GPT-CLI-.git
cd DARK-GPT-CLI-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env with your API keys
```

## Run

```bash
./darkgpt
# or
python darkgpt_cli.py
```

## Configuration

Set keys in `.env` (never commit real keys):

```env
OPENROUTER_API_KEY=your-openrouter-key-here
NVIDIA_API_KEY=your-nvidia-key-here
```

User config lives in `~/.config/darkgpt/config.json`. MCP config in `~/.config/darkgpt/mcp.json`.

## Key Commands

| Command | Purpose |
|---|---|
| `/model` `/provider` | Select model/provider |
| `/target <name>` | Set active target for graph/evidence/report |
| `/targets` `/kg` `/evidence` | Inspect target knowledge graph and evidence |
| `/recon [focus]` | Parallel recon orchestration into active target graph |
| `/report md\|html [out]` | Generate report from graph + evidence |
| `/lessons` `/patterns` `/learn` | Inspect/toggle self-learning |
| `/skills` `/mcp` `/tools` | Manage skills, MCP servers, tools |
| `/diff` `/undo` `/verify` `/review` | Edit safety and verification |
| `/checkpoint` `/sessions` `/resume` | Session checkpoints |
| `/permissions` `/hooks` | Runtime controls |

## Security Notes

- Use only on systems you own or have explicit written authorization to test.
- Evidence and reports may contain sensitive data; handle them as confidential.
- The agent records evidence locally; review before sharing.
- `.env`, local caches, and private runtime state should not be committed.

## License

MIT — see [LICENSE](LICENSE).
