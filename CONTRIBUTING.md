# Contributing to DarkGPT

Thanks for helping make DarkGPT better.

## Ground rules

- **No malicious use.** Do not contribute features intended for unauthorized access or harm.
- **Preserve evidence.** Avoid changes that silently delete user data, logs, or evidence artifacts.
- **Keep changes verifiable.** Include tests or a clear manual verification path.

## Dev setup

```bash
git clone https://github.com/sharma5pac/DARK-GPT-CLI-.git
cd DARK-GPT-CLI-
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
./darkgpt
```

## Before opening a PR

```bash
python -m compileall agent utils tui tests
python tests/test_darkgpt_all.py
```

## Good PR targets

- MCP/tool integrations
- recon/graph/report improvements
- skill packs and prompts
- TUI UX polish
- bug fixes with regression tests

## Security issues

Do not open public issues for sensitive vulnerabilities. Report privately to the maintainer first.