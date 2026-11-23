"""Disk-backed background jobs for DarkGPT.

Jobs run as detached subprocesses so they survive TUI restarts. Metadata and
logs live under ~/.cache/darkgpt/jobs/. The API stays compatible with the
in-process manager used by agent/core.py and the TUI.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import scripting

JOB_DIR = Path.home() / ".cache" / "darkgpt" / "jobs"
STATE_PATH = JOB_DIR / "jobs.json"
MAX_OUTPUT = 24000
DEFAULT_WAIT = 30
MAX_WAIT = 300


def _now() -> float:
    return time.time()


def _ensure() -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)


def _read_state() -> Dict[str, Any]:
    _ensure()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("jobs", {})
            return data
    except Exception:
        pass
    return {"version": 1, "jobs": {}}


def _write_state(state: Dict[str, Any]) -> None:
    _ensure()
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _tail_text(path: str, tail: int = 4000) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return ""
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > tail:
            return "...[tail %d chars]\n%s" % (tail, text[-tail:])
        return text
    except Exception:
        return ""


@dataclass
class Job:
    id: str
    kind: str
    label: str
    cwd: str
    command: List[str]
    script_path: str = ""
    status: str = "running"
    exit_code: Optional[int] = None
    started: float = field(default_factory=_now)
    ended: Optional[float] = None
    log_path: str = ""
    pid: Optional[int] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Job":
        return Job(
            id=str(d.get("id") or ""),
            kind=str(d.get("kind") or "shell"),
            label=str(d.get("label") or ""),
            cwd=str(d.get("cwd") or os.getcwd()),
            command=list(d.get("command") or []),
            script_path=str(d.get("script_path") or ""),
            status=str(d.get("status") or "running"),
            exit_code=d.get("exit_code"),
            started=float(d.get("started") or _now()),
            ended=d.get("ended"),
            log_path=str(d.get("log_path") or ""),
            pid=d.get("pid"),
        )

    def to_dict(self, include_output: bool = False, tail: int = 4000) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "exit_code": self.exit_code,
            "cwd": self.cwd,
            "script_path": self.script_path or None,
            "started": self.started,
            "ended": self.ended,
            "duration": max(0.0, (self.ended or _now()) - self.started),
            "log_path": self.log_path or None,
            "output_bytes": Path(self.log_path).stat().st_size if self.log_path and Path(self.log_path).exists() else 0,
            "pid": self.pid,
        }
        if include_output:
            d["output"] = _tail_text(self.log_path, tail=tail)
        return d


class JobManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        _ensure()

    def _new_id(self) -> str:
        return f"job_{uuid.uuid4().hex[:10]}"

    def _refresh_job(self, job: Job) -> Job:
        if job.status == "running" and job.pid and not _pid_alive(int(job.pid)):
            # process gone; infer status from log marker
            tail = _tail_text(job.log_path, tail=2000)
            status = "done"
            code = job.exit_code
            m = None
            for line in reversed(tail.splitlines()):
                if line.startswith("[exit "):
                    m = line
                    break
            if m:
                # [exit N status=... duration=...]
                try:
                    parts = m.strip("[]").split()
                    for p in parts:
                        if p.startswith("exit") and len(parts) > 1:
                            pass
                    code = int(parts[1])
                    for p in parts:
                        if p.startswith("status="):
                            status = p.split("=", 1)[1]
                except Exception:
                    pass
            job.status = status
            job.exit_code = code if code is not None else 0
            job.ended = job.ended or _now()
        return job

    def _load_jobs(self) -> Dict[str, Job]:
        state = _read_state()
        jobs = {}
        dirty = False
        for jid, jd in (state.get("jobs") or {}).items():
            if not isinstance(jd, dict):
                continue
            job = Job.from_dict(jd)
            old = (job.status, job.exit_code, job.ended)
            job = self._refresh_job(job)
            if (job.status, job.exit_code, job.ended) != old:
                state["jobs"][jid] = job.to_dict()
                dirty = True
            jobs[jid] = job
        if dirty:
            _write_state(state)
        return jobs

    def _save_job(self, job: Job) -> None:
        state = _read_state()
        state.setdefault("jobs", {})[job.id] = job.to_dict()
        # keep last 500 jobs
        items = sorted(state["jobs"].items(), key=lambda kv: float((kv[1] or {}).get("started") or 0), reverse=True)[:500]
        state["jobs"] = dict(items)
        _write_state(state)

    async def start_shell(self, command: str, cwd: str = "", label: str = "") -> Dict[str, Any]:
        cmd = str(command or "").strip()
        if not cmd:
            return {"ok": False, "error": "empty command"}
        jid = self._new_id()
        log_path = str(JOB_DIR / f"{jid}.log")
        job = Job(id=jid, kind="shell", label=label or cmd[:80], cwd=cwd or os.getcwd(), command=["/bin/bash", "-lc", cmd], log_path=log_path)
        return await self._start(job)

    async def start_script(self, language: str, code: str, args: Optional[List[Any]] = None, cwd: str = "", label: str = "") -> Dict[str, Any]:
        try:
            path, command = await asyncio.to_thread(scripting.build_argv, language, code, list(args or []))
        except scripting.ScriptError as exc:
            return {"ok": False, "error": str(exc)}
        jid = self._new_id()
        log_path = str(JOB_DIR / f"{jid}.log")
        job = Job(id=jid, kind="script", label=label or f"{language} script", cwd=cwd or os.getcwd(), command=command, script_path=str(path), log_path=log_path)
        return await self._start(job)

    async def _start(self, job: Job) -> Dict[str, Any]:
        try:
            _ensure()
            logf = open(job.log_path, "ab", buffering=0)
            logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {job.kind} {job.label}\n".encode("utf-8", "replace"))
            logf.flush()
            # Wrap command to capture exit code/status and persist across TUI restarts.
            wrapped = (
                "set -o pipefail; "
                + " ".join([json.dumps(c) for c in job.command]) if False else ""
            )
            # Build a shell wrapper for consistent exit marker
            cmd_str = " ".join(shlex_quote(c) for c in job.command)
            wrapper = f"{cmd_str}; ec=$?; echo; echo \"[exit $ec status=$( [ $ec -eq 0 ] && echo done || echo error ) duration=${{SECONDS}}s]\"; exit $ec"
            proc = await asyncio.create_subprocess_exec(
                "/bin/bash", "-lc", wrapper,
                cwd=job.cwd,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            job.pid = proc.pid
            logf.close()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"cannot start job: {exc}"}
        async with self._lock:
            self._save_job(job)
        return {"ok": True, "job": job.to_dict(), "message": f"Started {job.id} (pid {job.pid}). Use task_wait/task_log/task_list to monitor. Survives TUI restart."}

    def list(self, limit: int = 20) -> Dict[str, Any]:
        jobs = sorted(self._load_jobs().values(), key=lambda j: j.started, reverse=True)[: max(1, min(limit, 100))]
        return {"ok": True, "jobs": [j.to_dict() for j in jobs], "count": len(self._load_jobs())}

    def get(self, job_id: str) -> Optional[Job]:
        return self._load_jobs().get(str(job_id or "").strip())

    async def wait(self, job_ids: List[str], timeout: int = DEFAULT_WAIT) -> Dict[str, Any]:
        ids = [str(i).strip() for i in (job_ids or []) if str(i).strip()]
        missing = [i for i in ids if i not in self._load_jobs()]
        timeout = max(1, min(int(timeout or DEFAULT_WAIT), MAX_WAIT))
        deadline = _now() + timeout
        done = False
        while _now() < deadline:
            jobs = self._load_jobs()
            selected = [jobs[i] for i in ids if i in jobs]
            if selected and all(j.status != "running" for j in selected):
                done = True
                break
            await asyncio.sleep(0.5)
        jobs = self._load_jobs()
        selected = [jobs[i] for i in ids if i in jobs]
        return {"ok": True, "done": done, "jobs": [j.to_dict(include_output=True) for j in selected], "missing": [i for i in ids if i not in jobs]}

    async def kill(self, job_id: str) -> Dict[str, Any]:
        job = self.get(job_id)
        if not job:
            return {"ok": False, "error": f"unknown job: {job_id}"}
        if job.status == "running" and job.pid and _pid_alive(int(job.pid)):
            try:
                os.killpg(int(job.pid), signal.SIGTERM)
                await asyncio.sleep(0.5)
                if _pid_alive(int(job.pid)):
                    os.killpg(int(job.pid), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(int(job.pid), signal.SIGKILL)
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "error": f"kill failed: {exc}", "job": job.to_dict(include_output=True)}
            job.status = "killed"
            job.ended = _now()
            try:
                with open(job.log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(f"\n[killed by user]\n")
            except Exception:
                pass
            self._save_job(job)
        job = self._refresh_job(job)
        self._save_job(job)
        return {"ok": True, "job": job.to_dict(include_output=True)}

    def log(self, job_id: str, tail: int = 4000) -> Dict[str, Any]:
        job = self.get(job_id)
        if not job:
            return {"ok": False, "error": f"unknown job: {job_id}"}
        job = self._refresh_job(job)
        self._save_job(job)
        return {"ok": True, "job": job.to_dict(include_output=True, tail=max(200, min(tail, MAX_OUTPUT)))}


def shlex_quote(s: str) -> str:
    # minimal safe quote
    if s == "":
        return "''"
    if re_safe(s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def re_safe(s: str) -> bool:
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", s))
