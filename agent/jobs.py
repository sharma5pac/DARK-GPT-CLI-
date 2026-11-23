"""Lightweight in-process background jobs for DarkGPT.

Lets the agent fire shell commands/scripts without blocking on one terminal,
then poll/wait/log/kill them. Jobs live for the process lifetime; output is
bounded and kept in memory + optional log file under ~/.cache/darkgpt/jobs.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import scripting

JOB_DIR = Path.home() / ".cache" / "darkgpt" / "jobs"
MAX_OUTPUT = 24000
DEFAULT_WAIT = 30
MAX_WAIT = 300


@dataclass
class Job:
    id: str
    kind: str  # shell|script
    label: str
    cwd: str
    command: List[str]
    script_path: str = ""
    status: str = "running"  # running|done|error|killed|timeout
    exit_code: Optional[int] = None
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    output: List[str] = field(default_factory=list)
    output_len: int = 0
    log_path: str = ""
    _proc: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def to_dict(self, include_output: bool = False, tail: int = 4000) -> Dict[str, Any]:
        out = "".join(self.output)
        if len(out) > tail:
            out = out[-tail:]
            out = f"...[tail {tail} chars]\n{out}"
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
            "duration": max(0.0, (self.ended or time.time()) - self.started),
            "log_path": self.log_path or None,
            "output_bytes": self.output_len,
        }
        if include_output:
            d["output"] = out
        return d

    def append(self, text: str) -> None:
        if not text:
            return
        self.output_len += len(text)
        self.output.append(text)
        total = sum(len(c) for c in self.output)
        if total > MAX_OUTPUT:
            joined = "".join(self.output)
            self.output = [joined[-MAX_OUTPUT:]]
        try:
            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(text)
        except Exception:
            pass


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def _new_id(self) -> str:
        return f"job_{uuid.uuid4().hex[:10]}"

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
            JOB_DIR.mkdir(parents=True, exist_ok=True)
            Path(job.log_path).write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {job.kind} {job.label}\n", encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=job.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"cannot start job: {exc}"}
        job._proc = proc
        async with self._lock:
            self._jobs[job.id] = job
        asyncio.create_task(self._pump(job))
        return {"ok": True, "job": job.to_dict(), "message": f"Started {job.id}. Use task_wait/task_log/task_list to monitor."}

    async def _pump(self, job: Job) -> None:
        proc = job._proc
        try:
            assert proc is not None
            if proc.stdout is not None:
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    job.append(line.decode("utf-8", "replace"))
            code = await proc.wait()
            job.exit_code = int(code)
            job.status = "done" if code == 0 else "error"
        except asyncio.CancelledError:
            try:
                if proc and proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            except Exception:
                pass
            job.status = "killed"
        except Exception as exc:  # noqa: BLE001
            job.append(f"\n[job error] {exc}\n")
            job.status = "error"
            job.exit_code = job.exit_code if job.exit_code is not None else -1
        finally:
            job.ended = time.time()
            job.append(f"\n[exit {job.exit_code} status={job.status} duration={max(0.0, job.ended-job.started):.2f}s]\n")
            job._done.set()

    def list(self, limit: int = 20) -> Dict[str, Any]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.started, reverse=True)[: max(1, min(limit, 100))]
        return {"ok": True, "jobs": [j.to_dict() for j in jobs], "count": len(self._jobs)}

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(str(job_id or "").strip())

    async def wait(self, job_ids: List[str], timeout: int = DEFAULT_WAIT) -> Dict[str, Any]:
        ids = [str(i).strip() for i in (job_ids or []) if str(i).strip()]
        jobs = [self._jobs[i] for i in ids if i in self._jobs]
        missing = [i for i in ids if i not in self._jobs]
        if not jobs and missing:
            return {"ok": False, "error": "unknown job ids", "missing": missing}
        timeout = max(1, min(int(timeout or DEFAULT_WAIT), MAX_WAIT))
        try:
            await asyncio.wait_for(asyncio.gather(*(j._done.wait() for j in jobs)), timeout=timeout)
            done = True
        except asyncio.TimeoutError:
            done = False
        return {
            "ok": True,
            "done": done,
            "jobs": [j.to_dict(include_output=True) for j in jobs],
            "missing": missing,
        }

    async def kill(self, job_id: str) -> Dict[str, Any]:
        job = self.get(job_id)
        if not job:
            return {"ok": False, "error": f"unknown job: {job_id}"}
        if job.status == "running" and job._proc and job._proc.returncode is None:
            try:
                job._proc.kill()
                await job._proc.wait()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"kill failed: {exc}", "job": job.to_dict(include_output=True)}
        return {"ok": True, "job": job.to_dict(include_output=True)}

    def log(self, job_id: str, tail: int = 4000) -> Dict[str, Any]:
        job = self.get(job_id)
        if not job:
            return {"ok": False, "error": f"unknown job: {job_id}"}
        return {"ok": True, "job": job.to_dict(include_output=True, tail=max(200, min(tail, MAX_OUTPUT)))}
