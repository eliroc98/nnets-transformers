"""Standard-library attempt logs, honest liveness/progress and command capture.

Logs describe work; only validated scientific manifests authorize shard reuse.
Use Job around Python entry points, or this module's CLI around any subprocess.
"""
import argparse
from contextlib import AbstractContextManager
from datetime import datetime
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import resource
import subprocess
import sys
import threading
import time
import traceback
import uuid
import warnings
from zoneinfo import ZoneInfo

LOG_SCHEMA = "attempt-v1"
_ACTIVE_JOB = None


def redact(command):
    result, secret = [], False
    for arg in command:
        if secret:
            result.append("<redacted>")
            secret = False
        elif any(key in arg.lower().split("=", 1)[0] for key in ("password", "token", "api-key", "api_key", "authorization")) and arg.startswith("-"):
            if "=" in arg:
                result.append(arg.split("=", 1)[0] + "=<redacted>")
            else:
                result.append(arg)
                secret = True
        else:
            result.append(arg)
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


class _Writer:
    """Retain every rotated segment; never split JSON records across files."""
    def __init__(self, path, limit=10*1024*1024):
        self.path, self.limit = Path(path), limit
        self.handle = self.path.open("a", buffering=1)
        self.size = self.path.stat().st_size
        self.segment = len(list(self.path.parent.glob(self.path.stem+".[0-9]*"+self.path.suffix)))

    def write(self, text):
        size = len(text.encode())
        if self.limit and self.size and self.size+size > self.limit:
            self.handle.flush()
            self.handle.close()
            self.segment += 1
            self.path.rename(self.path.with_name(f"{self.path.stem}.{self.segment:06d}{self.path.suffix}"))
            self.handle = self.path.open("a", buffering=1)
            self.size = 0
        self.handle.write(text)
        self.size += size

    def flush(self):
        self.handle.flush()

    def close(self):
        self.handle.close()


class Interrupted(KeyboardInterrupt):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Signal {signal.Signals(signum).name}")


class _Stream:
    def __init__(self, job, original, severity):
        self.job, self.original, self.severity = job, original, severity
        self.stream = "stderr" if original is job.original_err else "stdout"
        self.pending = ""

    def write(self, value):
        with self.job.lock:
            self.pending += value.replace("\r", "\n")
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                if line:
                    self.job.event("output", severity=self.severity, message=line, stream=self.stream)
        return len(value)

    def flush(self):
        with self.job.lock:
            if self.pending:
                line, self.pending = self.pending, ""
                self.job.event("output", severity=self.severity, message=line, stream=self.stream)
            self.job.console.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return self.original.encoding

    def fileno(self):
        return self.original.fileno()


class _Handler(logging.Handler):
    def __init__(self, job):
        super().__init__(logging.DEBUG)
        self.job = job

    def emit(self, record):
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + "".join(traceback.format_exception(*record.exc_info))
        self.job.event("message", severity=record.levelname, message=message, logger=record.name)


class Job(AbstractContextManager):
    def __init__(self, job_id, *, root="logs", artifact_root=None, command=None,
                 heartbeat_seconds=10., resume_command=None, previous_attempt=None,
                 console_level="INFO", file_level="INFO", timezone="Europe/Rome",
                 rotation_bytes=10*1024*1024, telemetry_seconds=60.):
        if not job_id or Path(job_id).is_absolute() or ".." in Path(job_id).parts:
            raise ValueError("job_id must be a relative path without '..'")
        if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.job_id, self.root = job_id, Path(root).resolve()
        self.zone = ZoneInfo(timezone)
        self.attempt_id = datetime.now(self.zone).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
        self.directory = self.root / job_id / self.attempt_id
        self.artifact_root = str(Path(artifact_root).resolve()) if artifact_root else None
        self.command = redact(list(command if command is not None else sys.argv))
        self.heartbeat_seconds = heartbeat_seconds
        self.console_level = logging._nameToLevel[console_level.upper()]
        self.file_level = logging._nameToLevel[file_level.upper()]
        self.resume_command, self.previous_attempt = resume_command, str(previous_attempt) if previous_attempt else None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.context, self.counters = {}, {}
        self.phase, self.state = "startup", "running"
        self.started = time.monotonic()
        self.started_at = self.now()
        self.last_progress_at, self.last_artifact = None, None
        self.last_heartbeat_at = self.started_at
        self.exit_code = None
        self.signals = {}
        self.rotation_bytes = rotation_bytes
        self.telemetry_seconds = telemetry_seconds
        self.counter_started = {}
        self.process_identity = {"hostname": socket.gethostname(), "pid": os.getpid(), "start_ticks": None}
        try:
            self.process_identity["start_ticks"] = Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19]
        except OSError:
            pass

    def configure(self, *, console_level, file_level, heartbeat_seconds, timezone,
                  root=None, rotation_bytes=10*1024*1024, telemetry_seconds=60.):
        """Apply YAML settings after bootstrap logging, before heavy setup."""
        self.stop.set()
        self.thread.join()
        if root is not None and Path(root).resolve() != self.root:
            new_root = Path(root).resolve()
            target = new_root / self.job_id / self.attempt_id
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            self.console.close()
            self.events.close()
            previous = self.directory
            try:
                shutil.move(str(previous), str(target))
                self.directory, self.root = target, new_root
            finally:
                self.console = _Writer(self.directory / "console.log", rotation_bytes)
                self.events = _Writer(self.directory / "events.jsonl", rotation_bytes)
        self.console_level = logging._nameToLevel[console_level]
        self.file_level = logging._nameToLevel[file_level]
        self.zone = ZoneInfo(timezone)
        self.heartbeat_seconds = heartbeat_seconds
        self.rotation_bytes = rotation_bytes
        self.telemetry_seconds = telemetry_seconds
        self.console.limit = self.events.limit = rotation_bytes
        self.stop.clear()
        self.thread = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        self.thread.start()
        self.event("logging_configured", message=f"Logs: {self.directory}; status: {self.directory / 'status.json'}",
                   console_level=console_level, file_level=file_level, heartbeat_seconds=heartbeat_seconds,
                   timezone=timezone, rotation_bytes=rotation_bytes, retain_all_segments=True)
        self.telemetry()

    def telemetry(self):
        measurements = {"peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        "gpu_allocated_bytes": None, "gpu_reserved_bytes": None, "gpu_peak_bytes": None}
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            measurements.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved(), gpu_peak_bytes=torch.cuda.max_memory_allocated())
        self.event("telemetry", **measurements)

    def now(self):
        return datetime.now(self.zone).isoformat(timespec="milliseconds")

    def _status(self, updated):
        return {"log_schema": LOG_SCHEMA, "job_id": self.job_id, "attempt_id": self.attempt_id,
                "state": self.state, "phase": self.phase, "context": self.context,
                "counters": self.counters, "started_at": self.started_at, "updated_at": updated,
                "last_progress_at": self.last_progress_at, "last_heartbeat_at": self.last_heartbeat_at,
                "elapsed_seconds": time.monotonic()-self.started, "pid": os.getpid(),
                "process_identity": self.process_identity, "log_segments_retained": True,
                "exit_code": self.exit_code, "last_committed_artifact": self.last_artifact,
                "resume_command": self.resume_command, "previous_attempt": self.previous_attempt,
                "logs": {k: str(self.directory / k) for k in ("console.log", "events.jsonl", "status.json", "invocation.json")}}

    def event(self, event, *, severity="INFO", message=None, **fields):
        with self.lock:
            now = self.now()
            record = {"log_schema": LOG_SCHEMA, "timestamp": now, "severity": severity,
                      "event": event, "job_id": self.job_id, "attempt_id": self.attempt_id,
                      "pid": os.getpid(), "worker": threading.current_thread().name,
                      "phase": self.phase, "context": dict(self.context), **fields}
            if message is not None:
                record["message"] = message
            self.events.write(json.dumps(record, allow_nan=False) + "\n")
            self.events.flush()
            level = logging._nameToLevel.get(severity, logging.INFO)
            line = f"{now} {severity} [{self.job_id}/{self.attempt_id}] {self.phase}: {message or event}"
            if level >= self.file_level:
                self.console.write(line + "\n")
                self.console.flush()
            if level >= self.console_level:
                self.original_out.write(line + "\n")
                self.original_out.flush()
            atomic_json(self.directory / "status.json", self._status(now))

    def stage(self, phase, **context):
        self.phase = phase
        self.context.update(context)
        self.event("stage", message=f"Stage: {phase}")

    def progress(self, counter, completed, total=None, *, skipped=0, failed=0):
        if any(type(v) is not int for v in (completed, skipped, failed)) or total is not None and type(total) is not int:
            raise ValueError("Progress counters must be integers")
        if min(completed, skipped, failed) < 0 or total is not None and completed+skipped+failed > total:
            raise ValueError("Invalid progress counters")
        with self.lock:
            started = self.counter_started.setdefault(counter, time.monotonic())
            duration = time.monotonic()-started
            rate = completed/duration if completed and duration > 0 else None
            remaining = total-completed-skipped-failed if total is not None else None
            self.counters[counter] = {"completed": completed, "total": total, "skipped": skipped, "failed": failed}
            self.last_progress_at = self.now()
            self.event("progress", counter=counter, **self.counters[counter], rate_per_second=rate,
                       eta_seconds=remaining/rate if remaining is not None and rate else None,
                       message=f"{counter}: completed={completed}, skipped={skipped}, failed={failed}, total={total}")

    def artifact(self, path):
        path = Path(path).resolve()
        self.last_artifact = {"path": str(path), "size_bytes": path.stat().st_size}
        self.event("artifact_committed", **self.last_artifact)

    def _heartbeat(self):
        heartbeat = telemetry = time.monotonic()
        interval = min(self.heartbeat_seconds, self.telemetry_seconds) if self.telemetry_seconds else self.heartbeat_seconds
        while not self.stop.wait(interval):
            now = time.monotonic()
            if now-heartbeat >= self.heartbeat_seconds:
                with self.lock:
                    self.last_heartbeat_at = self.now()
                    self.event("heartbeat", message="Process alive; progress unchanged")
                heartbeat = now
            if self.telemetry_seconds and now-telemetry >= self.telemetry_seconds:
                self.telemetry()
                telemetry = now

    def __enter__(self):
        global _ACTIVE_JOB
        if _ACTIVE_JOB is not None:
            raise RuntimeError("A Job is already active in this process; reuse it")
        if self.previous_attempt and not (Path(self.previous_attempt) / "status.json").is_file():
            raise ValueError("Previous attempt has no status.json")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.console = _Writer(self.directory / "console.log", self.rotation_bytes)
        self.events = _Writer(self.directory / "events.jsonl", self.rotation_bytes)
        self.original_out, self.original_err = sys.stdout, sys.stderr
        invocation = {"log_schema": LOG_SCHEMA, "command": self.command, "cwd": str(Path.cwd()),
                      "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
                      "artifact_root": self.artifact_root, "resume_command": self.resume_command,
                      "previous_attempt": self.previous_attempt,
                      "environment": {k: os.environ.get(k) for k in ("PYENV_VERSION", "VIRTUAL_ENV", "POETRY_VIRTUALENVS_CREATE", "CUDA_VISIBLE_DEVICES")},
                      "dependencies": {}}
        for name in ("torch", "transformer-lens", "hydra-core", "omegaconf", "networkx"):
            try:
                invocation["dependencies"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                invocation["dependencies"][name] = None
        for key, cmd in {"code_revision": ["git", "rev-parse", "HEAD"], "dirty_state": ["git", "status", "--short"]}.items():
            result = subprocess.run(cmd, capture_output=True, text=True)
            invocation[key] = result.stdout.strip() if result.returncode == 0 else None
        atomic_json(self.directory / "invocation.json", invocation)
        self.event("startup", message=f"Logs: {self.directory}; status: {self.directory / 'status.json'}; artifacts: {self.artifact_root}")
        self.root_logger = logging.getLogger()
        self.old_handlers, self.old_level = self.root_logger.handlers[:], self.root_logger.level
        self.root_logger.handlers = [_Handler(self)]
        self.root_logger.setLevel(logging.DEBUG)
        self.warning_filters = warnings.filters[:]
        self.old_showwarning = warnings.showwarning
        logging.captureWarnings(True)
        sys.stdout, sys.stderr = _Stream(self, self.original_out, "INFO"), _Stream(self, self.original_err, "INFO")
        if threading.current_thread() is threading.main_thread():
            def interrupt(signum, frame):
                raise Interrupted(signum)
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.signals[sig] = signal.signal(sig, interrupt)
        self.thread = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        _ACTIVE_JOB = self
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_JOB
        self.stop.set()
        self.thread.join()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            if exc is None:
                self.state, self.exit_code = "completed", 0
            elif isinstance(exc, KeyboardInterrupt):
                self.state, self.exit_code = "interrupted", 128 + getattr(exc, "signum", signal.SIGINT)
            elif isinstance(exc, SystemExit):
                self.exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
                self.state = "completed" if self.exit_code == 0 else "failed"
            else:
                self.state, self.exit_code = "failed", 1
            if exc is not None and self.exit_code != 0:
                self.event("exception", severity="ERROR", message="".join(traceback.format_exception(exc_type, exc, tb)))
            self.event("shutdown", message=f"{self.state}; exit={self.exit_code}; logs={self.directory}", exit_code=self.exit_code)
        finally:
            sys.stdout, sys.stderr = self.original_out, self.original_err
            self.root_logger.handlers, self.root_logger.level = self.old_handlers, self.old_level
            logging.captureWarnings(False)
            warnings.showwarning = self.old_showwarning
            warnings.filters[:] = self.warning_filters
            for sig, handler in self.signals.items():
                signal.signal(sig, handler)
            self.console.close()
            self.events.close()
            _ACTIVE_JOB = None
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--artifact-root")
    parser.add_argument("--heartbeat-seconds", type=float, default=10.)
    parser.add_argument("--resume-command")
    parser.add_argument("--previous-attempt")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A subprocess command is required after --")
    job = Job(args.job_id, root=args.log_root, artifact_root=args.artifact_root, command=command,
              heartbeat_seconds=args.heartbeat_seconds, resume_command=args.resume_command,
              previous_attempt=args.previous_attempt)
    try:
        with job:
            job.stage("subprocess")
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
            try:
                for line in child.stdout:
                    print(line, end="", flush=True)
                code = child.wait()
            except BaseException as exc:
                try:
                    os.killpg(child.pid, getattr(exc, "signum", signal.SIGTERM))
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
            finally:
                child.stdout.close()
            if code != 0:
                raise SystemExit(code if code > 0 else 128-code)
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)


if __name__ == "__main__":
    main()
