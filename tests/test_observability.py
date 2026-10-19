import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
import uuid
import warnings

from nnets_transformer.observability import Job, redact

ROOT = Path("artifacts/validation-t13") / uuid.uuid4().hex
ROOT.mkdir(parents=True)


class LoggingContracts(unittest.TestCase):
    def command(self, name, code, *extra):
        return [sys.executable, "-u", "-m", "nnets_transformer.observability", "--job-id", name,
                "--log-root", str(ROOT / "logs"), "--heartbeat-seconds", ".05", *extra,
                "--", sys.executable, "-u", "-c", code]

    def attempt(self, name):
        # Distributed filesystems can give two attempt directories identical mtimes.
        return max((ROOT / "logs" / name).iterdir(),
                   key=lambda p: json.loads((p / "status.json").read_text())["started_at"])

    def records(self, attempt):
        return [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]

    def test_success_stderr_exception_and_exit(self):
        for name, code, exit_code, state in [
            ("success", "import sys; print('out'); print('err',file=sys.stderr)", 0, "completed"),
            ("exception", "raise RuntimeError('deliberate failure')", 1, "failed"),
            ("nonzero", "raise SystemExit(7)", 7, "failed"),
        ]:
            result = subprocess.run(self.command(name, code), capture_output=True, text=True)
            attempt = self.attempt(name)
            print(f"{name}: {attempt.resolve()}")
            self.assertEqual(result.returncode, exit_code, result.stderr)
            status = json.loads((attempt / "status.json").read_text())
            self.assertEqual((status["state"], status["exit_code"]), (state, exit_code))
            console = (attempt / "console.log").read_text()
            self.assertIn(str(attempt.resolve()), console)
            if name == "success":
                self.assertIn("out", console)
                self.assertIn("err", console)
            elif name == "exception":
                self.assertIn("Traceback", console)
                self.assertIn("RuntimeError: deliberate failure", console)
            self.assertTrue(all("timestamp" in r and r["attempt_id"] == attempt.name for r in self.records(attempt)))

    def test_quiet_heartbeat_and_live_atomic_status(self):
        child = subprocess.Popen(self.command("quiet", "import time; print('ready',flush=True); time.sleep(.35)"),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        snapshots = []
        deadline = time.monotonic() + 10
        while child.poll() is None and time.monotonic() < deadline:
            for path in (ROOT / "logs" / "quiet").glob("*/status.json"):
                snapshots.append(json.loads(path.read_text()))
            time.sleep(.01)
        output, _ = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 0, output)
        running = [s for s in snapshots if s["state"] == "running"]
        self.assertTrue(running)
        self.assertTrue(all(s["last_progress_at"] is None for s in running))
        self.assertGreater(len({s["last_heartbeat_at"] for s in running}), 1)
        self.assertTrue(any(r["event"] == "heartbeat" for r in self.records(self.attempt("quiet"))))

    def test_interrupt_and_resume_preserve_attempt(self):
        child = subprocess.Popen(self.command("resume", "import time; print('ready',flush=True); time.sleep(30)"),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            line = child.stdout.readline()
            if "ready" in line:
                break
            if child.poll() is not None:
                self.fail("subprocess stopped before ready")
        child.send_signal(signal.SIGTERM)
        child.communicate(timeout=10)
        self.assertEqual(child.returncode, 143)
        previous = self.attempt("resume")
        status = json.loads((previous / "status.json").read_text())
        self.assertEqual(status["state"], "interrupted")
        old = (previous / "events.jsonl").read_bytes()
        result = subprocess.run(self.command("resume", "print('resumed')", "--previous-attempt", str(previous),
                                             "--resume-command", "repeat validated shards"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        new = self.attempt("resume")
        self.assertNotEqual(previous, new)
        self.assertEqual(old, (previous / "events.jsonl").read_bytes())
        self.assertEqual(json.loads((new / "status.json").read_text())["previous_attempt"], str(previous))

    def test_concurrent_attempts(self):
        processes = [subprocess.Popen(self.command("concurrent", f"import time; print('worker-{i}'); time.sleep(.15)"),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for i in range(2)]
        for process in processes:
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, output)
        attempts = list((ROOT / "logs" / "concurrent").iterdir())
        self.assertEqual(len(attempts), 2)
        records = [(p / "console.log").read_text() for p in attempts]
        self.assertTrue(all(("worker-0" in r) != ("worker-1" in r) for r in records))

    def test_progress_warning_artifact_and_reconfiguration(self):
        prior_handlers = logging.getLogger().handlers[:]
        for i in range(2):
            with Job("inprocess", root=ROOT / "logs", heartbeat_seconds=1) as job:
                job.stage("validate", checkpoint="step1")
                job.progress("shards", 1, 3, skipped=1)
                status = json.loads((job.directory / "status.json").read_text())
                progress_at = status["last_progress_at"]
                self.assertIsNotNone(progress_at)
                job.event("check")
                self.assertEqual(json.loads((job.directory / "status.json").read_text())["last_progress_at"], progress_at)
                warnings.warn(f"warning-{i}")
                logging.info("one-record")
                artifact = ROOT / f"artifact-{i}.txt"
                artifact.write_text("complete")
                job.artifact(artifact)
            self.assertEqual(logging.getLogger().handlers, prior_handlers)
            self.assertEqual(sum(r.get("message") == "one-record" for r in self.records(job.directory)), 1)
            self.assertTrue(any(f"warning-{i}" in r.get("message", "") for r in self.records(job.directory)))
            self.assertEqual(json.loads((job.directory / "status.json").read_text())["last_committed_artifact"]["size_bytes"], 8)

    def test_runtime_configuration_and_sigint(self):
        with Job("configured", root=ROOT / "logs") as job:
            job.configure(console_level="WARNING", file_level="DEBUG", heartbeat_seconds=.02, timezone="Europe/Rome")
            job.event("detail", severity="DEBUG", message="persisted-debug-record")
            time.sleep(.08)
        records = self.records(job.directory)
        self.assertTrue(any(r["event"] == "heartbeat" for r in records))
        self.assertIn("persisted-debug-record", (job.directory / "console.log").read_text())
        child = subprocess.Popen(self.command("sigint", "import time; print('ready',flush=True); time.sleep(30)"),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            line = child.stdout.readline()
            if "ready" in line:
                break
            if child.poll() is not None:
                self.fail("subprocess stopped before ready")
        child.send_signal(signal.SIGINT)
        child.communicate(timeout=10)
        self.assertEqual(child.returncode, 130)
        status = json.loads((self.attempt("sigint") / "status.json").read_text())
        self.assertEqual((status["state"], status["exit_code"]), ("interrupted", 130))

    def test_unwritable_root_prevents_work(self):
        if os.geteuid() == 0:
            self.skipTest("Permission denial requires an unprivileged process")
        root = ROOT / "read-only"
        root.mkdir()
        root.chmod(0o500)
        marker = ROOT / "must-not-run"
        try:
            command = [sys.executable, "-m", "nnets_transformer.observability", "--job-id", "blocked",
                       "--log-root", str(root), "--", sys.executable, "-c",
                       f"from pathlib import Path; Path({str(marker)!r}).touch()"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PermissionError", result.stderr)
            self.assertFalse(marker.exists())
        finally:
            root.chmod(0o700)

    def test_invalid_root_prevents_work_and_redacts(self):
        root = ROOT / "not-a-directory"
        root.write_text("keep")
        marker = ROOT / "must-not-exist"
        result = subprocess.run([sys.executable, "-m", "nnets_transformer.observability", "--job-id", "bad",
                                 "--log-root", str(root), "--", sys.executable, "-c",
                                 f"from pathlib import Path; Path({str(marker)!r}).touch()"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())
        self.assertEqual(redact(["run", "--token", "secret", "--password=secret", "--safe", "x"]),
                         ["run", "--token", "<redacted>", "--password=<redacted>", "--safe", "x"])


if __name__ == "__main__":
    unittest.main()
