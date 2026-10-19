"""Cluster shell lifecycle checks without Slurm, GPUs, models or network access."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


class ClusterScriptContracts(unittest.TestCase):
    def test_slurm_advance_term_requeues_only_when_enabled(self):
        for auto, expected in (("1", 0), ("0", 143)):
            with self.subTest(auto_requeue=auto), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "scripts").mkdir()
                (root / "bin").mkdir()
                shutil.copyfile("scripts/full_model.sbatch", root / "scripts/full_model.sbatch")
                (root / "scripts/cluster_env.sh").write_text('export PATH="$PWD/bin:$PATH"\n')
                (root / "scripts/run_everything.sh").write_text('''sleeper=""
trap 'kill "$sleeper" 2>/dev/null; wait "$sleeper" 2>/dev/null; touch terminated; exit 143' TERM
sleep 30 &
sleeper=$!
touch ready
wait
''')
                (root / "bin/poetry").write_text('''#!/bin/bash
while [ "$#" -gt 0 ]; do
    if [ "$1" = -- ]; then shift; command=("$@"); continue; fi
    shift
done
exec "${command[@]}"
''')
                (root / "bin/scontrol").write_text('#!/bin/bash\nprintf "%s\\n" "$*" > requeue\n')
                for path in (root / "bin").iterdir():
                    path.chmod(0o755)
                process = subprocess.Popen(["bash", "scripts/full_model.sbatch", "EleutherAI/pythia-14m"], cwd=root,
                    env={**os.environ, "PROJECT_DIR": str(root), "SLURM_JOB_ID": "123", "AUTO_REQUEUE": auto},
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                try:
                    deadline = time.monotonic()+10
                    while not (root / "ready").exists() and time.monotonic() < deadline:
                        if process.poll() is not None:
                            self.fail(process.communicate()[0])
                        time.sleep(.02)
                    self.assertTrue((root / "ready").exists())
                    process.send_signal(signal.SIGTERM)
                    output, _ = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, expected, output)
                    self.assertTrue((root / "terminated").exists())
                    self.assertEqual((root / "requeue").exists(), auto == "1")
                    if auto == "1":
                        self.assertEqual((root / "requeue").read_text().strip(), "requeue 123")
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()

    def test_model_lock_and_batch_shell_signal_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "bin").mkdir()
            shutil.copyfile("scripts/run_everything.sh", root / "scripts/run_everything.sh")
            (root / "scripts/cluster_env.sh").write_text('export PROJECT_DIR="$PWD"\nexport PATH="$PWD/bin:$PATH"\n')
            for name, body in {
                "python": '#!/bin/bash\necho "[step0]"\n',
                "poetry": '''#!/bin/bash
if [[ "$*" == *export_results.py* ]]; then touch exported; exit 0; fi
trap 'touch terminated; exit 143' TERM INT
touch ready
sleep 30 &
wait
''',
            }.items():
                path = root / "bin" / name
                path.write_text(body)
                path.chmod(0o755)
            env = {**os.environ, "MODELS": "EleutherAI/pythia-14m", "MIN_FREE_GB": "0", "KEEP": "means"}
            command = ["bash", "scripts/run_everything.sh"]
            first = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                deadline = time.monotonic()+10
                while not (root / "ready").exists() and time.monotonic() < deadline:
                    if first.poll() is not None:
                        self.fail(first.communicate()[0])
                    time.sleep(.02)
                self.assertTrue((root / "ready").exists())
                duplicate = subprocess.run(command, cwd=root, env={**env, "OUTPUT_ROOT": "runs/other"},
                                           capture_output=True, text=True, timeout=10)
                self.assertEqual(duplicate.returncode, 1, duplicate.stdout+duplicate.stderr)
                self.assertIn("already has an active job", duplicate.stdout)
                first.send_signal(signal.SIGTERM)  # Slurm --signal=B:TERM reaches only this shell.
                output, _ = first.communicate(timeout=10)
                self.assertEqual(first.returncode, 143, output)
                self.assertTrue((root / "terminated").exists(), output)
                self.assertFalse((root / "exported").exists(), output)
                self.assertEqual(len(list((root / "logs/run-everything").glob("*.log"))), 2)
            finally:
                if first.poll() is None:
                    os.killpg(first.pid, signal.SIGKILL)
                    first.communicate()
