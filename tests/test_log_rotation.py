import json
from pathlib import Path
import unittest
import uuid
from nnets_transformer.observability import Job


class RotationContracts(unittest.TestCase):
    def test_retains_complete_records_and_relocates(self):
        root = Path("artifacts/log-rotation") / uuid.uuid4().hex
        with Job("rotation", root=root / "bootstrap") as job:
            job.configure(console_level="WARNING", file_level="DEBUG", heartbeat_seconds=.05,
                timezone="Europe/Rome", root=root / "selected", rotation_bytes=2048, telemetry_seconds=0)
            for index in range(40):
                job.event("numbered", index=index)
        records=[]
        for path in job.directory.glob("events*.jsonl"):
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        self.assertEqual(sorted(r['index'] for r in records if r['event']=='numbered'),list(range(40)))
        self.assertGreater(len(list(job.directory.glob("console*.log"))),1)
        self.assertEqual(json.loads((job.directory/'status.json').read_text())['state'],'completed')
        self.assertTrue(any(r['event']=='telemetry' for r in records))
        self.assertEqual(job.directory.parent, (root/'selected/rotation').resolve())
