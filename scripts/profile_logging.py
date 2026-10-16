"""Bounded CPU logging overhead measurement, with persistent evidence."""
from pathlib import Path
import json
import time
from nnets_transformer.observability import Job

with Job("logging-profile", console_level="WARNING") as job:
    count = 200
    started = time.perf_counter()
    for index in range(count):
        job.event("profile", index=index)
    elapsed = time.perf_counter()-started
    result = {"events": count, "seconds": elapsed, "milliseconds_per_event": elapsed/count*1000,
              "includes": "JSON event, readable file, atomic status; console suppressed",
              "log_directory": str(job.directory)}
    path = job.directory / "profile.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    job.artifact(path)
    job.original_out.write(json.dumps(result, indent=2)+"\n")
