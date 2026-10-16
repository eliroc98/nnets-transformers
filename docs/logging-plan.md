**Persistent logging and progress requirements**

Every experiment, analysis/export job, dashboard process, and implementation validation command must leave an inspectable record. Logs must explain what is running, what finished, what failed, and how to resume. This specifies the logging implementation required by [the experiment plan](experiment-expansion-plan.md), TODO T13. The current scripts still use terminal `print` output; [README.md](../README.md) provides immediate stdout/stderr capture until structured logging is implemented.

**Locations and formats**

Use stable job/run IDs and a new attempt ID for each invocation, including resumes. Keep previous attempt logs intact. An experiment's artifact manifest points to its log directory; analysis and dashboard sessions identify their input runs and source artifact versions.

```text
logs/
  RUN_ID/
    ATTEMPT_ID/
      console.log       # readable timestamps, messages, warnings, tracebacks
      events.jsonl      # structured lifecycle/progress/metric/artifact events
      status.json       # atomically updated current-state snapshot
      invocation.json   # command, resolved config reference, code/environment IDs
  analysis/JOB_ID/ATTEMPT_ID/...
  dashboard/SESSION_ID/...
```

Print the absolute log directory, status path, and artifact root at startup. Provide a short status summary at shutdown and include log paths in error reports. Use the same logging infrastructure for dependency/setup checks and test/profile commands, recording their command, output log, and exit code. Preserve essential validation evidence in the TODO completion/handoff notes as well.

Use timezone-aware ISO timestamps with offsets; render human-readable times in Europe/Rome by default. Measure elapsed durations with a monotonic clock. JSON event records include severity, event name, run/job/attempt IDs, process/worker identity, phase, and relevant checkpoint/task/replacement/shard identifiers. Keep unavailable values explicit.

Default to INFO on the console and in persistent logs, with configurable DEBUG detail. Route Python logging, warnings, stdout/stderr from legacy/third-party/subprocess work, and uncaught exception tracebacks into the attempt record. Preserve subprocess return codes and return a nonzero status on failure; a successful log-writing command must not hide a failed experiment. Initialize logging before heavy setup/model loading, and fail visibly before starting work if required log files cannot be opened.

**Required lifecycle and progress events**

| Phase | Required information |
|---|---|
| Startup/configuration | Command, working directory, resolved config and hash, code revision/dirty state, exact Python/Poetry environment, dependency versions, selected device/dtype, planned work and log/output paths |
| Dataset preparation | Source/split/sample manifests, seed, requested/retained/excluded counts, candidate/tokenization/alignment checks and reasons for exclusions |
| Checkpoint loading | Immutable revision, stage/step/tokens if known, cache hit/download start/completion, load duration, failures and retry decisions |
| Reference preparation | Clean baseline, mean calibration and donor-cache start/completion, sample counts, durations and validity flags |
| Intervention execution | Current checkpoint/task/replacement, example/shard/intervention counts done/total, elapsed time, throughput, available ETA and periodically sampled memory |
| Scientific summaries | Declared metric/sign/group reduction, clean/corrupt/patched scores, uncertainty/invalid counts when computed, graph/threshold/centrality definitions and approximation settings |
| Analysis/export | Input artifact versions, operation/threshold/metric, completion counts, duration, saved paths and output sizes |
| Persistence/resume | Atomic shard completion and manifest update, validated existing work skipped, remaining work, cache/compatibility decisions, exact resume command |
| Warnings/retries | Actionable cause, context, retry number/limit, changed batch size or other execution settings, final disposition |
| Failure/interruption | Full traceback or signal, last committed work, exit status, saved partial-state/log paths and resume instructions |
| Completion | Actual completed/skipped/failed counts, total duration, output/log locations, final status |

Do not log only start/end messages for a long checkpoint. Emit progress at stage transitions and at least every 10 seconds during controllable processing loops; sample a lightweight heartbeat/status timestamp at that interval during longer blocking work where feasible. Track `last_progress_at` separately from `last_heartbeat_at`, so a live process is not mistaken for advancing work. A heartbeat reports liveness, not successful inference or a completed shard.

Keep separate counters for checkpoints, tasks, examples, shards and interventions; label each denominator. Calculate percent/ETA only when the work total and observed rate support it. Resumed/skipped work must remain distinguishable from work completed in the current invocation. Do not fabricate progress during downloads, native kernels, or stages whose totals are unknown; report the phase, elapsed time and last update instead.

**Status, resumption, and interrupted processes**

`status.json` contains state (`running`, `completed`, `failed`, `interrupted`), phase/context, current counters, started/updated/progress timestamps, elapsed time, process identity, last committed artifact, log references, and resume instructions. Write snapshots through a temporary file plus atomic rename; structured events remain append-only within an attempt.

Update completion only after required artifacts and their manifest entries have committed successfully. Graceful SIGINT/SIGTERM should preserve completed shards, write interruption details, flush logs and propagate an appropriate exit status. SIGKILL, host failure and out-of-memory termination may prevent final writes: an old heartbeat is evidence of potentially stale status, not proof of a particular cause. A monitor/dashboard should mark it potentially stale and check process/job identity before declaring failure. Never infer successful completion from a progress percentage or a missing process alone.

A resumed run gets a new attempt linked to its predecessor, validates configuration/artifact compatibility, and records exactly what it reuses. Logs are diagnostic records; manifests and verified artifacts remain the authority for resumption.

**Implementation and overhead**

Add shared standard-library logging setup and progress reporting, for example `observability.py` and `execution/progress.py`, and configure them from validated YAML. Wire logging into every CLI, legacy wrapper, analysis job and dashboard entry point. A queue/listener or a single writer should serialize worker events so concurrent jobs do not corrupt each other's logs; workers carry context IDs.

Do not format giant tensors, activation arrays, or every token into logs. Persist those observations as artifacts when requested and log their paths/counts. Keep INFO progress updates outside hooks and GPU hot loops; avoid `.item()`/synchronization solely to update a progress bar. Sample hardware telemetry infrequently and label availability. Rate-limit repeated warnings while retaining counts and a final summary.

Flush promptly enough to support `tail -F`; terminal progress bars must not be the only durable progress record. Rotate/compress large files with discoverable segment names, retaining all attempt segments by default. Cleanup/retention policies must be explicit and must not silently remove the evidence needed to inspect a run. Avoid dumping the complete process environment or authentication-bearing arguments; record relevant reproducibility metadata through an allowlist.

Logging configuration should expose levels, log root, progress/heartbeat intervals, telemetry frequency and rotation policy. Scientific artifact versions and log-schema versions are recorded independently. An example schema is illustrative until implemented:

```yaml
logging:
  root: logs
  console_level: INFO
  file_level: INFO
  structured_events: true
  progress_interval_seconds: 10
  heartbeat_interval_seconds: 10
  timezone: Europe/Rome
  retain_all_attempts: true
```

**Dashboard visibility**

Add a run-status/log panel to the [dashboard](dashboard-plan.md). Show the active phase/checkpoint, counters, last progress and heartbeat, warnings/failures, artifact/log paths, and a bounded recent-event tail. Filters include run, attempt, severity and phase. Read only the selected log tail/status; do not load complete logs into browser memory. Log dashboard startup, artifact-query/import errors, long analysis jobs, exports, refreshes and callback failures with enough query/version context to reproduce them. The panel never starts inference or treats logs as scientific metric tables.

**Acceptance criteria**

Implement T13 before new long-running sweeps; integrate its base setup with T02 and its dashboard panel with T09/T12. Use the README's console-capture recipe for earlier legacy/reproduction commands.

Validate on short CPU-only subprocesses/fixtures: success, a deliberate exception, stderr output, nonzero exit, a quiet long-running stage with heartbeat, interruption, resume, and concurrent attempts. Confirm persistent timestamps/context, complete tracebacks, honest progress counters, preserved exit codes, atomic status snapshots, independent logs, and readable output while execution is active. Test invalid/unwritable log roots before work starts. Check that logger configuration is idempotent and does not duplicate handlers/events after notebook or dashboard reloads. Record the actual log paths, checks and overhead measurements in TODO/handoff notes. No full model sweep is needed to validate this infrastructure.
