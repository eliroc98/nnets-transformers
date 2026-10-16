"""CPU-only reader for attempt logs/status (`logs/<job>/<attempt>/`)."""
from datetime import datetime
import json
from pathlib import Path


def attempts(root, limit=100, stale_factor=3.0, now=None):
    """Most recent attempts first. Running attempts with an old heartbeat are flagged, not declared dead."""
    root = Path(root)
    rows = []
    if not root.is_dir():
        return rows
    for path in root.glob("*/*/status.json"):
        try:
            status = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            rows.append({"job_id": path.parent.parent.name, "attempt_id": path.parent.name, "state": "unreadable status",
                         "directory": str(path.parent)})
            continue
        rows.append({"job_id": status.get("job_id"), "attempt_id": status.get("attempt_id"), "state": status.get("state"),
                     "phase": status.get("phase"), "updated_at": status.get("updated_at"),
                     "last_progress_at": status.get("last_progress_at"), "last_heartbeat_at": status.get("last_heartbeat_at"),
                     "exit_code": status.get("exit_code"), "counters": status.get("counters", {}),
                     "resume_command": status.get("resume_command"), "directory": str(path.parent),
                     "heartbeat_interval": (status.get("context") or {}).get("heartbeat_seconds")})
    rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    rows = rows[:limit]
    now = now or datetime.now().astimezone()
    for row in rows:
        row["liveness"] = ""
        if row.get("state") == "running" and row.get("last_heartbeat_at"):
            age = (now - datetime.fromisoformat(row["last_heartbeat_at"])).total_seconds()
            interval = row.get("heartbeat_interval") or 10
            row["liveness"] = f"possibly stale: no heartbeat for {age:.0f}s" if age > stale_factor*interval else f"heartbeat {age:.0f}s ago"
        row["counters"] = "; ".join(f"{name}: {c.get('completed')}/{c.get('total') if c.get('total') is not None else '?'}"
                                    f"{' (skipped ' + str(c['skipped']) + ')' if c.get('skipped') else ''}"
                                    for name, c in (row["counters"] or {}).items())
    return rows


def tail(directory, name="console.log", max_bytes=20000, severity=None):
    """Bounded tail; never reads whole large logs."""
    path = Path(directory) / name
    if not path.is_file():
        return "(no log file)"
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        handle.seek(max(0, size-max_bytes))
        text = handle.read().decode(errors="replace")
    lines = text.splitlines()[1:] if size > max_bytes else text.splitlines()
    if severity:
        lines = [line for line in lines if f" {severity} " in line]
    return "\n".join(lines)
