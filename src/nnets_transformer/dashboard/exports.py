"""Exports with provenance: filtered tables, standalone charts and view state."""
from datetime import datetime, timezone
import html
import io
import json


def provenance(repository, state):
    entry = repository.entry(state.export_id)
    manifest = entry["manifest"]
    trajectory = repository.trajectories(state.export_id)
    row = trajectory[trajectory.trajectory_id == state.trajectory_id]
    return {"export_id": state.export_id, "export_path": entry["path"], "manifest_sha256": entry["manifest_sha256"],
            "legacy": entry["legacy"], "limitations": entry["limitations"],
            "sources": [{k: s.get(k) for k in ("path", "run_id", "manifest_sha256")} for s in manifest.get("sources", [])],
            "graph": manifest.get("graph"), "roles": manifest.get("roles"),
            "trajectory": row.iloc[0].to_dict() if len(row) else None,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def table_bytes(frame, fmt, meta):
    if fmt == "parquet":
        buffer = io.BytesIO()
        out = frame.copy()
        out.attrs = {}
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(out, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"nnets_provenance": json.dumps(_clean(meta), default=str).encode()})
        pq.write_table(table, buffer)
        return buffer.getvalue()
    if fmt == "csv":
        header = "".join(f"# {line}\n" for line in json.dumps(_clean(meta), default=str, sort_keys=True).splitlines())
        return (header + frame.to_csv(index=False)).encode()
    raise ValueError("Table export format must be csv or parquet")


def chart_html(figure, meta, title):
    """Standalone interactive chart (one figure, not the dashboard application)."""
    body = figure.to_html(include_plotlyjs=True, full_html=True)
    block = f'<script type="application/json" id="nnets-provenance">{html.escape(json.dumps(_clean(meta), default=str))}</script>'
    head = f"<title>{html.escape(title)}</title>{block}"
    return body.replace("<head>", "<head>" + head, 1)
