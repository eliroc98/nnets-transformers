"""CPU-only scientific provenance for fresh reference matrix exports."""
import hashlib
import json
from pathlib import Path
from .graphs import Component


def read_fresh(matrix_path):
    """Return verified metadata, or None when the CSV has no fresh-run manifest.

    A partial intervention sample is preserved as partial graph coverage. This
    reader neither loads model code nor silently completes missing measurements.
    """
    matrix_path = Path(matrix_path)
    path = matrix_path.parent / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") not in {"fresh-reference-v1", "fresh-reference-v2"}:
        return None
    if manifest.get("state") != "completed":
        raise ValueError("Fresh artifact run is not completed")
    if matrix_path.name != manifest["artifacts"]["similarities"]:
        raise ValueError("Matrix is not the declared fresh export")
    expected = {**manifest["inputs"], **manifest["output_hashes"]}
    for name, checksum in expected.items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact path")
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Scientific artifact checksum failed: {name}")
    raw_inventory = json.loads((path.parent / "inventory.json").read_text())
    layers = manifest["identity"]["configuration"]["model"]["n_layers"]
    inventory = []
    for component in raw_inventory["components"]:
        inventory.append(Component.parse(component["hook"], layers))
        if component["head_count"] is not None:
            inventory.extend(Component.parse(f"{component['hook']}.{head}", layers)
                             for head in range(component["head_count"]))
    return {"manifest": manifest, "inventory": inventory,
            "scoring_version": manifest["identity"]["scoring_version"],
            "architecture": manifest["identity"]["model"].get("architecture", "parallel"),
            "dataset_sha256": manifest["identity"]["dataset_sha256"],
            "inventory_provenance": raw_inventory["version"],
            "aggregation": "mean_per_example_cosine_then_similarity_threshold"}
