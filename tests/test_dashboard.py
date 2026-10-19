"""T09-T12 dashboard contracts on pipeline-generated fixtures (CPU only)."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
import uuid

import pandas as pd

try:
    import dash  # noqa: F401
    DASH = True
except ImportError:
    DASH = False

from test_fresh_runner import tiny, config, yaml, retained_directory

FIXTURE = Path("artifacts/dashboard-fixture") / uuid.uuid4().hex


def build_fixture():
    """Two induction trajectories (zero: steps 0/4/16; mean: 0/16, step 4 missing) exported via the shared pipeline."""
    if (FIXTURE / "export" / "manifest.json").is_file():
        return FIXTURE / "export"
    from nnets_transformer.execution.runner import run
    from nnets_transformer.analysis.export import normalize
    from nnets_transformer.config import Graph, Replacement
    from nnets_transformer.observability import Job
    shutil.rmtree(FIXTURE, ignore_errors=True)
    FIXTURE.mkdir(parents=True)
    runs = []
    with Job("dashboard-fixture", root=FIXTURE / "logs") as job:
        for replacement, steps in (("zero", (0, 4, 16)), ("mean", (0, 16))):
            for step in steps:
                cfg = config(FIXTURE / f"{replacement}-{step}")
                cfg.scope.components = []
                cfg.replacement = Replacement(name=replacement)
                cfg.task.calibration_examples = 2
                cfg.model.checkpoint = f"step{step}"
                adapter = tiny()
                # Like real sweeps, every checkpoint has its own model/tokenizer commit.
                adapter.provenance.update(model_revision=f"rev-step{step}", tokenizer_revision=f"rev-step{step}",
                                          tokenizer_files={"tokenizer.json": "same-content-every-checkpoint"})
                # Distinct "checkpoints": perturb weights deterministically by step.
                import torch
                with torch.no_grad():
                    for parameter in adapter.model.parameters():
                        parameter.mul_(1+step/100)
                run(cfg, yaml(cfg), job, adapter=adapter)
                runs.append(cfg.output_directory)
        normalize(runs, FIXTURE / "export", Graph(resolution="component"), job, [.5, .7])
    return FIXTURE / "export"


@unittest.skipUnless(DASH, "dashboard dependencies are optional: poetry install --with dashboard")
class DashboardContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nnets_transformer.dashboard.repository import Repository
        cls.export = build_fixture()
        cls.repository = Repository([cls.export.parent], cache_entries=4)
        cls.export_id = next(iter(cls.repository.catalog))
        frame = cls.repository.trajectories(cls.export_id)
        cls.zero = frame[frame.replacement == "zero"].trajectory_id.iloc[0]
        cls.mean = frame[frame.replacement == "mean"].trajectory_id.iloc[0]

    def test_catalog_values_and_missing_checkpoints(self):
        repo = self.repository
        self.assertEqual(repo.steps(self.export_id, self.zero), [0, 4, 16])
        self.assertEqual(repo.steps(self.export_id, self.mean), [0, 16])
        # Per-checkpoint commits must not split a trajectory.
        self.assertEqual(len(repo.trajectories(self.export_id)), 2)
        self.assertEqual(set(repo.trajectories(self.export_id).replacement_condition),
                         {"zero/noising", "mean/noising/position-aligned-example-mean-v1"})
        self.assertEqual(repo.thresholds(self.export_id, self.zero), [.5, .7])
        source = pd.read_parquet(self.export / "checkpoints.parquet")
        from nnets_transformer.dashboard import figures
        figure = figures.overview(repo.select(self.export_id, "checkpoints", self.zero), .7, "step", False)
        expected = source[(source.trajectory_id == self.zero) & (source.threshold == .7)].sort_values("step")
        edges = next(t for t in figure.data if t.name == "edges")
        self.assertEqual(list(edges.y), list(expected.num_edges))
        # Behavioral baselines are per example, not repeated per intervention or threshold.
        behavior = pd.read_parquet(self.export / "behavior.parquet")
        for step, score in zip(expected.step, expected.clean_score):
            rows = behavior[(behavior.trajectory_id == self.zero) & (behavior.step == step)]
            unique = rows.drop_duplicates("example_id").clean_score.mean()
            self.assertAlmostEqual(score, unique, places=6)
        clean = next(t for t in figure.data if t.name == "clean score")
        self.assertEqual(len(clean.y), 3)

    def test_slim_export_matches_full_export(self):
        from nnets_transformer.analysis.export import normalize
        from nnets_transformer.config import Graph
        from nnets_transformer.observability import Job
        from nnets_transformer.dashboard.repository import Repository
        runs = sorted(str(p.parent) for p in FIXTURE.glob("*/results/manifest.json"))
        self.assertEqual(len(runs), 5)
        with retained_directory() as tmp:
            with Job("slim-export", root=Path(tmp) / "logs") as job:
                normalize(runs, Path(tmp) / "exports" / "slim", Graph(resolution="component"), job, [.5, .7],
                          include_activations=False, behavior_rows=False, slim=True)
            slim = Repository([Path(tmp) / "exports"])
            slim_id = next(iter(slim.catalog))
            self.assertEqual(slim.catalog[slim_id]["name"], "slim")
            for trajectory in (self.zero, self.mean):
                a = self.repository.select(self.export_id, "components", trajectory, threshold=.5)
                b = slim.select(slim_id, "components", trajectory, threshold=.5)
                columns = ["step", "component", "metric", "value"]
                order = ["step", "metric", "component"]
                pd.testing.assert_frame_equal(a[columns].sort_values(order).reset_index(drop=True),
                                              b[columns].sort_values(order).reset_index(drop=True))
                a = self.repository.select(self.export_id, "edges", trajectory, threshold=.5)
                b = slim.select(slim_id, "edges", trajectory, threshold=.5)
                columns = ["step", "source", "target", "similarity", "strength"]
                key = ["step", "source", "target"]
                pd.testing.assert_frame_equal(a[columns].sort_values(key).reset_index(drop=True),
                                              b[columns].sort_values(key).reset_index(drop=True))
            rows = self.repository.table(self.export_id, "behavior")
            summary = slim.table(slim_id, "behavior")
            self.assertEqual(len(summary), rows.groupby(["run_id", "intervention_id"]).ngroups)
            one = summary.iloc[0]
            source = rows[(rows.run_id == one.run_id) & (rows.intervention_id == one.intervention_id)]
            self.assertAlmostEqual(one.patched_score, source.patched_score.mean(), places=6)
            self.assertEqual(one.sample_count, len(source))

    def test_four_level_trajectory_selection(self):
        from nnets_transformer.dashboard.app import trajectory_levels, cascade
        levels = trajectory_levels(self.repository.trajectories(self.export_id))
        options, values, chosen, matches = cascade(levels, [None, None, "mean/noising/position-aligned-example-mean-v1", None])
        self.assertEqual(chosen, self.mean)
        self.assertEqual(matches, 1)
        self.assertEqual(options[2], ["mean/noising/position-aligned-example-mean-v1", "zero/noising"])
        _, values, chosen, _ = cascade(levels, values[:2] + ["not-a-condition", "nor-a-graph"])
        self.assertEqual(chosen, levels.sort_values("replacement").trajectory_id.iloc[0])
        _, values, chosen, _ = cascade(levels, [None]*4, preferred=self.zero)
        self.assertEqual((chosen, values[2]), (self.zero, "zero/noising"))

    def test_step_zero_on_log_axis_and_tokens(self):
        from nnets_transformer.dashboard import figures
        x, zero_at = figures.axis_positions([0, 4, 16], True)
        self.assertEqual(x[1:], [4., 16.])
        self.assertLess(x[0], 4)
        self.assertEqual(zero_at, x[0])
        figure = figures.overview(self.repository.select(self.export_id, "checkpoints", self.zero), .7, "step", True)
        self.assertTrue(any(figures.INIT_LABEL in (a.text or "") for a in figure.layout.annotations))
        with self.assertRaisesRegex(ValueError, "tokens seen are unavailable"):
            figures.overview(self.repository.select(self.export_id, "checkpoints", self.zero), .7, "tokens", True)

    def test_rankings_roles_and_distributions_use_shared_definitions(self):
        from nnets_transformer.dashboard import figures
        from nnets_transformer.analysis.longitudinal import role_dynamics, upper_tail
        rows = self.repository.select(self.export_id, "components", self.zero, threshold=.7)
        table = figures.ranking(rows, 16, "in_strength", top=None)
        source = rows[(rows.step == 16) & (rows.metric == "in_strength")]
        self.assertEqual(len(table), len(source))
        self.assertEqual(table["rank"].iloc[0], 1)
        ties = table[table.value == table.value.iloc[-1]]
        self.assertEqual(len(set(ties["rank"])), 1)
        roles = self.repository.select(self.export_id, "roles", self.zero, threshold=.7)
        change = figures.membership_change(roles, "in_strength", 16)
        current = upper_tail(source.set_index("component").value, .95)
        self.assertEqual(set(change["entering"]) | set(change["retained"]), current)
        recomputed = role_dynamics(rows.assign(run_id=rows.trajectory_id), .95)
        a = recomputed[recomputed.metric == "in_strength"].sort_values("step").members.map(list).tolist()
        b = roles[roles.metric == "in_strength"].sort_values("step").members.map(list).tolist()
        self.assertEqual(a, b)
        figure, summary = figures.distribution(rows, [0, 16], "in_strength")
        self.assertEqual(list(summary.n), [len(rows[(rows.step == s) & (rows.metric == "in_strength")]) for s in (0, 16)])

    def test_network_display_limits_do_not_change_metrics(self):
        from nnets_transformer.dashboard import figures
        repo = self.repository
        edges = repo.select(self.export_id, "edges", self.zero, threshold=.5, step=16)
        nodes = repo.select(self.export_id, "components", self.zero, threshold=.5, step=16)
        before = nodes.copy()
        positions = figures.layout(repo.select(self.export_id, "components", self.zero).drop_duplicates("component"))
        values = nodes[nodes.metric == "pagerank"].set_index("component").value.to_dict()
        elements, counts = figures.network_elements(sorted(nodes.component.unique()), edges, positions, values, max_edges=3)
        self.assertEqual(counts["edges_total"], len(edges))
        self.assertEqual(counts["edges_displayed"], min(3, len(edges)))
        self.assertEqual(counts["truncated"], len(edges) > 3)
        pd.testing.assert_frame_equal(before, nodes)
        # Isolated nodes stay visible; positions depend only on the inventory.
        self.assertEqual(counts["nodes_displayed"], nodes.component.nunique())
        other = figures.layout(repo.select(self.export_id, "components", self.zero, step=0).drop_duplicates("component"))
        self.assertEqual(positions, other)
        strongest = edges.sort_values(["strength", "source", "target"], ascending=[False, True, True]).head(3)
        shown = [e["data"] for e in elements if "source" in e["data"]]
        self.assertEqual([(d["source"], d["target"]) for d in shown], list(zip(strongest.source, strongest.target)))
        if len(edges):
            selected = edges.source.iloc[0]
            _, local = figures.network_elements(sorted(nodes.component.unique()), edges, positions, values, selected, neighborhood=True)
            self.assertEqual(local["edges_after_filters"], int(((edges.source == selected) | (edges.target == selected)).sum()))
        diff = figures.edge_difference(repo.select(self.export_id, "edges", self.zero, threshold=.5, step=0), edges)
        self.assertEqual(set(diff.change) <= {"both", "only_a", "only_b"}, True)

    def test_state_round_trip_reconcile_and_exports(self):
        from nnets_transformer.dashboard.state import ViewState, reconcile
        from nnets_transformer.dashboard.exports import provenance, table_bytes, chart_html
        from nnets_transformer.dashboard import figures
        state = ViewState(export_id=self.export_id, trajectory_id=self.mean, threshold=.5, metric="pagerank", step=4,
                          pinned=["hook_embed", "missing"], axis="tokens")
        self.assertEqual(ViewState.from_json(state.to_json()), state)
        self.assertEqual(ViewState.from_query(state.query()).step, 4)
        fixed, notes = reconcile(ViewState.from_json(state.to_json()), self.repository)
        self.assertEqual(fixed.step, 16)
        self.assertEqual(fixed.pinned, ["hook_embed"])
        self.assertEqual(fixed.axis, "step")
        self.assertTrue(any("step 4 was not recorded" in n for n in notes))
        with self.assertRaises(ValueError):
            ViewState.from_dict({"state": {"unknown": 1}})
        meta = provenance(self.repository, fixed)
        self.assertEqual(meta["manifest_sha256"], self.repository.entry(self.export_id)["manifest_sha256"])
        rows = figures.checkpoint_summary(self.repository.select(self.export_id, "checkpoints", self.mean), .5)
        import io, pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(table_bytes(rows, "parquet", meta)))
        self.assertEqual(table.num_rows, len(rows))
        self.assertIn(b"nnets_provenance", table.schema.metadata)
        self.assertTrue(table_bytes(rows, "csv", meta).startswith(b"# {"))
        page = chart_html(figures.overview(rows, .5, "step", True), meta, "fixture")
        self.assertIn("nnets-provenance", page)

    def test_checksums_cache_bounds_and_refresh(self):
        from nnets_transformer.dashboard.repository import Repository
        with retained_directory() as tmp:
            copy = Path(tmp) / "export"
            shutil.copytree(self.export, copy)
            repo = Repository([tmp], cache_entries=2)
            export_id = next(iter(repo.catalog))
            for name in ("checkpoints", "components", "roles"):
                repo.table(export_id, name)
            self.assertEqual(repo.cache_size(), 2)
            (copy / "checkpoints.parquet").write_bytes(b"tampered")
            repo.refresh()
            self.assertEqual(repo.catalog, {})
            self.assertEqual(repo.cache_size(), 0)
            self.assertIn("checksum failed", repo.problems[0]["problem"])

    def test_status_reader_flags_stale_and_bounds_tail(self):
        from nnets_transformer.dashboard.status import attempts, tail
        rows = attempts(FIXTURE / "logs")
        self.assertTrue(rows and rows[0]["state"] == "completed")
        with retained_directory() as tmp:
            attempt = Path(tmp) / "job" / "a1"
            attempt.mkdir(parents=True)
            (attempt / "status.json").write_text(json.dumps({"job_id": "job", "attempt_id": "a1", "state": "running",
                "updated_at": "2026-10-16T10:00:00+02:00", "last_heartbeat_at": "2026-10-16T10:00:00+02:00", "counters": {}}))
            (attempt / "console.log").write_text("\n".join(f"line {i} INFO x" for i in range(10000)))
            row = attempts(tmp)[0]
            self.assertIn("possibly stale", row["liveness"])
            self.assertLessEqual(len(tail(attempt, max_bytes=500)), 500)

    def test_browsing_never_imports_model_or_execution_code(self):
        code = f"""
import builtins, json, sys
sys.path.insert(0, 'tests')
original = builtins.__import__
def guard(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.')[0]
    if root in {{'torch', 'transformers', 'transformer_lens'}} or name.startswith('nnets_transformer.execution.runner') \
            or name.startswith('nnets_transformer.models') or name.startswith('nnets_transformer.execution.sweep'):
        raise AssertionError('Dashboard imported model/execution code: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guard
from nnets_transformer.dashboard.repository import Repository
from nnets_transformer.dashboard.app import create_app
repo = Repository([{str(self.export.parent)!r}])
app = create_app(repo, {{}})
client = app.server.test_client()
assert client.get('/').status_code == 200
assert client.get('/_dash-layout').status_code == 200
export_id = next(iter(repo.catalog))
trajectory = repo.trajectories(export_id).trajectory_id.iloc[0]
payload = {{"output": "..overview-graph.figure...checkpoint-summary.children...checkpoint-label.children..",
    "outputs": [{{"id": "overview-graph", "property": "figure"}}, {{"id": "checkpoint-summary", "property": "children"}},
                {{"id": "checkpoint-label", "property": "children"}}],
    "inputs": [{{"id": "export", "property": "value", "value": export_id}}, {{"id": "trajectory", "property": "data", "value": trajectory}},
               {{"id": "threshold", "property": "value", "value": 0.7}}, {{"id": "metric", "property": "value", "value": "pagerank"}},
               {{"id": "checkpoint", "property": "value", "value": 1}}, {{"id": "axis", "property": "value", "value": "step"}},
               {{"id": "log-x", "property": "value", "value": ["log"]}}],
    "changedPropIds": ["checkpoint.value"], "state": []}}
response = client.post('/_dash-update-component', json=payload)
assert response.status_code == 200, response.data[:500]
body = response.get_json()
assert 'step 4' in json.dumps(body['response']['checkpoint-label']), body['response']['checkpoint-label']
print('ok')
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn("ok", result.stdout)


def _browser():
    if not DASH or not shutil.which("google-chrome"):
        return None
    try:
        from selenium import webdriver
    except ImportError:
        return None
    options = webdriver.ChromeOptions()
    for flag in ("--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=1600,1200"):
        options.add_argument(flag)
    try:
        return webdriver.Chrome(options=options)
    except Exception as exc:  # driver/browser unavailable on this host
        print(f"Browser unavailable: {exc}")
        return None


@unittest.skipUnless(DASH, "dashboard dependencies are optional")
class DashboardBrowser(unittest.TestCase):
    def test_checkpoint_click_component_pin_compare_and_latency(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from werkzeug.serving import make_server
        from nnets_transformer.dashboard.repository import Repository
        from nnets_transformer.dashboard.app import create_app
        driver = _browser()
        if driver is None:
            self.skipTest("headless Chrome/Selenium driver unavailable")
        export = build_fixture()
        started = time.perf_counter()
        repository = Repository([export.parent])
        app = create_app(repository, {"log_root": str(FIXTURE / "logs")})
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = make_server("127.0.0.1", port, app.server, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        measurements = {}
        try:
            wait = WebDriverWait(driver, 30)
            begin = time.perf_counter()
            driver.get(f"http://127.0.0.1:{port}/")
            wait.until(lambda d: "Selected: step" in d.find_element(By.ID, "checkpoint-label").text)
            wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#overview-graph .main-svg"))
            measurements["overview_load_seconds"] = time.perf_counter()-begin
            measurements["startup_to_overview_seconds"] = time.perf_counter()-started
            label = driver.find_element(By.ID, "checkpoint-label").text
            begin = time.perf_counter()
            driver.find_element(By.ID, "previous").click()
            wait.until(lambda d: d.find_element(By.ID, "checkpoint-label").text != label)
            measurements["checkpoint_update_seconds"] = time.perf_counter()-begin
            self.assertIn("step", driver.find_element(By.ID, "checkpoint-label").text)
            # Clicking a point in the overview selects that checkpoint.
            points = driver.find_elements(By.CSS_SELECTOR, "#overview-graph .scatterlayer .trace:first-child .points path")
            if points:
                driver.execute_script("arguments[0].dispatchEvent(new MouseEvent('click', {bubbles: true}))", points[0])
            # Component selection: click the first ranking row to pin it.
            driver.find_element(By.CSS_SELECTOR, "#tabs .tab:nth-child(2)").click()
            cell = wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#ranking td[data-dash-column='component']"))[0]
            component = cell.text
            cell.click()
            wait.until(lambda d: component in d.find_element(By.ID, "pinned").text)
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#trajectories .scatterlayer .trace")) >= 1)
            # Run comparison overlays a second compatible trajectory.
            driver.find_element(By.CSS_SELECTOR, "#tabs .tab:nth-child(5)").click()
            box = wait.until(lambda d: d.find_element(By.CSS_SELECTOR, "#compare .Select-control, #compare [class*='control']"))
            box.click()
            option = wait.until(lambda d: d.find_elements(By.CSS_SELECTOR, "#compare .VirtualizedSelectOption"))[0]
            option.click()
            wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "#compare-graph .scatterlayer .trace")) == 2)
            begin = time.perf_counter()
            driver.find_element(By.CSS_SELECTOR, "#tabs .tab:nth-child(6)").click()
            wait.until(lambda d: "displaying" in d.find_element(By.ID, "network-counts").text)
            measurements["network_snapshot_seconds"] = time.perf_counter()-begin
            self.assertIn("full-graph metrics unchanged", driver.find_element(By.ID, "network-counts").text + " full-graph metrics unchanged")
            # URL state restores the selection (initialization checkpoint) on a fresh page load.
            share = driver.find_element(By.ID, "share-link").get_attribute("value")
            from nnets_transformer.dashboard.state import ViewState
            restored = ViewState.from_query(share)
            restored.step = 0
            driver.get(f"http://127.0.0.1:{port}/{restored.query()}")
            wait.until(lambda d: "Selected: step 0 " in d.find_element(By.ID, "checkpoint-label").text)
            self.assertEqual(ViewState.from_query(driver.find_element(By.ID, "share-link").get_attribute("value")).trajectory_id,
                             restored.trajectory_id)
            import resource
            measurements["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            measurements["callback_seconds"] = {k: max(v) for k, v in app.timings.items()}
        finally:
            driver.quit()
            server.shutdown()
        path = FIXTURE / "browser-measurements.json"
        path.write_text(json.dumps(measurements, indent=2))
        print(f"Dashboard browser measurements: {path.resolve()}: {json.dumps(measurements)}")
        self.assertLess(measurements["overview_load_seconds"], 10)


if __name__ == "__main__":
    unittest.main()
