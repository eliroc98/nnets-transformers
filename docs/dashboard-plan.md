**Interactive dashboard plan — 16 October 2026**

Purpose: explore graph metrics, component roles, and behavioral performance over training checkpoints, using saved experimental results. The dashboard is implemented under `src/nnets_transformer/dashboard/` (T09–T12; see the main plan for evidence); views are tabs in `app.py` rather than a separate `views/` package. The [experiment plan](experiment-expansion-plan.md) remains the canonical TODO and session-handoff record.

**First version and user flow**

Start with metric trajectories and component drill-down using the existing induction results. Support new artifact schemas through the same data-access layer as they become available. Network exploration is a later dashboard milestone; the metrics interface should be useful without it and without waiting for all model/task sweeps.

A typical session: choose a run, task, replacement and graph definition; inspect metric trajectories; click a training step; examine that checkpoint's component rankings and distributions; select components to follow over time; compare another checkpoint or compatible run; export the selected evidence and view settings.

```text
Run / model / task / replacement / graph resolution / threshold
Training axis: steps | tokens seen | normalized progress

[ Node count ] [ Edge count ] [ Density ] [ Behavioral score ]
[ Metric trajectories across training        | Selected step ]
[ Strength / centrality distributions       | Component ranks ]
[ Selected component trajectories           | Details         ]

Overview | Components | Compare | Network (later)
Checkpoint selector / previous / next / optional playback
```

The selected checkpoint, components and run filters are shared between panels. Selecting a node in the later network view opens the same component details; selecting a component in a ranking or heatmap updates its trajectory. Brushing a training interval zooms the linked trajectory panels without silently changing the selected checkpoint.

**Views and interactions**

| View | Measurements | Interactions |
|---|---|---|
| Training overview | Node/edge counts, conventional and admissible-pair density, behavior, selected metric summaries | Linked x axes, hover with exact checkpoint metadata, click-to-select, interval zoom, threshold/run overlays |
| Components | Weighted in/out degree, betweenness, incoming/outgoing closeness when available, PageRank, layer/head identity | Search, metric ranking, component pinning, metric-by-step heatmap, raw-value and rank trajectories |
| Distributions | Edge strengths and component metric distributions | Select checkpoint(s), histogram/ECDF and quantiles, consistent bins and sample counts |
| Role dynamics | Upper-tail membership, turnover, Jaccard overlap, checkpoint similarity and layer occupancy | Configurable percentile with declared tie policy, select checkpoint pair, inspect entering/leaving components |
| Run comparison | Selected graph/behavior metrics and component summaries | Facets or overlays by size, family, task, replacement or training seed; explicit reference and alignment axis |
| Network explorer | Directed influence graph at a selected checkpoint | Stable layer layout, metric coloring/sizing, neighborhood selection, strength filtering, graph-difference view |
| Run status/logs | Phase, checkpoint, completed/total counters, last progress/heartbeat, warnings/errors and log locations | Run/attempt/severity filters, bounded recent-event tail, refresh and resume instructions |

Default to unsmoothed trajectories; optionally overlay a labeled smoothing method/window while keeping measured values available. Show phase annotations only when recorded for the selected run; historical Pythia-14M boundaries are not global defaults. Show uncertainty only when supported by saved observations and label whether it reflects examples, donor draws, or independent training runs.

Use separate aligned panels for behavioral and graph quantities. IOI/pronoun logit differences, greater-than/SVA probability differences, losses, and graph metrics retain their definitions and units. Do not present different task scores as one shared performance scale.

**Filters, checkpoint navigation, and comparisons**

Global selectors: run ID; family and scale; training seed; task and dataset; replacement and donor/calibration settings; component/head graph resolution; graph-semantics version; node-inventory convention; threshold; dtype; analysis method. Populate valid combinations from manifests and available partitions, with unavailable combinations explained.

Use training step as the initial x axis and allow tokens seen or within-stage normalized progress when metadata supports them. Do not infer token counts for unrelated families from Pythia's batch size. Present OLMo stages/ingredients as separate trajectories by default.

The checkpoint control indexes actual available checkpoints sorted numerically; irregular checkpoint spacing remains visible in the chart. Previous/next and playback select recorded snapshots, with no fabricated intermediate graphs. A logarithmic axis must preserve initialization: display step zero separately and clearly, or use an explicitly labeled transform. Missing expected snapshots and partial runs have visible coverage/status, and unsupported metrics show as unavailable rather than zero.

Compare full-inventory and active-node counts explicitly. Strength and shortest-path distance conventions, approximation settings, smoothing, and percentile selection remain in the view metadata. Cross-family comparisons use matched graph resolutions and relative depth where relevant; component/head IDs are linked across checkpoints within the same run, not treated as equivalent across independent training runs or families.

**Data contract and legacy migration**

The dashboard is a CPU artifact reader. It never imports the experiment runner or model loaders, starts a forward pass, or silently recomputes missing scientific results. Shared analysis functions remain in `analysis/`; browser callbacks orchestrate queries and visualizations.

| Artifact | Required data |
|---|---|
| Run manifest | Run/model/tokenizer IDs, task/scoring definitions, replacement and seed settings, source/schema versions, completion state |
| Checkpoint table | Checkpoint ID, stage, step, tokens seen if known, ordering, completion and artifact references |
| Graph metric table | Run/checkpoint/threshold/resolution/graph-version keys, metric name/value, inventory and distance conventions |
| Component inventory and metrics | Canonical component ID, type/layer/head, checkpoint and graph keys, metric name/value, validity/status |
| Edge table or sparse matrix | Canonical source/target IDs, unthresholded strength or similarity with declared semantics, checkpoint/run keys |
| Behavioral table | Task-specific primary metric, clean/corrupt/patched values when available, sample count, score definition and optional uncertainty |
| Analysis metadata | Percentile/tie rules, smoothing, centrality approximation parameters, source artifact hashes |

Prefer partitioned Parquet plus JSON/YAML manifests, reusing the experiment artifact contract. Preserve graph and metric versions in every query key; conversions must not silently repair historical semantics.

For legacy inputs:

- `all_metrics.pickle` supplies checkpoint/threshold counts, density, and associated behavioral values.
- `all_distributions.pickle` supplies per-component strengths/centralities and checkpoint/threshold metadata.
- `edge_weights.pickle` supplies distributions but lacks endpoint identities. It cannot reconstruct graph connectivity.
- `ablation_data/step*.csv` similarity matrices or existing graph exports can supply connectivity through an explicit, versioned adapter.
- `induction_data.json` and `loss_data.json` supply historical task measurements. The legacy fields need labels describing their fixed passage and token score.

Use an explicit local import/conversion step for existing trusted pickle artifacts; normal browsing consumes normalized tables. Do not mutate the source files. Mark unavailable provenance, uncertainty, isolated-node inventory, or token counts as unknown. Checkpoint-dependent behavioral baselines repeated across threshold rows must be deduplicated before plotting or aggregation.

**Technology and package integration**

Use Plotly Dash for linked charts and controls: Plotly is already a project dependency, and this keeps the application in Python alongside the existing analysis. Dash supports clientside callbacks and shared state patterns suitable for linked views. [Official clientside callback documentation](https://dash.plotly.com/clientside-callbacks).

Add Dash Cytoscape only for the network milestone; it provides interactive graph layouts and node/edge selection events. [Official Cytoscape documentation](https://dash.plotly.com/cytoscape), [interaction events](https://dash.plotly.com/cytoscape/events).

Proposed layout:

```text
src/nnets_transformer/dashboard/
  __main__.py          # local artifact-root/host/port configuration
  app.py               # app factory; no loading at import time
  catalog.py           # discover completed runs and available dimensions
  repository.py        # versioned artifact queries and bounded cache
  legacy.py            # explicit legacy-to-normalized import
  state.py             # validated filters and checkpoint/component selections
  figures.py           # pure Plotly figure builders
  callbacks.py         # linked selections and query orchestration
  views/
    overview.py
    components.py
    compare.py
    network.py
  exports.py           # chart/data/state exports with provenance
configs/dashboard.yaml
tests/dashboard/       # fixtures, query/figure checks and browser interactions
```

Declare optional dashboard dependencies separately from core inference dependencies, but install/manage them with Poetry in the same required pyenv environment `nnets-transformer` at `/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer`. Keep Poetry virtualenv creation disabled; follow [README setup](../README.md) for activation and verification. Validate CPU-only dashboard behavior in that environment without model downloads or CUDA access. Use localhost by default, with an SSH tunnel for access to a dashboard running on the experiment server. Hosted deployment is a separate decision. Implemented launch (T12): `poetry install --with dashboard`, then `poetry run python -m nnets_transformer.dashboard serve` (see the README); legacy pickles are imported explicitly with `python -m nnets_transformer.dashboard import-legacy`.

**Responsiveness and scale**

- Read summary tables first; load component and edge partitions only for selected views/checkpoints. Keep the large graph pickle and full activation data out of startup paths.
- Cache filtered summaries, role computations and graph layouts using source manifest/hash, filters and metric/graph versions. Keep cache size bounded and invalidate on new artifact versions.
- Keep only compact selection state in browser storage; large tables stay server-side. Avoid mutable global selection state so browser sessions do not overwrite each other.
- Use precomputed thresholds initially. A custom threshold requiring new centrality runs belongs to an explicit CPU analysis job, with a pending state, cancellation/stale-result handling and reusable outputs. A slider should not launch expensive centrality calculations repeatedly.
- Preserve zoom/selection on unrelated updates; debounce expensive controls, apply multi-filter changes together, and discard out-of-order callback responses. Snapshot the selected manifest version so charts share one dataset state.
- Limit network display size explicitly and show displayed-versus-total counts. Use a deterministic layout keyed to the complete component inventory for one architecture so nodes do not move unnecessarily across checkpoints. Top-edge/neighborhood limits affect the view, not the full-graph metric values.
- Profile WebGL traces and aggregation for larger displays rather than sending every observation to the browser. Chart reduction must retain genuine extrema and leave original values available for export. [Dash performance guidance](https://dash.plotly.com/performance).
- Initial performance targets, to validate on a documented artifact/browser/CPU fixture: overview load within 3 seconds after server startup, warm summary/filter updates within 500 ms, and a bounded network snapshot within 2 seconds. These are engineering targets, not measured results. Track payload size and peak process memory alongside latency.

**Process logs and run monitoring**

Follow [the shared logging specification](logging-plan.md), T13. The dashboard exposes a bounded log/status panel for selected runs and records its own startup, queries/import errors, long analysis jobs, exports, refreshes and callback failures under `logs/dashboard/SESSION_ID/`. Link process activity to artifact versions and preserve tracebacks/exit information. Show heartbeat and last actual progress separately, identify potentially stale status honestly, and keep prior attempts inspectable. Legacy console captures lack structured counters/status; display those fields as unavailable. Browsing logs remains CPU-only and never launches experiment execution.

**Exports and repeatable exploration**

Export the exact filtered table as CSV/Parquet, selected charts as standalone interactive HTML, and view state as JSON. Include run/artifact IDs, filters, graph/scoring definitions and all analysis parameters. A standalone chart is not a standalone copy of the full dashboard application. Optional static SVG/PNG export can be added through a separately declared dependency.

Support URL state for small selectors plus a saved JSON view for complete reproduction. Refresh should validate saved IDs against the catalog and explain unavailable runs or checkpoints. Add an explicit refresh action for completed new result shards; live polling is optional and must not read half-written artifacts.

**Implementation and acceptance criteria**

Track progress through T09–T12 in the main plan; update each item immediately upon completion. The metrics MVP can start with legacy artifacts before the expanded sweeps are available.

1. T09: implement the artifact catalog/adapter and overview. Load existing metrics, graph count/density trajectories, and behavioral scores; link checkpoint selection to a summary. Validate against source rows without GPU/model access. Integrate shared process logging and read available status/log records; clearly label legacy limitations.
2. T10: add component rankings/trajectories, distributions, role dynamics, compatible-run comparisons, and semantic/coverage labels. Check that every linked view uses the same filters and aggregation definitions.
3. T11: add the bounded network explorer and checkpoint difference view using identified endpoints. Preserve layout and component selection; display truncation counts and keep full-graph metrics unchanged by display filters.
4. T12: add exports/saved views, cache invalidation, refresh behavior, browser interaction coverage and measured latency/memory checks. Document CPU-only setup and the implemented launch command.

Use deterministic fixtures with multiple runs, missing checkpoints, initialization, alternate thresholds, isolated nodes, legacy/updated semantics and absent optional metrics. Verify filter correctness, no double-counted baselines, directed component identity, no fabricated uncertainty, state round trips, and agreement with shared analysis outputs. Include a browser smoke test for checkpoint click, component selection and comparison, and a test that fails if browsing attempts to load a model or invoke experiment execution.
