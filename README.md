# nnets-transformer

How does a Transformer's component influence network evolve during training?

This research project intervenes on internal Transformer activations, measures downstream changes, constructs directed influence graphs, and follows graph topology and component roles across training checkpoints. The fresh runner supports the **Pythia family** (14M–2.8B configured, plus deduplicated and training-seed variants), **exact zero, calibrated-mean and donor activation replacement** (noising and matched denoising), a **seeded induction probe** and the four **Tigges et al. (2024) paper tasks** (IOI, gendered pronouns, greater-than, SVA), with checkpoint-grouped sweeps, CPU analysis/export and a CPU-only dashboard.

The [experiment expansion plan](docs/experiment-expansion-plan.md) describes the proposed modular package, experiment YAMLs, optimization work, additional models, replacement methods, and inference-only NLP tasks. [AGENTS.md](AGENTS.md) provides guidance for contributors and coding agents. The [interactive dashboard plan](docs/dashboard-plan.md) specifies the implemented linked training-metric plots, component drill-down, run comparisons and network exploration from saved results.

The expanded work will run experiments afresh with improved methods while retaining the paper's questions about structural training regimes and component-role stability. Matching historical results is not required; old artifacts are preserved for traceability. See [the active scope and handoff](docs/experiment-expansion-plan.md).

**Current implementation**

| File or directory | Purpose |
|---|---|
| [src/nnets_transformer/](src/nnets_transformer/) | Typed config/CLI, model adapters, tasks, replacements, serial and optimized execution, sweeps, CPU analysis/export, dashboard and attempt logging |
| [configs/](configs/) and [tests/](tests/) | Composed YAMLs and deterministic scientific/execution validation |
| [scripts/](scripts/) | Source-model clean-logit check and execution/logging profilers |
| [pyproject.toml](pyproject.toml) | Poetry dependencies and Python requirement |
| [docs/experiment-expansion-plan.md](docs/experiment-expansion-plan.md) | Design, TODO checklist with completion evidence, and session handoff |

The installable package provides independent adapter/task/replacement/metric/execution modules, composed typed YAMLs, a serial reference and an optimized batched backend validated against it, resumable chunked shards, checkpoint-grouped sweeps, verified CPU analysis/Parquet export, a dashboard and YAML-configured attempt logging. Still open: running the pilot sweeps (T05b/T05c), OLMo 2 (T06, blocked: the pinned TransformerLens has no OLMo support) and the optional Appendix D/E diagnostics (T07/T08). Intervention scopes stay with the existing choices: a whole component output or one attention head, applied across all sequence positions.

**Setup**

Use the existing pyenv environment **`nnets-transformer`**, located at:

```text
/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer
```

Activate it in every new shell before using Poetry:

```bash
export PYENV_VERSION=nnets-transformer
export VIRTUAL_ENV=/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer
export PATH="$VIRTUAL_ENV/bin:$PATH"
export POETRY_VIRTUALENVS_CREATE=false
```

The repository's `.python-version` selects that pyenv name. Project-local [poetry.toml](poetry.toml) disables virtualenv creation and in-project environments. The explicit `VIRTUAL_ENV`/`PATH` settings select the existing environment for Poetry, including in noninteractive sessions; environment settings do not persist between separate shell/tool processes. The environment's bundled activation script currently contains a stale path, so use the exports above rather than sourcing that script. Poetry itself can remain installed outside this environment.

Verify the setting and the interpreter before installing or changing dependencies:

```bash
poetry config virtualenvs.create
poetry run python -c 'import sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path("/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer").resolve(), sys.prefix; print(sys.executable); print(sys.prefix)'
```

The first command must report `false`; the second must identify the specified pyenv environment. Manage dependencies with Poetry in that activated environment:

```bash
poetry install
```

This installs the source package and the `nnets-transformer` CLI. Use `poetry add`, `poetry remove`, and the project lockfile for dependency changes. Reuse this same environment for experiments, tests, notebooks, and the dashboard; do not create another virtualenv. Avoid incidental lockfile refreshes during implementation or fresh experiments. TransformerLens is pinned to baseline Git commit `50ee38b9a20695e90b8ce6a4aa20fd509c8175d9`. Hydra/OmegaConf were added for configuration; all pre-existing locked versions/revisions were retained. The dashboard's dependencies are optional Poetry groups (`dashboard`: Dash, dash-cytoscape; `dashboard-test`: Selenium), installed with `poetry install --with dashboard` (add `,dashboard-test` for the browser test); adding them changed no existing locked version. On this machine, `POETRY_SYSTEM_GIT_CLIENT=true poetry install` avoids a Dulwich chmod failure in the pre-existing shared source checkout.

Check PyTorch without downloading a model:

```bash
poetry run python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

The main script selects CUDA when available and otherwise uses the CPU. GPU execution needs a compatible PyTorch/CUDA installation. The expansion plan targets one H100 NVL with approximately 94 GiB of memory.

**Running experiments**

Every command below runs from the repository root after the environment exports above. Select the GPU with `CUDA_VISIBLE_DEVICES`; nothing in the YAMLs names a device index. Each command prints its log directory (`logs/JOB_ID/ATTEMPT_ID/`, with `console.log`, `events.jsonl`, live `status.json` and `invocation.json`) and refuses to overwrite an existing output directory unless `execution.resume=true`.

1. Check the installation without model downloads:

   ```bash
   poetry run -- python -m nnets_transformer.observability --job-id validation -- \
     python -u -m unittest discover -s tests -v
   poetry run nnets-transformer --job-id inspect +experiment=pilot command=inspect   # print the resolved pilot YAML only
   ```

2. Bounded end-to-end check on the GPU (about a minute; downloads two Pythia-14M checkpoints):

   ```bash
   CUDA_VISIBLE_DEVICES=0 poetry run nnets-transformer --job-id sweep-smoke \
     +experiment=sweep_smoke resources=h100 resources.disable_optional_vision=true \
     sweep.output_root=runs/sweep-smoke-new
   ```

3. Pilot sweeps (T05): induction plus IOI/pronouns/greater-than/SVA at paper sizes (70/70/200/200; induction 128), zero/mean/matched-donor replacements, every whole component, 12 checkpoints from `step0` to `step143000`. Each checkpoint is downloaded and loaded once for all of its task × replacement cells:

   ```bash
   CUDA_VISIBLE_DEVICES=0 poetry run nnets-transformer --job-id pilot-14m +experiment=pilot model=pythia_14m
   CUDA_VISIBLE_DEVICES=0 poetry run nnets-transformer --job-id pilot-160m +experiment=pilot model=pythia_160m \
     sweep.evict_checkpoint_cache=true
   CUDA_VISIBLE_DEVICES=0 poetry run nnets-transformer --job-id pilot-1b +experiment=pilot model=pythia_1b \
     sweep.evict_checkpoint_cache=true
   ```

   Outputs go to `runs/pilot/<model>/<checkpoint>/<task>/<replacement>/`, with the sweep manifest at `runs/pilot/<model>/sweep.json`. Measured on the H100 NVL, one paper-sized SVA/zero cell takes 14.5 s on Pythia-14M and 72 s on Pythia-1B (plus about 3.5 minutes for each new 1B checkpoint download), using 16 MB and 27 MB of disk. From these measurements, expect roughly 30 minutes for the 14M pilot, about 1 hour for 160M and about 3 hours for 1B (estimates, not measured sweeps). `sweep.evict_checkpoint_cache=true` deletes each downloaded checkpoint after use. Matched donors are undefined for induction, so that cell is skipped and recorded in `sweep.json`.

   Useful overrides:
   - `sweep.examples=64` gives a labeled development subset.
   - `sweep.replacements=[zero,mean,donor,donor_denoising,donor_resample,mean_global]` adds clean→corrupt recovery, seeded resampling and global means.
   - `scope=head` gives head-resolution runs (much larger: 8–16× more interventions).
   - `sweep.checkpoints=[step1000,step8000]` narrows the checkpoints.
   - `model=pythia_160m_seed1` runs a training-seed replicate.
   - `resources.dtype=bfloat16` is an opt-in precision condition; FP32 is the default.

   **Everything (all main Pythia models × all 154 checkpoints × 5 tasks × 6 replacements):** run `scripts/run_everything.sh` in tmux, and preview it first with `DRY_RUN=1 scripts/run_everything.sh`. It processes models smallest first and writes to `runs/full/`. After each cell it compacts the results (`sweep.compact`). Compaction keeps:
   - the similarity matrix;
   - all per-example behavioral rows;
   - the manifests;
   - per-example cosines as float16.

   It deletes the bulky float32 activation shards (relative L2 and zero-norm fields), so the whole grid needs about 115 GB instead of terabytes. A compacted cell is final: it can be exported and analysed, but not resumed. The script skips models that would not fit above `MIN_FREE_GB` (default 100), and pauses resumably if disk runs low. Ctrl+C stops it; rerunning the same command continues. This is a months-long GPU job, so expect to stop and resume it.

4. Resume an interrupted sweep by repeating its command with `execution.resume=true`. Completed cells are skipped without loading their checkpoint, and partial cells resume from verified shards. The exact command is also saved as `resume_command` in the attempt's `status.json`. Changed code or scientific settings invalidate resume by design, so use a new `sweep.output_root` for a new condition.

5. Export a trajectory (several checkpoints of one task/replacement) to normalized Parquet tables and a standalone HTML report, on CPU:

   ```bash
   poetry run nnets-transformer --job-id export-pilot-14m-ioi-mean +experiment=export \
     "export.inputs=[$(ls -d runs/pilot/pythia-14m/step*/ioi/mean | paste -sd, -)]" \
     output_directory=exports/pilot-14m-ioi-mean
   ```

   `export.inputs` can list several conditions; checkpoints of one condition form one trajectory. The per-example activation table is large and off by default (`export.include_activations=true` adds it for uncompacted runs). Thresholds default to 0.1/0.3/0.5/0.7, and upper-tail roles to the 0.95 quantile. Analysing a single run as a graph remains `+experiment=analysis input_matrix=RUN/similarities.csv output_directory=...`.

6. Browse the exports in the dashboard (CPU-only; reads `exports/` as set in [configs/dashboard.yaml](configs/dashboard.yaml)):

   ```bash
   poetry install --with dashboard          # once
   poetry run python -m nnets_transformer.dashboard import-legacy --output exports/legacy-induction   # optional: archived paper results
   poetry run python -m nnets_transformer.dashboard serve      # http://127.0.0.1:8050/
   ```

   From your workstation, use `ssh -L 8050:127.0.0.1:8050 SERVER`. The server binds to localhost unless `allow_remote: true` is set. The tabs are:
   - **Overview:** linked counts/density/behavior; click a point or use previous/next to select a checkpoint.
   - **Components:** rankings, pinned trajectories and a heatmap.
   - **Distributions.**
   - **Role dynamics:** upper-tail membership, turnover and Jaccard.
   - **Compare runs.**
   - **Network:** bounded display with a truncation label and a checkpoint difference view.
   - **Run status / logs:** live sweep progress.

   The buttons export the view JSON, the overview chart HTML, the ranking CSV and the checkpoint Parquet, all with provenance. **Load saved view** restores a view, and the share box gives a URL for small selections. **Refresh artifacts** picks up newly completed exports. Browsing never loads models or launches experiments.

Single runs (one checkpoint, task and replacement) use `command=run`, e.g. `+experiment=paper task=sva replacement=mean model=pythia_160m model.checkpoint=step8000 model.revision=resolve model.tokenizer_revision=resolve output_directory=runs/sva-160m-step8000`. `+experiment=paper_smoke` and `+experiment=induction[_smoke]` are bounded development presets. The branch is resolved to an immutable commit before anything is saved.

**What the runner measures**

Tasks own examples, prediction positions and answer groups. Induction prompts are `BOS A A[:-1]` with a seeded distinct-token pattern, scored by target log probability in nats; it is a development probe. The paper tasks are prepared from the checksummed official supplement (downloaded once to ignored `artifacts/tigges-2024-source/`) and retokenized per model:
- **IOI** uses the source generator with its `ABB->ABA, BAB->BAA` corruption. **Pronouns** use the released pairs. Both score a logit difference, with clean labels kept for corrupt and patched runs.
- **Greater-than** uses the full-vocabulary probability of suffixes above the clean year minus those at or below it.
- **SVA** uses the probability mass of agreeing minus disagreeing verb forms. It reports both number directions, and its candidate groups are unequal in size.

Replacements are separate from scopes:
- **Zero.**
- **Mean:** a per-position mean over a calibration split that never shares a prompt with evaluation, with per-position coverage recorded. `mean_global` is a separately labeled condition.
- **Donor:**
  - matched counterfactual, in noising or denoising direction;
  - seeded resampling of an equal-length, never-self donor.

Every record keeps raw clean/corrupt/patched scores. Recovery `(patched - corrupt) / (clean - corrupt)` is reported as undefined when the gap is negligible, and values outside [0, 1] are kept. Activation propagation is the masked cosine similarity, relative L2 change and zero-norm status at every observed component.

Runs write `manifest.json`, `resolved.yaml`, `dataset.json` (sample, calibration and donor manifests), `inventory.json`, chunked checksummed shards `shards/*.zip` (schema `fresh-reference-v2`: JSON behavior plus float32 activation arrays), `similarities.csv` (mean per-example cosine) and `behavior.json`. Resume verifies code, config, model and dataset identities, input and shard hashes, and the work plan. Example batches align with shards, so resumed outputs are bit-identical. The optimized backend (`execution.backend=optimized`, the default in sweep presets) reuses clean and corrupt caches per example batch. It runs independent interventions in separate batch slices and agrees with the serial reference within FP32 tolerance.

To validate a model adapter against the source Hugging Face model's clean logits, run `poetry run python scripts/check_clean_adapter.py --disable-optional-vision --model EleutherAI/pythia-14m --revision SHA --output runs/adapter-check.json`. The checkpoint must already be in `models/hf-cache`. See [scientific definitions](docs/scientific-definitions.md) for the versioned graph, metric and storage conventions.

The test suite needs no model downloads. It uses the locally cached Pythia-14M tokenizer and the supplement for the paper-task checks, and skips them if those are absent. The dashboard browser check needs `--with dashboard-test` and Chrome. Fixture artifacts are kept under ignored `artifacts/`.

**Process logs**

Every command above writes timestamped `console.log`, `events.jsonl`, an atomic live `status.json` and `invocation.json` under `logs/JOB_ID/ATTEMPT_ID/`. These record warnings, exceptions, exit codes, progress counters and resume commands, and distinguish last actual progress from heartbeat. Each invocation gets a new attempt directory, and logs never authorize reusing scientific artifacts. Follow a running job with `tail -F logs/JOB_ID/ATTEMPT_ID/console.log`, or use the dashboard's **Run status / logs** tab. Wrap any other command for the same structured capture:

```bash
poetry run -- python -m nnets_transformer.observability --job-id validation -- \
  python -u -m unittest discover -s tests -v
```

`--artifact-root PATH`, `--resume-command TEXT` and `--previous-attempt PATH` record job context. `--log-root PATH` and `--heartbeat-seconds SECONDS` control log placement and liveness updates. YAML `logging.*` settings configure levels, timezone, intervals, rotation and telemetry.

**Interpreting results**

Each similarity matrix has intervened sources as rows and observed targets as columns. Influence strength is `1 - cosine_similarity`. Graphs (`influence-v2`) keep admissible source/target pairs strictly below a similarity threshold, keep isolated inventory nodes, order components by architecture reachability, and use the separate positive distance `1 / (strength + epsilon)` for shortest paths. The influence graph measures propagated sensitivity to component or head replacement; it does not isolate individual causal paths. Upper-tail graph positions are descriptive structural roles, not validated functional mechanisms.

**Archived results**

The original fixed-passage workflow (`main.py`, `graphs.py`, `cn_figures.py`, `graph_analysis.ipynb` and the historical reference tooling) was removed from the working tree on 16 October 2026; it remains in Git history at commit `850411f`. Its local outputs are under the ignored `artifacts/legacy-results/`: `ablation_data/`, the notebook pickles, `induction_data.json`, `loss_data.json`, figures, the `analysis*` directories and early reference runs. `python -m nnets_transformer.dashboard import-legacy` makes them browsable, with their legacy semantics labeled.

**What belongs in Git**

Keep source code, experiment/configuration files, documentation, small curated text inputs/fixtures, `pyproject.toml`, `poetry.lock`, `poetry.toml`, and `.python-version` eligible for version control. The lockfile preserves the dependency resolution used to reproduce experiments; do not regenerate it incidentally.

[.gitignore](.gitignore) excludes generated runs/results, logs and monitoring state, analysis exports/figures, downloaded datasets/models, caches, model weights, serialized tensors/tables, archives, and local environment/tool state. Generated directory rules are anchored to the repository root so future source packages such as `src/nnets_transformer/analysis/` and `src/nnets_transformer/models/` stay visible. Curated CSV/JSON/JSONL fixtures under source/configuration/test directories remain eligible.

Notebooks, if added, should be committed with outputs cleared. Keep downloaded inputs and generated exports under the designated ignored directories and record reproducible generators, seeds, versions and source hashes in tracked configuration/documentation.

**Planned expansion**

| Axis | Initial direction |
|---|---|
| Configuration | Composed YAMLs with validated model, task, replacement, scope, metric, and resource settings |
| Intervention scopes | Existing whole-component and individual attention-head interventions, applied across all sequence positions |
| Models | Pythia scaling and training-seed replication, followed by OLMo 2 checkpoint trajectories |
| Replacements | Zero, calibrated mean, unrelated dataset donors, and matched counterfactual donors |
| Tasks | Induction plus the four Tigges et al. tasks: IOI, gendered pronouns, greater-than, and SVA; optional Appendix E benchmarks and later recall tasks |
| Performance | Baseline reuse, selective caching, vectorized metrics, independent intervention batching, and validated BF16 execution |
| Artifacts | Immutable revision manifests, per-example measurements, resumable shards, and analysis independent of inference |
| Logging | Persistent console/event logs, live status and progress, error/resume history, and a dashboard log panel |
| Interactive dashboard | CPU-only exploration of saved graph metrics across checkpoints, component/role views, run comparisons, and exportable selections |

These tasks can be scored with forward inference without training or autoregressive generation. Patching still requires baseline, donor/reference, and intervened forwards. Gradient attribution patching requires a separate backward-capable execution path.

The pilot preset (`+experiment=pilot`) uses Pythia 14M, 160M, and 1B at 12 checkpoints. The paper-sized task preset uses 70 IOI, 70 pronoun, 200 greater-than, and 200 SVA examples; smaller development subsets are labeled separately. The [paper task/scoring specification](docs/tigges-2024-task-protocols.md) records the formulas and appendix details: IOI/pronouns use logit differences, while greater-than/SVA sum full-vocabulary probabilities over answer groups. It also describes optional head diagnostics and SciQ/PIQA/WinoGrande/ARC Easy evaluations. The [full plan](docs/experiment-expansion-plan.md) gives module boundaries, an illustrative YAML, method definitions, optimization priorities, and validation criteria. So far only bounded validation runs and timing cells have been executed; the pilot sweeps themselves have not been run.

The dashboard is implemented (launch commands above). See [dashboard milestones T09–T12](docs/experiment-expansion-plan.md) and the [detailed dashboard design](docs/dashboard-plan.md).
