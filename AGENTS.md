**Guidance for contributors and coding agents**

This file applies throughout the repository. Read [README.md](README.md) for the current workflow, [the expansion plan](docs/experiment-expansion-plan.md) before restructuring experiments, and the introduction/research questions and discussion in [paper/main.tex](paper/main.tex) for the research objectives. The plan describes proposed work; implement the scope requested in the active task.

**TODO updates and session continuity**

Use the implementation TODO checklist and session handoff notes in [the expansion plan](docs/experiment-expansion-plan.md) as the persistent source of progress. Read both at the start of each session and reconcile them with the working tree before continuing.

- Whenever a TODO item is completed, immediately mark it `[x]` and record a brief completion note with the changed files/artifacts and relevant validation results. Update the list before moving to the next item; do not defer updates until the end of the session.
- Mark items complete only when their acceptance criteria are met. Leave partial work unchecked, record what is finished and what remains, and add smaller checklist items when needed to track incremental progress.
- Keep stable TODO IDs and preserve completed entries. Add newly discovered work to the checklist within the user's requested scope.
- After each completed item, refresh the handoff notes with the current work, remaining steps, blockers, decisions, and next TODO ID. For experiments, include output paths, resolved configs, checkpoint/shard progress, and the command needed to resume.
- Refresh partial-work notes before ending a session or approaching a context/token limit. Record changes and verification in the repository, so another session can resume without relying on chat history. A fresh session should verify saved state and continue from the first applicable unfinished item.

**Required Python environment and dependency management**

Use only the existing pyenv environment `nnets-transformer` at `/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer` (Python 3.11.1) for this project. Manage dependencies with Poetry and the project `pyproject.toml`/lockfile. Reuse this environment for implementation, tests, notebooks, experiments, and the dashboard.

Before invoking Python or Poetry in a new shell/tool process:

```bash
export PYENV_VERSION=nnets-transformer
export VIRTUAL_ENV=/data/shared/pyenv/versions/3.11.1/envs/nnets-transformer
export PATH="$VIRTUAL_ENV/bin:$PATH"
export POETRY_VIRTUALENVS_CREATE=false
```

These exports do not persist between separate tool processes. The environment's bundled activation script contains a stale `VIRTUAL_ENV` path; use the explicit exports above rather than sourcing it or editing the shared environment. Verify that `poetry config virtualenvs.create` reports `false` and that `poetry run python` has `sys.prefix` resolving to the required path before any dependency installation/change. The repository's `.python-version` and project-local `poetry.toml` record this setup; keep virtualenv creation disabled. Do not create a `.venv`, a Poetry-managed environment, or another pyenv/Conda environment, and do not substitute system Python. If the specified environment is unavailable, report that concrete blocker rather than creating a replacement. Poetry itself may run from its existing external installation.

**Research objective and current state**

Preserve the objectives in `paper/main.tex`: (RQ1) determine whether global component-influence graph topology changes in structured regimes aligned with task performance during training; (RQ2) determine whether graph-metric prominence stays with the same components or is reassigned across checkpoints. Upper-tail graph positions are descriptive structural roles, not independently validated functional mechanisms. Training progression is the primary comparison; model scale, training seed, task, and intervention method are experimental axes.

The user clarified on 16 October 2026 that the expanded experiments should be run afresh with improved methods. Matching previous matrices, behavioral scores, figures, numerical values, or phase boundaries is not an objective or an acceptance criterion. Do not investigate old-result mismatches, recover old environments for numerical matching, or tune new methods to reproduce historical findings. Existing mismatch diagnostics are archived evidence, not blockers or the next task.

Generate new dataset/pairing manifests, checkpoint-specific reference activations, measurements, graphs, analyses, and figures under new run IDs/output locations. Validate the new pipeline using deterministic scientific cases, source-model clean-logit/intervention checks, serial/optimized agreement, precision tolerances, and resume equivalence. Fresh execution and validation must work without old result files or successful historical comparisons. Preserve legacy artifacts and explicit graph/scoring versions for traceability and optional browsing.

The current implementation is the `nnets_transformer` package: independent model/task/replacement/metric/execution modules, typed composed YAMLs, serial and optimized execution with chunked resumable shards, checkpoint-grouped sweeps, the four paper tasks plus induction, zero/mean/donor replacements, CPU analysis/Parquet export, a CPU-only dashboard and shared attempt logging. The legacy root scripts, notebook and historical reference were removed (kept in Git history at `850411f`); their outputs are archived under ignored `artifacts/legacy-results/`. Open work: running pilot sweeps (T05b/T05c), OLMo (T06, blocked on adapter support) and optional T07/T08. Follow the persistent checklist for implemented details.

**Architecture and extension rules**

- Keep model instrumentation, task preparation, replacement policies, intervention scopes, metrics, execution, and analysis independently extensible.
- Model adapters own hook names, component inventories, tensor axes, architecture ordering, and capability checks. Distinguish query heads from key/value heads.
- Tasks own examples, candidate tokenization, semantic token roles, paired inputs, and scoring specifications. Avoid fixed vocabulary IDs or task-specific logic in the runner.
- Keep intervention scopes to the existing whole-component and individual attention-head choices, using the current component inventory and applying replacements across all sequence positions. Represent an optional head index within the component selection. Separate replacement values (zero, mean, donor) and estimators (exact or gradient approximation) from these scopes. Task scoring positions and donor alignment belong to data/scoring preparation.
- Introduce composed, typed YAML configurations during the refactor. Validate incompatible options early and save resolved configurations.
- Keep notebooks focused on exploration and presentation; shared computation belongs in importable modules. Use one implementation of graph construction.
- Keep entry points and artifact readers usable, or document an explicit replacement when removing one.

**Interactive dashboard**

Follow [the dashboard plan](docs/dashboard-plan.md) and TODOs T09–T12. Keep dashboard dependencies optional and data access CPU-only, consuming saved legacy/normalized artifacts through an adapter. Share scientific analysis functions; dashboard browsing must not load models or invoke experiment execution. Preserve graph/scoring versions, metric units, missing-data status, and the existing component/head intervention scope. Display limits for graph exploration must not change full-graph metric values. Validate linked filters, checkpoint selection, saved-view/export provenance, and CPU/browser behavior before marking dashboard milestones complete.

**Scientific correctness and reproducibility**

- Preserve historical outputs as an archive; agreement with them is not required. Version changes to graph semantics, metrics, component inventories, or scoring rather than silently treating corrected results as equivalent to old results.
- The corrected graph definition `influence-v2` (single strength transform, separate positive path distance, complete inventory including isolates, architecture ordering) replaced the legacy conventions; see `docs/scientific-definitions.md`. Version any further change rather than editing it in place.
- Separate behavioral effects from activation changes. Component/head ablation measures propagated influence; graph edges summarize downstream observations and do not define intervention scopes.
- Follow [the Tigges et al. task/scoring specification](docs/tigges-2024-task-protocols.md) for the paper preset. Use logit differences for IOI/pronouns and group probability sums from full-vocabulary softmax for greater-than/SVA. Preserve clean answer definitions across paired runs, record sample/candidate manifests, and distinguish behavioral scores from optional head diagnostics. Keep the existing component/head intervention scopes.
- Validate prediction positions, leading-space tokenization, candidate lengths, masks, and donor alignment for each tokenizer.
- Keep calibration and evaluation data separate. Record donor identities and seeds, prevent self-donation in resampling, and recompute reference activations per checkpoint.
- Hold evaluation data and pairing policies fixed across checkpoints. Report undefined normalized recovery when the clean/corrupt score gap is negligible.
- Record immutable model/tokenizer revisions, checkpoint stage/step/tokens, dataset identity, seeds, code/environment versions, precision, and analysis conventions.
- Compare families using training tokens and competence alongside raw steps. Do not assume shared phase boundaries or corresponding head identities.

**Performance and execution**

Target one H100 NVL with approximately 94 GiB of memory. Select the GPU through the environment; keep machine-specific device indices out of reusable code.

- Reuse clean forwards/caches per checkpoint and example batch; load each checkpoint once for its selected work.
- Cache only requested activations, vectorize head/position reductions, and avoid `.item()` or CPU transfers inside scoring hooks.
- Batch independent interventions in separate example slices. Applying several interventions to one example changes the experiment.
- Use evaluation/inference mode for forward-only methods. Keep gradient methods on a separate execution path.
- Benchmark BF16 execution with FP32 reductions against a reference before adopting it. Treat quantization as an experimental condition.
- Save resumable shards and derive thresholds/figures from saved measurements. Bound activation, checkpoint, and result storage.
- Verify that fused attention or compilation preserves required hook behavior. Report measured performance rather than assumed speedups.

**Persistent logs and visible progress**

Follow [the logging specification](docs/logging-plan.md), TODO T13. Persist logs for every experiment, analysis/export job, dashboard process, and substantive setup/test/profile command. Print log paths at startup; keep readable console/file output, structured events, and atomic live status under `logs/<run-or-job-id>/<attempt-id>/`. Record lifecycle stages, progress counters, heartbeat versus actual progress, warnings, tracebacks, exit status, saved artifacts, and resume instructions. Preserve prior attempt logs and link them from run manifests and handoff notes.

Every entry point uses the shared `Job` logger; wire it into any new entry point before heavy setup, and wrap other substantive commands with `python -m nnets_transformer.observability`. Keep logging outside GPU hooks/hot loops and avoid tensor dumps or synchronization just for progress. Validate failure/interrupt/resume and concurrent logs, not only successful output. Every completed TODO should reference its validation log paths where applicable.

**Version-control artifact policy**

Keep Git focused on reproducible source/configuration: track source, cleared-output notebooks (if any), documentation, small curated text fixtures, `pyproject.toml`, `poetry.lock`, `poetry.toml`, and `.python-version`. Put generated runs/results/logs, downloads, weights, activation dumps and plots in the ignored artifact directories. Use root-anchored rules for generated `analysis/`, `models/` and similar directories so package source remains visible; do not blanket-ignore CSV/JSON/JSONL source fixtures. Verify representative inclusion/exclusion paths with `git check-ignore` when updating rules.

Clear notebook outputs/execution counts before committing while preserving code/markdown and any needed local result archive. When removing generated files from Git tracking, preserve their local contents. Logs remain available locally even though they are ignored. Record concise verification evidence and log paths in the tracked TODO/handoff notes.

**Working practices and validation**

Use the required `nnets-transformer` pyenv environment through Poetry and the existing lockfile for reproducible fresh runs; avoid incidental dependency upgrades. The deterministic suite is `poetry run python -u -m unittest discover -s tests -v`; no lint command is configured. Add meaningful validation when introducing scientific or execution logic. Historical equality is not a test oracle for the improved pipeline.

For implementation changes, start with small deterministic cases. Check no-op identity, replacement tensors, masks, hook cleanup, serial/batched equivalence, precision tolerances, and resume behavior as applicable. Validate a new model adapter against source-model clean logits and intervention semantics.

`command=sweep` presets such as `+experiment=pilot` launch real multi-checkpoint GPU sweeps; they are not smoke tests (use `command=inspect` to preview a preset). Do not edit hashed source modules (`records.py`, `config.py`, `models/`, `tasks/`, `interventions/`, `metrics/`, `execution/`) while a sweep runs: each cell records their hashes when it starts. Use the requested experiment scope and isolated output locations. Documentation-only changes need link/command inspection and whitespace checks, not model downloads or GPU experiments.

Keep archived data, generated figures, paper sources, and unrelated local edits intact. Update the README when implemented commands, dependencies, output formats, or capabilities change. Report what changed, what was checked, and any remaining limitations.
