**Implemented scientific versions**

Experiments are run afresh with the versioned definitions below; agreement with historical results is not an acceptance criterion. Every analysis declares its graph version, resolution, inventory source, architecture ordering, threshold and distance convention. Graph analysis never changes the all-position component/head intervention scopes.

**Graphs: `influence-v2`**

Edge `weight`/`strength` is `1 - cosine_similarity`, transformed once. An edge is retained when **similarity < threshold**, strictly; NaN observations are missing and never become edges. All selected inventory nodes are graph nodes, including isolates. Shortest-path metrics use `distance = 1 / (strength + epsilon)` (`inverse-strength-paths-v2`). Incoming closeness uses the graph, outgoing closeness its reversal, betweenness uses `distance`, and strengths/PageRank use `weight`. Mixed whole/head populations overlap and are rejected. A component view keeps whole components; a head view replaces eligible attention components with individual heads and keeps non-head components.

Ordering is architecture reachability for the supported sites. Within parallel Pythia blocks, q→rotary-q→z, k→rotary-k→z and v→z are admissible, and attention and MLP are independent branches. Head paths within one layer require matching heads. In sequential blocks, attention also precedes post-MLP, and earlier layers precede later ones. A new adapter must define and validate additional inventories. Graph metadata records full selected and active-node counts, admissible pairs, density relative to them, `measured_admissible_pairs`, `measurement_coverage` and `measurement_status`, so partial intervention subsets are labeled partial.

**Inventory and activation metrics**

`tl-existing-sites-v2` records the embedding and existing q/k/v/rotary-q/rotary-k/z/post-MLP hooks, optional attention-head indices and the tensor axes each hook owns. Query and key/value head counts are recorded separately, and grouped-query architectures are rejected until validated. Pythia weights load with recorded unprocessed normalization, writing and unembedding conventions.

`masked-cosine-relative-l2-v1` averages feature-axis cosine over valid positions (and over eligible heads for whole components), and also records relative Frobenius change and zero-norm fractions. Relative L2 is null, with an explicit status, when the clean norm is at most 1e-12. Behavioral effects and activation changes are stored separately.

**Tasks and scoring**

- `induction-repeat-target-logprob-v1` (`random-token-induction-v1` data): the prompt is `BOS A A[:-1]` with seeded distinct non-special tokens, scored by FP32 full-vocabulary log-probability of the withheld token, in nats. This is a development probe.
- `tigges-2024-{ioi,gendered_pronoun,greater_than,sva}-v1`: built from the checksummed official supplement and retokenized per model; see the [task protocols](tigges-2024-task-protocols.md).
  - IOI and pronouns use a logit difference.
  - Greater-than and SVA use full-vocabulary probability group sums.
  - Clean labels are anchored for corrupt and patched runs.
  - Greater-than candidates are standalone two-digit suffix tokens after the century token, with per-example canonical-continuation counts.

Sample, calibration, candidate, position, mask and donor manifests are saved in `dataset.json`.

**Replacements**

Zero, calibrated mean (`position-aligned-example-mean-v1` by default, `global-valid-position-mean-v1` as a labeled alternative) and donor replacement:
- matched counterfactual, noising or denoising;
- seeded non-self equal-length resampling.

Calibration prompts never equal any evaluation clean or corrupt prompt, and per-position coverage is recorded. Recovery `(patched - corrupt) / (clean - corrupt)` uses dataset means, with per-example values as diagnostics; it is undefined when the gap is at most `recovery_epsilon`.

**Storage and resume**

`fresh-reference-v2` commits one deterministic zip shard per example chunk (`records.json` plus a float32 `[record, observation, field]` array). The manifest holds the work plan, source/config/model/dataset identities and checksums. Resume validates all of them under a writer lock. Example batches align with shards, so resumed outputs are bit-identical. CPU analysis and export verify completed manifests and hashes before use.

**Archived results**

Historical outputs (fixed-passage token-2205 scores, legacy matrices, notebook pickles and figures) are kept under the ignored `artifacts/legacy-results/`. Their generating code is in Git history at commit `850411f`. The dashboard's `import-legacy` labels them `legacy-notebook-v1` / `legacy-induction-token2205-v1` and reports unavailable fields. Earlier diagnostics found that those outputs are not reproducible from the current recovered checkpoint revision. That is recorded for traceability and does not block new work.

**Environment note**

An unrelated unlocked TorchVision in the shared environment is ABI-incompatible with locked PyTorch. `resources.disable_optional_vision=true` (or `--disable-optional-vision`) disables Transformers' optional vision imports for these text-only runs and is recorded in manifests.
