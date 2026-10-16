**Task protocols from Tigges et al., NeurIPS 2024**

Source: [LLM Circuit Analyses Are Consistent Across Training and Scale](https://proceedings.neurips.cc/paper_files/paper/2024/file/47c7edadfee365b394b2a3bd416048da-Paper-Conference.pdf), by Curt Tigges, Michael Hanna, Qinan Yu, and Stella Biderman. Read Section 2.4 (printed pages 3–4), Appendices A–D (pages 22–25), and Appendix E (pages 25–26) together with the [official code supplement](https://proceedings.neurips.cc/paper_files/paper/2024/file/47c7edadfee365b394b2a3bd416048da-Supplemental-Conference.zip).

This document specifies additions to the [experiment plan](experiment-expansion-plan.md). These tasks and metrics are planned, not yet implemented. Interventions remain whole-component or individual attention-head replacements across all positions. Reading the paper's methods does not add edge/path interventions or CSPA to the project.

**Core task preset**

Keep induction as the project's existing task and add all four paper tasks to a `tigges_2024` experiment preset. Use the paper's reported sample sizes for that preset; smaller development samples and larger robustness samples get distinct configs and sample manifests.

| Task ID | Paper sample size | Clean/corrupt construction | Primary behavioral metric |
|---|---:|---|---|
| `ioi` | 70 | Mixed IOI templates; change the repeated subject name to reverse the preferred recipient | Correct-name minus distractor-name logit |
| `gendered_pronoun` | 70 | Paired name substitutions with opposite he/she targets; retain both target directions | Target-pronoun minus alternate-pronoun logit |
| `greater_than` | 200 | Year-range prompts; corrupt the starting year suffix to `01` | Sum of valid-year probabilities minus sum of invalid-year probabilities |
| `sva` | 200 | Matched singular/plural-subject prefixes; swap subject number while preserving the paired construction | Sum of agreeing-verb probabilities minus sum of disagreeing-verb probabilities |

The supplement confirms the primary reductions in `get_circuits_over_time.py`, `utils/metrics.py`, and `utils/data_utils.py`. In particular, `compute_probability_diff(mode="group_sum")` is used for greater-than and agreement: sum candidate probabilities within each group, then subtract. Do not replace this with mean group probability, a group logit margin, or softmax restricted to candidate tokens.

**Shared scoring contract**

Let `z_i[v]` be the logit for vocabulary token `v` at example `i`'s prediction position, and `p_i = softmax(z_i)` over the full vocabulary. Select the last valid prompt position using the attention mask and actual padding convention. Compute probabilities/reductions in FP32 and preserve the per-example score before averaging over examples. Aggregate batches by example count, including a shorter final batch.

For every paired intervention, preserve the clean example's target and contrast definitions when scoring clean, corrupt, and patched runs. For greater-than, that includes the original year boundary; for agreement, the original subject number. Relabeling corrupt baselines would invalidate the damage/recovery comparison.

Save `clean_score`, `corrupt_score`, `patched_score`, raw signed change `patched - clean`, and optional recovery `(patched - corrupt) / (clean - corrupt)`. The supplement also defines signed noising `(patched - clean) / (clean - corrupt)`, which is 0 for clean behavior and -1 at the corrupt baseline. State whether normalization uses dataset means or per-example gaps; the source's IOI helper uses mean scores. Keep undefined small-denominator cases explicit. The project's default remains raw effects plus dataset-level recovery, with per-example results retained.

**IOI**

Use the original IOI generator with mixed templates and a fixed dataset seed. The supplement's `generate_data_and_caches` uses seed 42, no prepended BOS, and `gen_flipped_prompts("ABB->ABA, BAB->BAA")`. Persist generated examples and prediction positions rather than regenerating at each checkpoint.

For correct indirect-object token `a_i` and subject distractor `b_i`, score `m_i = z_i[a_i] - z_i[b_i]`; report `mean_i(m_i)`. Pairwise accuracy `mean_i(1[m_i > 0])` and full-vocabulary top-1 accuracy are separate supplementary metrics. A tie fails the strict pairwise criterion.

**Gendered pronouns**

Use the released paired pronoun examples, map answer labels to he/she strings, and retokenize them for each model. For target `a_i` and alternate `b_i`, score `m_i = z_i[a_i] - z_i[b_i]`, averaged across examples. Preserve both target directions and report their counts; this probes the model's pronoun preferences under the dataset convention.

The source CSV stores Pythia-specific answer IDs, so do not reuse those IDs with another tokenizer. Validate exact continuation strings, including spaces, at the prompt boundary. Section 2.4 reports 70 examples, but `data/gender-pronoun.csv` contains 100 rows and `from_csv` shuffles without an explicit local random state. A reproducible paper-sized preset must select and record 70 stable row IDs with an explicit seed; it cannot claim to recover the authors' exact unknown subset. The supplement's generic task dispatchers do not consistently wire up every task, so CSV/helper existence alone is not a validated executable pipeline.

**Greater-than**

For a start-year suffix `y_i`, define `C_i = {y_i+1, ..., 99}` and `I_i = {00, ..., y_i}`. Equality is invalid. Score:

```text
m_i = sum_{u in C_i} p_i[token(u)] - sum_{u in I_i} p_i[token(u)]
```

Format candidates as two-digit suffix strings in the continuation context. The complement is within `00`–`99`, not the full vocabulary. Probability outside that candidate set still remains in the softmax denominator. Keep the original clean boundary when evaluating the `01` corruption.

The supplement's `YearDataset` is constructed from valid tokenized years in `[1100, 1800)`, balances suffixes by cycling through `02`–`98`, and uses noun-varied templates. Validate the century/suffix token boundary and require a one-token continuation for every candidate used by this preset. Do not silently take the first subtoken of a multi-token suffix as the source helper does. A tokenizer that cannot represent the exact protocol needs an explicitly labeled adaptation or an unsupported-task result.

**Subject–verb agreement**

Use the source's singular/plural sentence pairs from Newman et al.'s targeted syntactic evaluation data, preserving construction labels. The supplement pairs equal-length tokenized prompts, includes both number directions, and records the prediction position.

Construct disjoint singular and plural verb-token sets from `data/sva/combined_verb_list.csv`, filtering exact word-boundary forms for each tokenizer. The source uses `get_singular_and_plural(model, strict=False)` with its GPT-style token convention; generalized adapters must reproduce the intended word/token matches and persist resulting sets. Its filtering can produce unequal candidate-set sizes, which must be reported rather than normalized away.

For agreeing set `C_i` and disagreeing set `I_i`, score:

```text
m_i = sum_{v in C_i} p_i[v] - sum_{v in I_i} p_i[v]
```

Both sets contain many verb forms. An `are` versus `is` logit margin is a different, supplementary task variant. The paper-faithful main metric is group probability mass. The released verb table has 1,950 rows; the retained token sets depend on tokenizer and filtering policy.

**Appendix D: component diagnostics**

These are head-function scores, separate from task behavior and cosine influence strengths. The observation-only scores below can be added without changing intervention scopes. Cache attention patterns only for diagnostics that need them. Compute raw per-head scores at each checkpoint using fixed probes; full-model or influence-selected populations differ from the paper's EAP-IG circuit population and must be labeled accordingly.

| Diagnostic | Definition and implementation contract |
|---|---|
| Previous-token attention | With `A[q,k]` indexed destination/query then source/key, average `A[q,q-1]` over eligible causal positions and probe examples |
| Duplicate-token attention | For repeated segments of length `L`, average `A[q,q-L]` on the second copy |
| Induction/prefix matching | Average `A[q,q-L+1]` on valid second-copy query positions, attending to the successor of the earlier matching token |
| Copy score | From the first block's post-MLP residual at name positions, apply the selected head's value/output map and the specified final normalization/unembedding; count the original name in the top five vocabulary logits |
| Succession score | For ordered pairs such as numbers/days/months, apply the first MLP to selected embeddings, then a head's OV map and unembedding; measure how often the successor is the highest-scoring candidate |

Always verify destination/source axes, padding, BOS offsets, and eligible diagonal positions using a known attention pattern. The supplement uses swapped diagonal axes; copying its numerical offset into a different axis convention can reverse the measured direction. Its induction probe uses two repeated length-100 random-token segments with batch size 2; a more statistically stable probe is an explicitly labeled extension.

For copy score, the supplement's `compute_copy_score` evaluates IO, S1, and S2 name occurrences, divides successes by `3*N`, and returns percentages. Store a documented fraction or percentage. Its final-normalization assumptions need an adapter check: the appendix describes final layer normalization, while the helper comments assume it is folded into weights. Reproduce a declared loading/normalization convention rather than applying or omitting normalization implicitly.

For succession, compute selected embedding/candidate slices instead of a vocabulary-squared matrix. The appendix describes strict superiority to every alternative; `get_new_successor_head_scores_over_time.py` accepts ties with the maximum and compares within each sequence category. Record tie handling and candidate universe as separate metric versions rather than silently mixing definitions.

Reference-only appendix methods: CSPA reports `1 - D_CSPA / D_mean`, with each `D` the mean `KL(clean || ablated)` on OpenWebText. It preserves specific OV/QK suppression behavior and is not ordinary whole-head mean ablation. Appendix A's algorithmic scores divide the absolute effect of classified senders by that of all eligible senders, using Appendix B's path interventions. Neither procedure is part of the allowed component/head replacement setting. Document their definitions for interpretation; do not implement them as additional intervention scopes or silently call ordinary ablation a replication.

**Appendix E: additional behavioral benchmarks**

Add optional task entries for SciQ, PIQA, WinoGrande, and ARC Easy. The paper reuses Pythia training-run evaluations and reports multiple-choice accuracy based on the highest choice score. It does not specify enough prompt, few-shot, continuation, or normalization detail to reconstruct those evaluations from the PDF alone.

For an exact historical comparison, recover the original evaluation-harness version/config and per-task scoring before execution. Multi-token candidate likelihood can be scored with teacher forcing in batched forward inference, with no generation or fitting, but it is a separate contract from a single next-token task. If strict one-next-token execution is required, a validated single-token option-label prompt is a new adaptation and must be labeled as such. Keep these benchmarks disabled in the strict-next-token preset until that choice is explicit. Store per-choice scores and report `mean(argmax(choice_scores) == gold)` with a fixed tie policy; do not invent corruption pairs for these datasets without a separate task specification.

**Reproduction records and validation**

The official supplement inspected for this plan has SHA256 `89630b437cc49501597b9fcb9b2f994c7b52a97369e0153f4280c5cd107096c1`. The pronoun CSV hash is `376e5aa9ef1ca4932b88c90b627dac0bb87a2264bf428fc37bd0b26e9c765d07`; the verb table hash is `17488ddb7be1a23b835d6cf134cb3b6060366230e6df6d70cd8032e1171efad2`. Temporary downloaded files are not repository dependencies; future runs must record their own verified source manifests.

The circuit script uses 70/200/200 examples for IOI/greater-than/SVA, while some behavioral helpers request 1,000 examples. Make dataset size explicit. Seed both source sampling and shuffling and save sample identities. Preserve model preprocessing flags (`center_unembed`, `center_writing_weights`, `fold_ln`) because raw logits and head probes can depend on them.

Before implementing sweeps, validate fixed-logit examples for signs, sums, full-vocabulary normalization, the greater-than equality boundary, both SVA number directions, prediction positions, batch aggregation, and corrupt-baseline label anchoring. Validate head probes with known diagonal patterns and source-model reference computations. These validation checks remain implementation TODOs; no experiment has been run during this documentation task.
