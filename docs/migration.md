**Current entry points and historical compatibility**

`main.py` is a thin wrapper around the configured fresh CLI, with the same YAML overrides as `poetry run nnets-transformer`. Importing old helper names still delegates to the preserved legacy module. The old fixed-output checkpoint sweep is explicitly available through `poetry run -- python -m nnets_transformer.legacy sweep`; it is not a smoke test.

`graphs.py` and `cn_figures.py` are logged compatibility wrappers retaining their existing arguments. Their implementations are preserved in the package. The renderer delegates scientific graph construction to the shared versioned builder and explicitly selects `legacy-renderer-v1`.

`graph_analysis.ipynb` is retained output-free as a historical exploration notebook, with its original full-output archive intact. New notebook work imports `analysis.graphs`, `analysis.longitudinal` and `analysis.export`; the replacement reproducible computation is `command=export`. No new experiment requires running the historical notebook or trusting its old path-distance convention.

The fresh export writes normalized Parquet tables, a manifest carrying source identities/scientific versions and a standalone interactive `report.html`. Scientific plotting and role computations are shared package functions; the report is regenerated from saved measurements without loading a model. The optional old figure CLI remains a reader of trusted local legacy pickle files, not the fresh data format. Never load untrusted pickle files.
