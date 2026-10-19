"""Process-local task preparation reuse, independent of checkpoint activations."""
from copy import deepcopy
from dataclasses import asdict
from nnets_transformer.execution.artifacts import fingerprint
from . import get_task


class PreparedTasks:
    def __init__(self):
        self.entries = {}

    def prepare(self, adapter, config):
        files = adapter.provenance.get("tokenizer_files")
        # Without file hashes we cannot establish tokenizer equivalence.
        if not files:
            return get_task(config.name).prepare(adapter, config)
        key = fingerprint({"task": asdict(config), "tokenizer_files": files,
                           "vocab_size": adapter.vocab_size, "context_length": adapter.context_length})
        if key not in self.entries:
            if len(self.entries) >= 16:
                self.entries.pop(next(iter(self.entries)))
            self.entries[key] = get_task(config.name).prepare(adapter, config)
        examples, dataset = deepcopy(self.entries[key])
        dataset["tokenizer_revision"] = adapter.provenance["tokenizer_revision"]
        dataset.pop("dataset_sha256", None)
        dataset["dataset_sha256"] = fingerprint(dataset)
        return examples, dataset
