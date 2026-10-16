"""Model instrumentation boundary; tasks and runners do not own hook axes."""
from typing import Protocol
from nnets_transformer.records import Component, Example, Selection


class Adapter(Protocol):
    inventory: tuple[Component, ...]
    provenance: dict
    vocab_size: int
    context_length: int
    tokenizer: object

    def selections(self, resolution: str) -> list[Selection]: ...
    def forward(self, example: Example, observations: list[Selection], intervention=None): ...
