"""Tasks own input preparation, prediction positions and clean answer labels."""
from typing import Protocol
from nnets_transformer.records import Example


class Task(Protocol):
    def prepare(self, adapter, config) -> tuple[list[Example], dict]: ...
