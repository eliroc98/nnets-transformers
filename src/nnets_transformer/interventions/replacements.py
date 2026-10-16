"""Replacement values are separate from selected component/head scopes."""
import torch


class Zero:
    def apply(self, value, selection, component):
        if selection.head is None:
            value.zero_()
        else:
            value.select(component.head_axis, selection.head).zero_()


class Identity:
    """Validation control; not an experimental replacement condition."""
    def apply(self, value, selection, component):
        return None


def get_replacement(name):
    if name == "zero":
        return Zero()
    raise ValueError(f"Unimplemented replacement: {name}")


class TensorReplacement:
    """All-position reference tensor; head selection is owned by the scope."""
    def __init__(self, tensors, broadcast=False):
        self.tensors, self.broadcast = tensors, broadcast

    def apply(self, value, selection, component):
        source = self.tensors[selection.hook].to(device=value.device, dtype=value.dtype)
        if selection.head is not None:
            value = value.select(component.head_axis, selection.head)
            source = source.select(component.head_axis, selection.head)
        if not self.broadcast and value.shape != source.shape:
            raise ValueError('Donor/reference axes or all-position alignment mismatch')
        value.copy_(source.expand_as(value))
