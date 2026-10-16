"""Logged compatibility entry point; fresh analyses use nnets-transformer export."""

def __getattr__(name):
    import importlib
    return getattr(importlib.import_module("nnets_transformer.legacy.render"), name)


if __name__ == "__main__":
    from nnets_transformer.legacy.__main__ import launch
    launch("render")
