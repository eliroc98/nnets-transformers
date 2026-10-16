"""Fresh configured experiment entry point; legacy helpers remain import-compatible."""

def __getattr__(name):
    from nnets_transformer.legacy import sweep
    return getattr(sweep, name)


if __name__ == "__main__":
    from nnets_transformer.cli import main
    main()
