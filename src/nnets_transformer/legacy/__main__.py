"""Logged compatibility launcher: python -m nnets_transformer.legacy sweep|render|figures."""
import argparse
import runpy
import sys
from nnets_transformer.observability import Job, Interrupted


def launch(name, argv=None):
    module = {"sweep": "legacy.sweep", "render": "legacy.render", "figures": "analysis.legacy_figures"}[name]
    prior = sys.argv
    sys.argv = [prior[0], *(argv if argv is not None else prior[1:])]
    try:
        with Job("legacy-"+name) as job:
            job.stage("legacy_execution", implementation=module)
            runpy.run_module("nnets_transformer."+module, run_name="__main__")
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)
    finally:
        sys.argv = prior


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", choices=["sweep", "render", "figures"])
    args, rest = parser.parse_known_args()
    launch(args.entry, rest)
