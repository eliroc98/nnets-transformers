"""Launch the dashboard or import legacy artifacts, with persistent attempt logs.

    python -m nnets_transformer.dashboard serve [--config configs/dashboard.yaml] [--host H] [--port P]
    python -m nnets_transformer.dashboard import-legacy --source . --output exports/legacy-induction
"""
import argparse
from pathlib import Path

from nnets_transformer.observability import Job, Interrupted


def load_config(path):
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True) if Path(path).is_file() else {}
    roots = config.get("artifact_roots") or []
    if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
        raise ValueError("artifact_roots must be a list of directories")
    if config.get("host", "127.0.0.1") not in {"127.0.0.1", "localhost"} and not config.get("allow_remote", False):
        raise ValueError("Serve on localhost (use an SSH tunnel), or set allow_remote: true explicitly")
    for key in ("cache_entries", "network_max_edges", "status_attempts", "log_tail_bytes"):
        if key in config and int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--config", default="configs/dashboard.yaml")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--debug", action="store_true")
    legacy = sub.add_parser("import-legacy")
    legacy.add_argument("--source", default=".")
    legacy.add_argument("--output", required=True)
    legacy.add_argument("--quantile", type=float, default=.95)
    for command in (serve, legacy):
        command.add_argument("--log-root", default="logs")
    args = parser.parse_args(argv)
    job_id = "dashboard" if args.command == "serve" else "dashboard-legacy-import"
    try:
        with Job(job_id, root=args.log_root) as job:
            if args.command == "import-legacy":
                from .legacy import import_legacy
                output = import_legacy(args.source, args.output, job, args.quantile)
                print(f"Imported legacy artifacts into {output}")
                return
            job.stage("configuration")
            config = load_config(args.config)
            config.setdefault("log_root", args.log_root)
            host = args.host or config.get("host", "127.0.0.1")
            port = args.port or int(config.get("port", 8050))
            if host not in {"127.0.0.1", "localhost"} and not config.get("allow_remote", False):
                raise ValueError("Serve on localhost (use an SSH tunnel), or set allow_remote: true explicitly")
            from .repository import Repository
            from .app import create_app
            job.stage("catalog")
            repository = Repository(config.get("artifact_roots", ["exports"]), int(config.get("cache_entries", 32)))
            job.event("catalog_loaded", exports=len(repository.catalog), problems=repository.problems)
            app = create_app(repository, config, job)
            job.stage("serving", host=host, port=port)
            print(f"Dashboard: http://{host}:{port}/  (CPU-only; reads {config.get('artifact_roots')})", flush=True)
            # Werkzeug swallows KeyboardInterrupt (and so the Job's signal exception);
            # remember the signal so a stopped server is recorded as interrupted, not completed.
            import signal
            received = {}
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous = signal.getsignal(sig)
                def record(signum, frame, previous=previous):
                    received["signum"] = signum
                    previous(signum, frame)
                signal.signal(sig, record)
            app.run(host=host, port=port, debug=args.debug, use_reloader=False)
            if received:
                raise Interrupted(received["signum"])
    except Interrupted as exc:
        raise SystemExit(128+exc.signum)


if __name__ == "__main__":
    main()
