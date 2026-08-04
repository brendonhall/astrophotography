"""Flow CLI: run | validate | schema."""
from __future__ import annotations
import argparse
import json
import sys
import stages
from .graph import Graph
from .validate import validate
from .executor import run, FlowError
from . import builtins as _builtins


def _load_graph(args):
    if args.builtin:
        return getattr(_builtins, f"{args.builtin}_flow")()
    if not args.flow:
        raise SystemExit("provide a FLOW.json or --builtin")
    with open(args.flow) as f:
        return Graph.from_json(json.load(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="flow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("flow", nargs="?")
    r.add_argument("--builtin", choices=["linear", "starless"])
    r.add_argument("--input", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--no-cache", action="store_true")

    v = sub.add_parser("validate")
    v.add_argument("flow", nargs="?")
    v.add_argument("--builtin", choices=["linear", "starless"])

    sub.add_parser("schema")

    a = ap.parse_args(argv)

    if a.cmd == "schema":
        print(json.dumps(stages.list_stages(), indent=2))
        return 0
    if a.cmd == "validate":
        issues = validate(_load_graph(a))
        for i in issues:
            print(f"{i.level.upper()} [{i.where}] {i.message}")
        return 1 if any(i.level == "error" for i in issues) else 0
    if a.cmd == "run":
        try:
            rep = run(_load_graph(a), a.input, a.label, cache=not a.no_cache)
        except FlowError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        for w in rep.warnings:
            print(f"WARNING {w}")
        print(f"ran {len(rep.ran)}, cached {len(rep.cached)}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
