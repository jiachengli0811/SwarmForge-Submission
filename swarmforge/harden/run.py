from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml

from ..solve.config import read_config
from ..solve.record import named_records
from .config import read_harden
from .document import load
from .report import text
from .search import run as run_search
from .session import Session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
            ("baseline", "measure and profile the original"),
            ("evaluate", "read variants made elsewhere against the original"),
            ("search", "run the configured strategy and policy"),
            ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument("--out", type=Path, default=None,
                             help="where the run is written")
        command.add_argument("--original", type=Path, default=None,
                             help="the record to harden, or to compare "
                                  "against (default: the config's own)")
        command.add_argument("--workers", type=int, default=None)
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--max-evaluations", type=int, default=None,
                             dest="max_evaluations")
        command.add_argument("--quiet", action="store_true")
        command.add_argument("--scoring", type=_scoring, default=None,
                             help="bank, or a scoring block as a YAML "
                                  "mapping: in place of harden.scoring")
        command.add_argument("--missions", nargs="+", type=Path,
                             default=None, metavar="CASE",
                             help="mission case files -- the original's "
                                  "failures -- whose flights the reach "
                                  "efforts start from, in place of "
                                  "harden.scoring.missions")
        if name == "search":
            command.add_argument(
                "--resume", type=Path, default=None, metavar="RUN_DIR",
                help="carry on the run written to RUN_DIR; "
                     "--max-evaluations is then the total, its evaluations "
                     "included, and --out must be a folder of its own")
        if name == "evaluate":
            command.add_argument("--records-dir", type=Path, default=None,
                                 help="a mission campaign's controllers, or "
                                      "every */algorithm.json below it")
            command.add_argument("--record", action="append", default=[],
                                 metavar="NAME=PATH",
                                 help="one more variant; repeatable")
            command.add_argument(
                "--allow-fixed", action=argparse.BooleanOptionalAction,
                default=True,
                help="read a variant that re-values a parameter the original "
                     "fixes (a configuration fix), noting it, rather than "
                     "refusing it (default: on)")
    return parser


def _scoring(text: str):
    if text == "bank":
        return text
    value = yaml.safe_load(text)
    if not isinstance(value, dict) or "name" not in value:
        raise argparse.ArgumentTypeError(
            "--scoring takes bank or a YAML mapping with a name, not %r"
            % text)
    return value


def _overrides(args) -> dict:
    return {"workers": args.workers, "seed": args.seed,
            "max_evaluations": args.max_evaluations,
            "scoring": getattr(args, "scoring", None),
            "missions": getattr(args, "missions", None)}


def _evaluate(args) -> int:
    config = read_config(args.config)
    named = named_records(config.record_path, records_dir=args.records_dir,
                          records=tuple(args.record))
    named.pop("config", None)
    if not named:
        print("nothing to evaluate: pass --records-dir or --record")
        return 2
    overrides = _overrides(args)
    if overrides["max_evaluations"] is None:
        overrides["max_evaluations"] = len(named)
    harden = read_harden(args.config, overrides)
    with Session(config, harden, record_path=args.original,
                 out_dir=args.out, progress=not args.quiet) as session:
        considered = []
        for name, path in named.items():
            label = name if name not in session.archive else "ext-%s" % name
            candidate = session.consider(load(path), label=label,
                                         source=str(path),
                                         fixed=not args.allow_fixed)
            considered.append(candidate)
            if not args.quiet and (candidate.stage != "screened"
                                   or candidate.note):
                print("  %-9s %s: %s" % (candidate.id,
                                         candidate.stage.upper(),
                                         candidate.note))
        session.measure(considered)
        session.mark_round("evaluate")
    print()
    print(text(session))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "baseline":
        config = read_config(args.config)
        harden = read_harden(args.config, _overrides(args))
        with Session(config, harden, record_path=args.original,
                     out_dir=args.out, progress=not args.quiet) as session:
            pass
        print()
        print(text(session))
        return 0
    try:
        session = run_search(args.config, record_path=args.original,
                             out_dir=args.out, overrides=_overrides(args),
                             progress=not args.quiet, resume=args.resume)
    except NotImplementedError as pending:
        print("\nstopped after the original: %s" % pending)
        if args.out:
            print("the original's run is written to %s" % args.out)
        return 3
    print()
    print(text(session))
    return 0


if __name__ == "__main__":
    sys.exit(main())
