import argparse
from pathlib import Path


def weak(args):
    from .guided import run_guided
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.record).stem
    run_guided(str(args.config), name, str(args.record), count=args.count, seed=args.seed,
               out_dir=str(out), workers=args.workers, progress=not args.quiet)
    print("done", out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="weakness-guided testing")
    parser.add_argument("suite", choices=("weak",))
    parser.add_argument("config", type=Path)
    parser.add_argument("--record", type=Path, required=True, help="the controller under test")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    weak(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
