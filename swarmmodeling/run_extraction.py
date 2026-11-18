from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from swarmmodeling import emit
from swarmmodeling.pipeline import extract

TARGET = os.path.join(ROOT, "target")

PRESETS = {
    "boids": {
        "files": [os.path.join(TARGET, "pyglet-boids/boids", f)
                  for f in ("boid.py", "vector.py", "simulation.py")],
        "entry": "boid.Boid.update", "key": "pyglet-boids",
    },
    "adaptive": {
        "files": [os.path.join(TARGET, "adaptive_swarm/scripts/layered_planner", f)
                  for f in ("layered_planner_sim.py", "potential_fields.py", "tools.py")],
        "entry": "layered_planner_sim.Robot.local_planner", "key": "adaptive-swarm",
    },
    "vasarhelyi": {
        "files": [os.path.join(TARGET, "swarmlab-vasarhelyi/vasarhelyi/law.py")],
        "entry": "law.compute_vel_agent", "key": "vasarhelyi",
    },
    "gradient": {
        "files": [os.path.join(TARGET, "swarmalgorithms-gradient/gradient_swarm/law.py")],
        "entry": "law.gradient_step", "key": "gradient",
        "peers": "neighbours",
    },
}


def run(files, entry, key, *, out=None, dims=2, peers=None, settings=None, fixed=(),
        use_shortcuts=True, explain=False, record=False, profile=False, exact=False,
        self_size=None, outdir=None):
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit("input source not found: %s" % ", ".join(missing))
    ex = extract(files, entry, key, dims=dims, use_shortcuts=use_shortcuts, fixed=fixed,
                 peers=peers, settings=settings)
    algo = ex.algorithm
    if explain:
        print(json.dumps(ex.explain(), indent=2, sort_keys=True, default=str))
    if ex.unknown_fixed:
        print("warning: --fixed names no parameter of this algorithm: %s"
              % ", ".join(ex.unknown_fixed))
    text = emit.summary(algo)
    print(text)

    out = out or key
    outdir = outdir or os.path.join(HERE, "output")
    os.makedirs(outdir, exist_ok=True)

    def write(suffix, content):
        path = os.path.join(outdir, out + suffix)
        with open(path, "w") as f:
            f.write(content)
        print("wrote %s" % os.path.relpath(path))

    write(".json", emit.to_json(algo))
    write("_summary.txt", text + "\n")
    if (record or profile) and not algo.terms:
        print("no record or profile: %s" % next(
            (n for n in algo.notes if n.startswith(("refused", "no command"))), "no terms"))
        record = profile = False
    if record:
        from swarmforge.algorithm.serialize import algorithm_to_record
        from to_record import self_size_for
        if self_size is None and emit.uses_clearance(algo):
            self_size = ex.names.body_radius
        algorithm = emit.to_swarmff(algo, self_size=self_size_for(algo, self_size))
        write(".record.json", json.dumps(algorithm_to_record(algorithm), indent=1))
    if profile:
        doc = dict(ex.profile(), record=out + ".record.json")
        write(".profile.json", json.dumps(doc, indent=1, default=float))
    if exact:
        from swarmmodeling.exact import exact_law
        write(".exact.json", json.dumps(exact_law(ex), indent=1, default=float))
    return ex


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="named input algorithm")
    ap.add_argument("--files", nargs="+", help="source files to load")
    ap.add_argument("--entry", help="entry function: module.Class.method")
    ap.add_argument("--key", default="algorithm", help="the record's target_key (and output name)")
    ap.add_argument("--dims", type=int, help="spatial dimensions (default 2)")
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                    help="configuration constants to override: the scenario to extract")
    ap.add_argument("--peers", help="the collection holding the drone's peers, when two "
                                    "collections are coupled identically and nothing else tells")
    ap.add_argument("--fixed", nargs="+", default=[], metavar="NAME",
                    help="parameters a search must not change; every other one is tunable")
    ap.add_argument("--no-shortcuts", action="store_true",
                    help="inline known vector helpers instead of summarizing them")
    ap.add_argument("--explain", action="store_true",
                    help="print what each stage found (tick, loop, names) and why")
    ap.add_argument("--record", action="store_true", help="also write <name>.record.json")
    ap.add_argument("--profile", action="store_true", help="also write <name>.profile.json")
    ap.add_argument("--exact", action="store_true", help="also write <name>.exact.json")
    ap.add_argument("--self-size", type=float,
                    help="--record: the radius the source subtracts to get clearance, when the "
                         "code does not build the body with one literal")
    ap.add_argument("--outdir", help="output directory (default: output/)")
    args = ap.parse_args()

    if args.preset:
        spec = dict(PRESETS[args.preset])
    elif args.files and args.entry:
        spec = {"files": args.files, "entry": args.entry, "key": args.key}
    else:
        ap.error("give a preset or --files ... --entry ...")
    settings = dict(spec.get("settings", {}))
    for item in args.set:
        name, _, value = item.partition("=")
        settings[name] = float(value)
    run(spec["files"], spec["entry"], spec["key"], out=spec.get("out"),
        dims=args.dims or spec.get("dims", 2), peers=args.peers or spec.get("peers"),
        settings=settings, fixed=args.fixed, use_shortcuts=not args.no_shortcuts,
        explain=args.explain, record=args.record, profile=args.profile, exact=args.exact,
        self_size=args.self_size, outdir=args.outdir)


if __name__ == "__main__":
    main()
