from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import FOCUSES, read_config
from .models import MODELS, build_model
from .record import read_record
from .verdict import Verdict


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", default="obstacle_collision",
                        choices=sorted(MODELS))
    parser.add_argument("--record", type=Path, default=None,
                        help="a record other than the config's own")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--focus", choices=FOCUSES, default=None)
    parser.add_argument("--bank-dir", type=Path, default=None)
    parser.add_argument("--drones", type=int, default=None)
    parser.add_argument("--static", type=int, default=None)
    parser.add_argument("--dynamic", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--witnesses", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = read_config(args.config)
    changes = {}
    if args.focus:
        changes["focus"] = args.focus
    if args.bank_dir:
        changes["bank_dir"] = args.bank_dir.resolve()
    if changes:
        config = dataclasses.replace(
            config, sampling=dataclasses.replace(config.sampling, **changes))
    record = (read_record(args.record, config.physics.record_params)
              if args.record else None)
    model = build_model(args.model, config, record, drones=args.drones,
                        static_obstacles=args.static,
                        dynamic_obstacles=args.dynamic, horizon=args.horizon)
    reading, state, outcome = model.measure(args.samples, args.seed,
                                            witnesses=max(10, args.witnesses))
    spec = reading.spec
    print("%s / %s  record=%s" % (reading.family, reading.model,
                                  Path(reading.record).name))
    print("drones=%d static=%d dynamic=%d horizon=%d  focus=%s  states=%d  "
          "%.2fs" % (spec.drones, spec.static_obstacles,
                     spec.dynamic_obstacles, spec.horizon, reading.focus,
                     reading.samples, reading.seconds))
    print("sampling %s (%s)" % (reading.bank, "drawn and saved"
                            if reading.bank_created else "loaded"))
    print("contested share of the window states: %.4f%% +- %.4f%%"
          % (100.0 * reading.contested_share,
             100.0 * reading.contested_share_se))
    for verdict in Verdict:
        print("  %-10s %8.4f%%  (%d)" % (
            verdict.name, 100.0 * reading.fractions[verdict.name],
            outcome.count(verdict)))
    print("  score (VIOLATES + SINGULAR) = %.6f   over every window state "
          "%.3g" % (reading.score, reading.window_score))

    if args.out:
        document = reading.to_dict()
        if args.witnesses:
            index = reading.witnesses[:args.witnesses]
            document["witness_states"] = state.take(index).to_dicts()
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
