from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from swarmmodeling import emit
from swarmmodeling.runtime.model import load


def self_size_for(ir, given: float = None) -> float:
    if given is not None:
        return given
    if emit.uses_clearance(ir) and "self_size" not in ir.params:
        print("warning: this algorithm reads clearance and a record has to "
              "spell it as `dist - self_size`; assuming 0 (clearance == centre "
              "distance).  pyglet-boids needs --self-size 10")
    return 0.0


def convert(src: str, dst: str, self_size: float = 0.0) -> str:
    from swarmforge.algorithm.serialize import save_algorithm
    algorithm = emit.to_swarmff(load(src), self_size=self_size)
    save_algorithm(algorithm, dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="extracted-algorithm JSON (the AlgorithmIR)")
    ap.add_argument("-o", "--out", help="record path "
                                        "(default: <source>.record.json)")
    ap.add_argument("--self-size", type=float,
                    help="the radius the source subtracts to get clearance "
                         "(pyglet-boids: 10; a platform that already measures "
                         "to the surface: 0, the default)")
    ap.add_argument("--target", help="take --self-size from this adapter's "
                                     "own convention (boids, adaptive)")
    ap.add_argument("--summary", action="store_true",
                    help="print the converted algorithm")
    args = ap.parse_args()

    out = args.out or os.path.splitext(args.source)[0] + ".record.json"
    ir = load(args.source)
    given = args.self_size
    if given is None and args.target:
        from swarmmodeling.runtime import adapter
        given = getattr(adapter(args.target), "SELF_SIZE", None)
    algorithm = emit.to_swarmff(ir, self_size=self_size_for(ir, given))
    from swarmforge.algorithm.serialize import save_algorithm
    save_algorithm(algorithm, out)
    if args.summary:
        print(algorithm.summary())
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
