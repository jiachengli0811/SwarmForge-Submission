# solver

An interval branch-and-prune solver for the weakness models with a one-tick horizon. It answers the same question as the pipeline's sampling (`swarmforge.solve`), but proves the verdict over whole boxes of states. The pipeline does not use it. It reads `swarmforge.solve`, the configs and the saved sampling.

There are 13 models in scope: every model whose config has `horizon: 1`, on the config's own record.

| Family | Models |
|---|---|
| boids | timeout, fragmentation |
| gradient | obstacle_collision, agent_collision, loss_target |
| vasarhelyi | obstacle_collision, agent_collision, timeout, misalignment |
| adaptive-swarm | obstacle_collision, agent_collision, timeout, off_path |

## How it works

- **Box:** a set of states. Every drone's position and velocity is an interval; the bodies, target and route are fixed.
- **Engine** (`engine.py`, on `interval.py` and `expr.py`): evaluates the model on a box, statement by statement. That covers the movers' step, each law's window, gate, magnitude, direction and aggregate, the bounds, the plant, the update order, the formation slots, the margin after the tick, admissibility and the physics doom test.
- **Answer:** a verdict proved for every state in the box (SAFE, VIOLATES, DOOMED or INFEASIBLE), or *undecided*.
- **Discrete choices** (a peer in range, a gate firing, the nearest body, a clamp biting) are three-valued. Rounding is outward on every operation, so a proved verdict holds in real arithmetic.
- **Branch-and-prune** (`paving.py`): an undecided box is cut in four and asked again, down to a set depth.

Not modelled: horizons past 1, law constructs the four records do not use (`around`, `slide`, `detour`, a nearest-k cap, a clamp floor), and bodies of radius 0. Each raises `NotImplementedError`.

## Commands

Run from the repository root. Outputs go to `runs/solver/`.

```bash
python -m solver.check                    # engine against the pipeline on sampled states and small boxes
python -m solver.slices --workers 24      # solver and sampling on the same 2-D slices through sampled states
python -m solver.neighbourhood            # largest proved full-state box around sampled states
python -m solver.figures                  # one picture per slice
python -m solver.report                   # summary.md and an overview figure
```

`--cases` limits a run to some models, for example `--cases gradient/obstacle_collision,boids/timeout`; the default is all 13.

- **`check`** exits non-zero if the engine ever contradicts the pipeline.
- **`slices`** moves one drone over a square while the rest of a real sampled state stays fixed. It paves the square to `--max-depth` and samples `--samples` points over the same square. `--per-verdict N` takes N VIOLATES and N SAFE sampled states per model.
  - The full study setting (`--per-verdict 10 --max-depth 12 --samples 100000`) takes about five minutes on 24 workers, once the sampling exists.

## What it shows

- The engine agrees with the pipeline everywhere it commits.
- On a 2-D slice it gives the violation regions with proved bounds, in seconds per slice. Those bounds contain the sampled violating share.
- It finds small violation regions that sampling at small sizes misses.
- In the full state of 16 or more coordinates, the proved boxes around sampled states are small. The solver therefore does not replace the sampled score.
- On gradient, the mover's stand-off equals the laws' sensing range. A mover held at its stand-off then sits exactly at that edge, so whether a law senses it rests on floating-point rounding. Some of gradient's slices therefore keep undecided areas that do not shrink with depth.
