# swarmmodeling

Stage 1 of the pipeline. It reads a swarm controller's source code (Python, C/C++ or MATLAB) and writes its control law as a record.

```bash
python swarmmodeling/run_extraction.py boids --record          # also: adaptive, vasarhelyi, gradient
python swarmmodeling/run_extraction.py boids --explain         # what each stage found, and why
python swarmmodeling/run_extraction.py --files a.py b.py --entry module.Class.method --key name --record
```

`--fixed NAME ...` names parameters a search must leave alone; every other parameter is tunable. `--outdir DIR` changes where outputs go (default `swarmmodeling/output/`).

## Outputs

- `<name>.json`: the program-analysis view.
- `<name>_summary.txt`: a readable summary, also printed.
- `<name>.record.json` (with `--record`): the record that hardening reads.
- `<name>.profile.json` (with `--profile`): membership and plant details, which the record cannot hold.
- `<name>.exact.json` (with `--exact`): the executed command as an operation graph.

When the law cannot be decomposed into terms, no record is written and the summary says why.

## Stages

`pipeline.extract` runs seven stages:

1. **Parse** (`frontend/`, `program.py`): tree-sitter turns each language into one shared intermediate form. MATLAB is lowered to it first.
2. **Execute one tick** (`tick.py`, `execute.py`, `arrays.py`): the body of the time loop, with the controlled drone bound and calls inlined.
3. **Canonicalize** (`canonical.py`).
4. **Find the loop** (`loop.py`): state, command, plant and time step.
5. **Name the variables** (`naming.py`): position, velocity, peers, size and ranges, all inferred from structure.
6. **Classify the terms** (`classify.py`): one term per summand: source, direction, signal, aggregator, gate, bounds and weight.
7. **Emit** (`emit.py`, `exact.py`).

Writing a record uses the record format in `../swarmforge/algorithm`.
