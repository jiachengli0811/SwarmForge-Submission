# swarmforge

Stages 2 and 3 of the pipeline, plus the record format that stage 1 writes. Everything starts from two files: a family config (`configs_new/<family>.yaml`) and a controller record. The configs and records are packed in `configs_new/configs.zip`; unpack them first with `unzip configs_new/configs.zip -d configs_new`.

| Package | Role |
|---|---|
| `algorithm/` | The record format that extraction (stage 1) writes: algorithm, expressions and serializer. |
| `solve/` | The weakness models, the window states they are read on, the record's laws (`laws.py`, `dynamics.py`) and the sampling of window states. A model rolls the closed loop H ticks from a state and gives a verdict. |
| `runtime/` | Flies a record inside each target's own simulator (`targets/`), with obstacles and movers from a mission document. |
| `mission/` | Mission generation within the family's test range, flight, verdicts and the weakness spaces. Weakness-guided testing is `guided.py`, and `targeted.py` is its command line. |
| `harden/` | The hardening search: law-tree operators and patterns (`operators/`), screening and validity, property profiles (`properties/`), scoring (`scoring.py`, `space.py`), policy, strategy and session. |

## Measuring one record

```bash
python -m swarmforge.solve.measure configs_new/boids.yaml --model agent_collision [--record path/to/record.json]
```

This prints the share of the model's sampled states in each verdict (SAFE, VIOLATES, DOOMED, SINGULAR, INFEASIBLE) and the score (VIOLATES + SINGULAR). It samples the states first if none are saved. It uses the config's own record unless `--record` names another, such as a hardened candidate. `--samples`, `--seed` and the count overrides (`--drones`, `--static`, `--dynamic`, `--horizon`) draw or load a sampling of their own. `--witnesses N --out FILE` saves the first N violating states.

## Hardening

```bash
python -m swarmforge.harden.run search   configs_new/boids.yaml --out runs/harden   # the search
```

Each round, the strategy draws a parent and the policy proposes edits. Each variant is screened for validity and property profile, then measured on every model's sampling. A child is kept when it improves on its parent by more than the paired noise bar. A run writes:
- `candidates/<id>.json`: each record with its lineage.
- `verdicts/`
- `candidates.csv`: the `total` score per candidate; lower is less weak.
- `properties.csv`, `rounds.csv`, `operators.csv` and `index.json`.

## Testing

```bash
python -m swarmforge.mission.targeted weak configs_new/boids.yaml --record runs/harden/candidates/<id>.json \
    --out runs/weak --count 1000 --workers 10
```

- `weak` writes the missions to `missions-<name>/`, and `results.csv` (one row per mission: verdict, crash kind, and what each model's space saw) plus `guided.json` to `<name>/`. `<name>` is the record's file name unless `--name` is given.
