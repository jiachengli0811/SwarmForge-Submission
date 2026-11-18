# Data

The 16 scenario missions, and every candidate from hardening each family's original on each of them.

Only the GIFs are stored loose. Every JSON file (the mission documents and the hardened candidates) is packed in `data.zip`, with the paths below. Unpack it in place before using them:

```bash
cd data && unzip data.zip      # writes missions/<family>/scenario-N.json and harden/...
```

## Missions

`missions/<family>/` holds four failure scenarios per family, one mission each. The four in a family differ from one another in root cause, failure kind, or phase. Each scenario has two files:
- `scenario-N.json` (in `data.zip`): the mission document, as `swarmforge.mission.store.read_case` reads it. It is flown from its own `_meta.spawn_seed`.
- `scenario-N.gif`: the original from the hardening runs below (`baseline.json`) flying that mission until it fails.

| Family | Scenario | Original |
|---|---|---|
| adaptive-swarm | scenario-1 | body hit at tick 94 |
| adaptive-swarm | scenario-2 | body hit at tick 78 |
| adaptive-swarm | scenario-3 | timeout (1000 ticks) |
| adaptive-swarm | scenario-4 | body hit at tick 43 |
| boids | scenario-1 | drone contact at tick 73 |
| boids | scenario-2 | drone contact at tick 50 |
| boids | scenario-3 | body hit at tick 134 |
| boids | scenario-4 | timeout (800 ticks) |
| gradient | scenario-1 | body hit at tick 25 |
| gradient | scenario-2 | drone contact at tick 39 |
| gradient | scenario-3 | drone contact at tick 33 |
| gradient | scenario-4 | drone contact at tick 97 |
| vasarhelyi | scenario-1 | drone contact at tick 495 |
| vasarhelyi | scenario-2 | drone contact at tick 302 |
| vasarhelyi | scenario-3 | body hit at tick 685 |
| vasarhelyi | scenario-4 | timeout (2000 ticks) |

## Hardened candidates

`harden/<family>/<scenario>/<seed>/` (in `data.zip`) holds one hardening run of the family's original:
- full-size config;
- flow + plain region scoring, with `--missions` set to that one scenario mission;
- 400 evaluations;
- seed `s0`.

That makes 16 runs, with 20,907 candidate files.

- `candidates/<id>.json` is a candidate's record plus `_meta`. `_meta` gives the parent, generation and round, the edit that made it, the validity check, the property comparison with the original, and the weakness reading if measured. `baseline.json` is the original the run started from. Sampling paths in `_meta` name the saved sampling file.
- In each candidate's `_meta`:
  - `stage`: `measured`, `screened`, `invalid`, `duplicate` or `refused`;
  - `retained`: true if kept in the pool;
  - `score.scores`: one score per model, and `score.total`, the weighted weakness score (lower is less weak).

  A run's top candidates are its `retained` candidates with the lowest `score.total`.

Once `data.zip` is unpacked, every record loads with `swarmforge.harden.document.load` and can be flown or tested like any other, for example:

```bash
python -m swarmforge.mission.targeted weak configs_new/boids.yaml \
    --record data/harden/boids/scenario-1/s0/candidates/cand-0030.json --out runs/w --count 100
```

**These runs started from an earlier version of three originals.** Each run's `baseline.json` differs from the record in `configs_new/` in one parameter:

| Family | Parameter | In these runs | In `configs_new/` |
|---|---|---|---|
| boids | `attractor_factor` | 0.0035 | 0.014 |
| gradient | `goal_gain` | 100 | 10 |
| vasarhelyi | `p_rep` | 0.03 | 0.12 |

adaptive-swarm's original has the same parameter values. Its file differs only in format.

To compare a candidate with the original it descends from, pass that run's `baseline.json` as `--original`.
