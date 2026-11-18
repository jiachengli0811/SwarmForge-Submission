# SwarmForge

SwarmForge finds and repairs weaknesses in swarm controllers. It takes a controller's source code, extracts its control law, measures where that law fails, searches for edits to the law that make it fail less, and tests the result with generated missions.

## Pipeline

| Stage | Command | What it does |
|---|---|---|
| 1. Extraction | `python swarmmodeling/run_extraction.py <preset> --record` | Turns the controller's source code into a **record**: its law as a list of terms (who couples to whom, with which signal, gate and weight), plus bounds and parameters. |
| 2. Hardening | `python -m swarmforge.harden.run search <config> --out <dir>` | Measures the record on a fixed **sampling** of short window states, one sampling per weakness model. Then searches edits of its laws (insert, delete or replace nodes, plus named patterns) and keeps the children that score lower than their parent. |
| 3. Testing | `python -m swarmforge.mission.targeted weak <config> --record <file> --out <dir>` | Flies whole missions against a controller, aimed at the cells of the weakness space where the controller is weak. |

There are four weakness models per family:
- `obstacle_collision`: a drone hits a body.
- `agent_collision`: two drones touch.
- `timeout`: the swarm falls behind the pace its mission needs.
- one hold model per family: boids `fragmentation`, gradient `loss_target`, vasarhelyi `misalignment`, adaptive-swarm `off_path`.

A state counts against the controller when the controller fails there although some command would not.

## Run

```bash
pip install -r requirements.txt                  # Python 3.12

# 1. extraction: source code -> record
python swarmmodeling/run_extraction.py boids --record --outdir runs/extraction

# 2. hardening
python -m swarmforge.harden.run search configs_new/boids.yaml --out runs/harden --workers 10

# 3. weakness-guided testing of a hardened candidate
python -m swarmforge.mission.targeted weak configs_new/boids.yaml \
    --record runs/harden/candidates/<id>.json --out runs/weak --count 1000 --workers 10
```

## Layout

```
swarmmodeling/    stage 1: source code -> record (Python, C/C++ and MATLAB front ends)
swarmforge/       stages 2 and 3: weakness models, simulators, missions, hardening, plus the record format (algorithm/) and an interval solver (solver/)
missions/         each family's representative missions, flown by the scoring
data/             the 16 scenario missions and every candidate of the hardening runs on them (JSON in data.zip)
target/           the controllers under study (the files the runtime and extraction load)
```