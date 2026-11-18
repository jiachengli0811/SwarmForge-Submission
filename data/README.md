# Data

The 16 scenario missions, and every candidate from hardening each family's original on each of them.

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
