# Astar Island — Challenge Reference

Concise reference for the Astar Island Viking Civilisation Prediction challenge.
Full docs at [app.ainm.no](https://app.ainm.no).

---

## Goal

Observe a black-box Norse civilisation simulator through limited viewports and predict the **probability distribution of terrain types** across the entire 40×40 map after 50 simulated years.

---

## Round Structure

1. Admin creates a round with a fixed map, hidden parameters, and **5 random seeds**
2. You get **50 viewport queries total** (shared across all 5 seeds), each revealing max 15×15 cells
3. Each query runs one stochastic simulation — same seed gives different outcomes each time
4. Submit an **H×W×6 probability tensor** per seed
5. Scored by entropy-weighted KL divergence against ground truth (Monte Carlo from hundreds of runs)

---

## Terrain Types & Prediction Classes

| Internal Code | Terrain    | Class Index | Behavior |
|:---:|------------|:---:|----------|
| 10  | Ocean      | 0 (Empty)  | Static, impassable, borders map |
| 11  | Plains     | 0 (Empty)  | Static, buildable land |
| 0   | Empty      | 0          | Generic empty cell |
| 1   | Settlement | 1          | **Dynamic** — active Norse settlement |
| 2   | Port       | 2          | **Dynamic** — coastal settlement with harbour |
| 3   | Ruin       | 3          | **Dynamic** — collapsed settlement |
| 4   | Forest     | 4          | Mostly static, can reclaim ruins |
| 5   | Mountain   | 5          | Static, never changes |

**Static cells** (Ocean, Mountain, Forest) are predictable. **Dynamic cells** (Settlement, Port, Ruin) are where scoring matters.

---

## Simulation Mechanics

Each of 50 years cycles through these phases in order:

### 1. Growth
- Settlements produce food from adjacent terrain
- Population grows when conditions are favorable
- Coastal settlements develop ports and build longships
- Prosperous settlements expand by founding new settlements nearby

### 2. Conflict
- Settlements raid each other; longships extend raiding range
- Desperate (low food) settlements raid more aggressively
- Successful raids loot resources and damage defenders
- Conquered settlements can change faction allegiance

### 3. Trade
- Ports within range trade if not at war
- Generates wealth and food; tech diffuses between partners

### 4. Winter
- Variable severity each year; all settlements lose food
- Settlements collapse from starvation/raids/winter → become **Ruins**
- Population disperses to nearby friendly settlements

### 5. Environment
- Nearby thriving settlements can reclaim Ruins → new outposts
- Coastal ruins can be restored as Ports
- Unclaimed ruins get overtaken by Forest or fade to Plains

---

## Map Generation

Each map is procedurally generated from a **map seed** (visible to you):
- Ocean borders surround the map
- Fjords cut inland from random edges
- Mountain chains form via random walks
- Forest patches with clustered groves
- Initial settlements placed on land, spaced apart

You can reconstruct the initial terrain layout locally from the seed.

---

## Settlement Properties

Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, faction allegiance (`owner_id`).

**Initial states** expose only position and port status. Full stats (population, food, wealth, defense) are only visible through simulation queries.

---

## Key Constraints

| Constraint | Value |
|---|---|
| Map size | 40×40 (default) |
| Seeds per round | 5 |
| Total queries per round | 50 (shared across all seeds) |
| Max viewport size | 15×15 |
| Simulation length | 50 years |
| Prediction tensor | H×W×6 probabilities per cell |
| Hidden parameters | Same for all seeds in a round, unknown to you |

---

## Scoring

### Ground Truth
Computed by running the simulation hundreds of times with the true hidden parameters → probability distribution per cell.

### Metric: Entropy-Weighted KL Divergence

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)          (per cell, p=truth, q=prediction)
entropy(cell) = -Σ pᵢ × log(pᵢ)             (weight — higher entropy = more important)

weighted_kl = Σ entropy(cell) × KL(cell) / Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- **100** = perfect match, **0** = terrible
- Only **dynamic cells** (non-zero entropy) contribute — static cells are excluded
- Higher-entropy cells (more uncertain outcomes) count more

### Critical: Never Use Zero Probabilities
If ground truth has `pᵢ > 0` but your prediction has `qᵢ = 0`, KL divergence → ∞, destroying your score. Always enforce a **minimum floor of 0.01** per class, then renormalize.

### Aggregation
- **Round score** = average of 5 seed scores (unsubmitted seeds score 0)
- **Leaderboard score** = best round score ever (weighted by round weight)

---

## API Reference

**Base URL:** `https://api.ainm.no/astar-island`
**Auth:** Cookie (`access_token`) or Bearer token header.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/rounds` | Public | List all rounds |
| GET | `/rounds/{round_id}` | Public | Round details + initial states for all seeds |
| GET | `/budget` | Team | Remaining query budget |
| POST | `/simulate` | Team | Run one simulation, observe viewport (costs 1 query) |
| POST | `/submit` | Team | Submit H×W×6 prediction tensor for one seed |
| GET | `/my-rounds` | Team | Your scores, rank, budget per round |
| GET | `/my-predictions/{round_id}` | Team | Your predictions with argmax/confidence |
| GET | `/analysis/{round_id}/{seed_index}` | Team | Post-round ground truth comparison |
| GET | `/leaderboard` | Public | Global leaderboard |

### POST /simulate — Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,        // 0–4
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,       // 5–15
  "viewport_h": 15        // 5–15
}
```

### POST /simulate — Response
```json
{
  "grid": [[...]],              // viewport_h × viewport_w terrain codes
  "settlements": [{             // settlements within viewport
    "x": 12, "y": 7,
    "population": 2.8, "food": 0.4, "wealth": 0.7,
    "defense": 0.6, "has_port": true, "alive": true, "owner_id": 3
  }],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "queries_used": 24, "queries_max": 50
}
```

### POST /submit — Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "prediction": [[[0.85, 0.05, 0.02, 0.03, 0.03, 0.02], ...], ...]
}
```
- `prediction[y][x]` = 6 probabilities summing to 1.0 (±0.01 tolerance)
- Resubmitting overwrites — only the last submission counts

### Rate Limits
- `/simulate`: 5 req/sec per team
- `/submit`: 2 req/sec per team

---

## Authentication

1. Log in at [app.ainm.no](https://app.ainm.no) with Google
2. DevTools → Application → Cookies → copy `access_token`
3. Use as cookie or `Authorization: Bearer <token>` header

---

## Strategic Considerations

- **50 queries for 5 seeds on a 40×40 map** — each query only reveals 15×15 = 225 of 1600 cells
- **Stochastic outcomes** — querying the same viewport twice gives different results (useful for building distributions)
- **Hidden parameters are shared** — insights from one seed transfer to all 5
- **Focus on dynamic cells** — static cells (ocean, mountain) are trivially predictable and don't affect scoring
- **Always submit something** — even uniform 1/6 beats a missing submission (which scores 0)
