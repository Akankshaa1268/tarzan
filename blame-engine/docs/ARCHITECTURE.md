# BlameEngine — System Architecture

## Overview

```
                        ┌─────────────────────────────────────────┐
                        │           USERS                          │
                        │   F1 Fans  ◄──────►  Team Engineers      │
                        └───────────┬─────────────────┬───────────┘
                                    │                 │
                        ┌───────────▼─────────────────▼───────────┐
                        │        FRONTEND  (React 18)              │
                        │                                          │
                        │  ┌──────────┐  ┌──────────┐             │
                        │  │ Landing  │  │  Autopsy │             │
                        │  └──────────┘  └──────────┘             │
                        │  ┌──────────┐  ┌──────────┐             │
                        │  │Simulator │  │  Teams   │             │
                        │  └──────────┘  └──────────┘             │
                        │  ┌──────────┐  ┌──────────┐             │
                        │  │Leaderbd  │  │  Auth    │             │
                        │  └──────────┘  └──────────┘             │
                        │         Vercel CDN                       │
                        └───────────────────┬─────────────────────┘
                                            │  REST API (HTTPS)
                        ┌───────────────────▼─────────────────────┐
                        │         BACKEND  (FastAPI / Python)      │
                        │                                          │
                        │  /api/v1/autopsy  ───► telemetry.py      │
                        │  /api/v1/simulate ───► strategy.py       │
                        │  /api/v1/leaderboard                     │
                        │  /api/v1/heatmap                         │
                        │  /api/v1/report   (Pro only)             │
                        │  /api/v1/auth                            │
                        │                                          │
                        │         Railway / Render                 │
                        └──────┬────────────────────┬─────────────┘
                               │                    │
               ┌───────────────▼──┐    ┌────────────▼──────────────┐
               │  PostgreSQL 16   │    │   FastF1 + Pandas/NumPy   │
               │                  │    │                            │
               │  users           │    │  Session cache (disk)      │
               │  races           │    │  Lap time processing       │
               │  telemetry_cache │    │  Degradation curves        │
               │  race_autopsies  │    │  Blame attribution model   │
               │  simulations     │    │  Optimal window model      │
               │  strategy_dec.   │    │                            │
               │  optimal_models  │    │  Data Sources:             │
               │  leaderboard     │    │  ├── FastF1 v3.4           │
               │                  │    │  ├── Ergast API            │
               │  Railway Postgres│    │  ├── OpenF1               │
               └──────────────────┘    │  └── FIA Timing           │
                                       └───────────────────────────┘
```

## Data Flow — Race Autopsy

```
User selects: year=2023, gp=Monaco, driver=VER
         │
         ▼
GET /api/v1/autopsy/2023/Monaco/VER
         │
         ├─► Check telemetry_cache table
         │       └─► Cache HIT → return cached result
         │       └─► Cache MISS →
         │               ├─► fastf1.get_session(2023, "Monaco", "R").load()
         │               ├─► fastf1.get_session(2023, "Monaco", "Q").load()
         │               ├─► compute_qualifying_cost()
         │               ├─► compute_tyre_degradation_cost()
         │               ├─► compute_pit_execution_cost()
         │               ├─► compute_strategy_error()
         │               ├─► compute_car_pace_deficit()
         │               ├─► compute_incident_impact()
         │               ├─► Store in telemetry_cache + race_autopsies
         │               └─► Return JSON blame report
         │
         ▼
Frontend renders:
  ├── Waterfall chart (animated, per category)
  ├── Blame % progress bars
  ├── AI verdict text
  ├── Race timeline
  └── Raw telemetry tab
```

## Data Flow — Strategy Simulation

```
User starts sim: circuit=Monaco, year=2023, driver=VER
         │
         ▼
Load real telemetry for all other cars (FastF1)
User controls: pit timing, compound, push/save, SC reaction
         │
         ├─► Per lap: user decision → lap time model
         │       ├── base_time (real FastF1 data)
         │       ├── +/- tyre_deg_delta(compound, stint_age)
         │       ├── +/- push_factor
         │       ├── +/- weather_factor
         │       └── Other cars: exact historical lap times
         │
         ├─► Post-race: score_simulation()
         │       ├── pit_timing_score (vs optimal window)
         │       ├── tyre_efficiency_score (vs deg model)
         │       ├── race_outcome_score (position delta)
         │       └── sc_reaction_score (SC opportunities)
         │
         ├─► Store: simulations + strategy_decisions tables
         ├─► Update: optimal_strategy_models (if IQ >= 75)
         └─► Update: leaderboard
```

## Crowdsourced Intelligence Model

```
Every simulation run:
  ├── Score IQ
  ├── IQ >= 75? → "above average" decision
  │       └─► Weighted into OptimalStrategyModel
  │               ├── pit_window_confidence += weight
  │               ├── compound_preference updated
  │               └── undercut_delta refined
  └── IQ < 75? → stored but not weighted into model

Model output (GET /api/v1/optimal/{circuit}/{year}):
  ├── Optimal pit window [lap X–Y] with confidence %
  ├── Recommended compound
  ├── Undercut delta threshold
  └── Sample size (n simulations)
```

## Database Schema (simplified)

```
users ──────────────────────────┐
  id, email, name               │
  strategy_iq, is_pro           │
                                 │
simulations ────────────────────┤
  id, user_id (FK)              │
  circuit, year, driver         │
  strategy_iq                   │
  vs_real_delta, vs_optimal_delta│
                                 │
strategy_decisions ─────────────┘
  id, simulation_id (FK)
  lap_number, decision_type
  decision_value
  is_optimal, time_delta

races
  id, year, gp_name, circuit

telemetry_cache
  id, race_id (FK)
  driver_code
  laps_json (full lap data)

race_autopsies
  id, race_id (FK)
  driver_code
  qualifying_cost .. incident_impact
  total_loss, ai_verdict

optimal_strategy_models
  circuit, year
  pit_window_1_start .. pit_window_2_confidence
  sample_size, model_confidence

leaderboard
  user_id (FK)
  overall_iq, global_rank
```

## Performance Considerations

- **FastF1 cache**: Session data is ~50MB per race. Cached to disk on first load.
- **DB caching**: Computed autopsies cached in `race_autopsies` table (never recomputed).
- **API response times**: Cached autopsy ~50ms. Fresh FastF1 fetch ~8-30s.
- **Scalability**: Add Redis for hot-path caching. Background task queue (Celery/ARQ) for async FastF1 fetches.
