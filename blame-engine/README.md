# 🏎 The Blame Engine

> **The first F1 platform where fans generate intelligence teams actually need.**

A dual-sided Formula 1 strategy intelligence platform powered by real FastF1 telemetry data. Built for F1 fans who want to understand strategy and teams who need crowdsourced decision intelligence.

---

## 🌐 Live Demo

- Frontend: Deploy `blame-engine.jsx` as a React artifact on claude.ai
- API: Deploy `backend/main.py` to Railway/Render

---

## ✨ Core Features

| Feature | Description |
|---|---|
| 🔬 **Race Autopsy** | Quantified time-loss breakdown across 6 categories using FastF1 telemetry |
| 🎮 **Be The Strategist** | Live race simulation with real car data, your decisions |
| 🧠 **Strategy IQ** | Scoring system + global leaderboard + crowdsourced model |
| 🏁 **Team Dashboard** | Pro features: driver comparison, fan heatmaps, API |
| 📊 **Waterfall Charts** | Visual blame attribution with animated charts |
| 🤖 **AI Verdict** | Plain-English summary of what went wrong and by how much |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React/JSX)                  │
│  Landing | Race Autopsy | Simulator | Leaderboard | Pro  │
└─────────────────────┬───────────────────────────────────┘
                      │ REST API
┌─────────────────────▼───────────────────────────────────┐
│               BACKEND (FastAPI / Python)                 │
│  Auth | Autopsy | Simulate | Leaderboard | Reports       │
└──────┬──────────────────────────────────────┬───────────┘
       │                                      │
┌──────▼──────┐                    ┌──────────▼───────────┐
│ PostgreSQL  │                    │  FastF1 + Pandas     │
│  Database   │                    │  Telemetry Engine    │
│  (models)   │                    │  NumPy analysis      │
└─────────────┘                    └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+, Python 3.11+, PostgreSQL 15+

### Frontend
```bash
# The main app is a single React component
# Upload blame-engine.jsx to claude.ai as an artifact
# Or create a React app:
npx create-react-app blame-engine --template typescript
cp blame-engine.jsx src/App.jsx
npm start
```

### Backend
```bash
cd backend/

# Install dependencies
pip install fastapi uvicorn fastf1 pandas numpy sqlalchemy asyncpg python-dotenv

# Set environment variables
cp .env.example .env
# Edit .env with your DATABASE_URL

# Initialize database
python models.py

# Start server
uvicorn main:app --reload --port 8000
```

### Environment Variables
```env
DATABASE_URL=postgresql://user:password@localhost/blameengine
SECRET_KEY=your-secret-key-here
FASTF1_CACHE_DIR=./cache
ENVIRONMENT=development
```

---

## 📡 API Reference

### Authentication
```
POST /api/v1/auth/signup    { email, password, name }
POST /api/v1/auth/login     { email, password }
GET  /api/v1/auth/me        (requires Bearer token)
```

### Race Autopsy
```
GET  /api/v1/autopsy/{year}/{gp}/{driver}
     Returns: blame attribution, total loss, AI verdict

POST /api/v1/autopsy/compare
     Body: { year, gp, drivers: ["VER", "HAM", "NOR"] }
     Returns: side-by-side comparison
```

### Strategy Simulation
```
POST /api/v1/simulate
     Body: { circuit, year, driver, decisions: [...] }
     Returns: IQ score, deltas, leaderboard entry

GET  /api/v1/optimal/{circuit}/{year}
     Returns: crowdsourced optimal strategy model

POST /api/v1/simulate/score
     Body: { simulation_id, iq_score, decisions, position }
```

### Leaderboard & Heatmaps
```
GET  /api/v1/leaderboard?circuit=Monaco&limit=20
GET  /api/v1/heatmap/{circuit}/{year}
GET  /api/v1/telemetry/{year}/{gp}/{driver}
GET  /api/v1/report/{year}/{gp}/{driver}    (Pro only)
```

---

## 📊 Race Autopsy — How It Works

```python
# FastF1 telemetry processing
import fastf1
session = fastf1.get_session(year, gp, 'R')
session.load()
laps = session.laps.pick_driver(driver)

# 1. Qualifying cost: grid position delta vs optimal
qualifying_cost = compute_qualifying_delta(laps, session)

# 2. Tyre deg: actual vs modelled optimal deg curve
tyre_cost = compute_tyre_delta(laps, deg_model)

# 3. Pit execution: stationary time + traffic on rejoin
pit_cost = compute_pit_delta(laps, session.car_data)

# 4. Strategy error: chosen window vs optimal pit window
strategy_cost = compute_strategy_delta(laps, optimal_model)

# 5. Car pace: isolated fundamental performance gap
pace_cost = compute_pace_deficit(laps, reference_cars)

# 6. Incident: time lost to track incidents
incident_cost = compute_incident_impact(laps, session.race_control_messages)
```

---

## 🧠 Strategy IQ Scoring

| Component | Weight | Description |
|---|---|---|
| Pit Timing | 35% | How close to optimal window |
| Tyre Efficiency | 25% | Compound choice & stint management |
| Race Outcome | 25% | Final position delta vs real |
| SC Reaction | 15% | Safety car opportunity exploitation |

**Score bands:** Elite (85+), Competent (70-84), Needs Work (<70)

---

## 🗄️ Database Models

| Table | Purpose |
|---|---|
| `users` | Auth, IQ scores, preferences |
| `races` | Race metadata, circuit info |
| `telemetry_cache` | FastF1 lap data cache |
| `race_autopsies` | Computed blame attribution |
| `simulations` | User strategy sessions |
| `strategy_decisions` | Per-lap user decisions |
| `optimal_strategy_models` | Crowdsourced aggregated model |
| `leaderboard` | Global IQ rankings |

---

## 🌐 Deployment

### Frontend → Vercel
```bash
vercel --prod
```

### Backend → Railway
```bash
railway init
railway up
railway variables set DATABASE_URL=...
```

### Backend → Render
```yaml
# render.yaml
services:
  - type: web
    name: blameengine-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 📦 Requirements

**Backend (`requirements.txt`)**
```
fastapi==0.110.0
uvicorn==0.28.0
fastf1==3.4.0
pandas==2.2.0
numpy==1.26.4
sqlalchemy==2.0.28
asyncpg==0.29.0
python-dotenv==1.0.1
python-jose==3.3.0
passlib==1.7.4
pydantic==2.6.3
```

**Frontend dependencies** (if building standalone React app)
```
react 18+
recharts (for charts)
```

---

## 🔮 Roadmap

- [ ] Live race mode (OpenF1 streaming API)
- [ ] Head-to-head simulator (multiplayer)
- [ ] Circuit-specific AI models per team
- [ ] Mobile app (React Native)
- [ ] Discord bot integration
- [ ] Team tier API (enterprise pricing)

---

## 📄 License

MIT — free for fan use. Commercial team use requires Pro subscription.

---

*Data sources: FastF1, Ergast API, OpenF1, official FIA timing data.*
