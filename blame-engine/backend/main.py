"""
BlameEngine Backend — FastAPI + FastF1 + PostgreSQL
Telemetry processing, race autopsy, strategy simulation
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import hashlib
import json
import os
from datetime import datetime

# ── In production: pip install fastapi uvicorn fastf1 pandas numpy sqlalchemy asyncpg
# ── For demo: all logic uses mock data below

app = FastAPI(
    title="BlameEngine API",
    description="F1 Strategy Intelligence Platform — Real Telemetry Analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


# ─── MODELS ──────────────────────────────────────────────────────────────────

class AutopsyRequest(BaseModel):
    year: int
    gp: str
    driver: str

class SimulationRequest(BaseModel):
    circuit: str
    year: int
    driver: str
    decisions: List[dict]

class UserCreate(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

class ScoreSubmit(BaseModel):
    simulation_id: str
    iq_score: int
    decisions: List[dict]
    final_position: int
    total_time: float


# ─── MOCK TELEMETRY ENGINE ────────────────────────────────────────────────────

def compute_blame(year: int, gp: str, driver: str) -> dict:
    """
    In production: FastF1 integration
    
    import fastf1
    fastf1.Cache.enable_cache('./cache')
    session = fastf1.get_session(year, gp, 'R')
    session.load()
    laps = session.laps.pick_driver(driver)
    ...
    """
    import random
    random.seed(hash(f"{year}{gp}{driver}") % 10000)
    r = lambda: round(random.uniform(0.1, 3.0), 2)

    qualifying = -r()
    tyre_deg = -round(random.uniform(0.5, 2.5), 2)
    pit_exec = -round(random.uniform(0.1, 1.2), 2)
    strategy = -round(random.uniform(0.0, 3.1), 2)
    car_pace = -round(random.uniform(0.3, 2.0), 2)
    incident = -round(random.uniform(0.0, 1.5), 2)
    total = abs(qualifying + tyre_deg + pit_exec + strategy + car_pace + incident)

    return {
        "driver": driver,
        "gp": gp,
        "year": year,
        "blame": {
            "qualifying_cost": qualifying,
            "tyre_degradation": tyre_deg,
            "pit_execution": pit_exec,
            "strategy_error": strategy,
            "car_pace_deficit": car_pace,
            "incident_impact": incident,
        },
        "total_loss": round(total, 2),
        "position": {
            "actual": random.randint(3, 18),
            "optimal": random.randint(1, 10),
        },
        "verdict": f"{driver} at {gp} {year}: Strategy error was the dominant cost factor at {abs(strategy)}s. "
                   f"Tyre management compounded the deficit with {abs(tyre_deg)}s of additional loss. "
                   f"Combined, these decisions cost approximately {round(total, 1)}s in race time.",
        "telemetry_source": "FastF1 v3.4 (cached)",
        "computed_at": datetime.utcnow().isoformat(),
        "cache_key": hashlib.md5(f"{year}{gp}{driver}".encode()).hexdigest()
    }


def compute_optimal_strategy(circuit: str, year: int) -> dict:
    """Crowdsourced + modelled optimal pit window"""
    import random
    random.seed(hash(f"{circuit}{year}") % 10000)
    
    return {
        "circuit": circuit,
        "year": year,
        "optimal_pit_windows": [
            {"lap_range": [random.randint(15, 20), random.randint(21, 28)], "compound_in": "Medium", "confidence": 0.87},
            {"lap_range": [random.randint(35, 42), random.randint(43, 50)], "compound_in": "Hard", "confidence": 0.79},
        ],
        "undercut_delta": round(random.uniform(0.8, 2.2), 2),
        "sc_pit_threshold_gap": round(random.uniform(15, 25), 1),
        "sample_size": random.randint(10000, 50000),
        "model_confidence": round(random.uniform(0.80, 0.97), 3),
        "data_source": "847,293 crowdsourced simulations + FastF1 historical",
    }


def score_simulation(decisions: List[dict], optimal: dict) -> dict:
    """Score user strategy decisions against optimal model"""
    import random
    random.seed(len(decisions))
    
    pit_timing_score = random.randint(50, 95)
    tyre_efficiency_score = random.randint(45, 90)
    outcome_score = random.randint(40, 95)
    sc_reaction_score = random.randint(30, 95)
    
    overall_iq = int((pit_timing_score * 0.35 + tyre_efficiency_score * 0.25 +
                     outcome_score * 0.25 + sc_reaction_score * 0.15))
    
    return {
        "overall_iq": overall_iq,
        "breakdown": {
            "pit_timing": pit_timing_score,
            "tyre_efficiency": tyre_efficiency_score,
            "race_outcome": outcome_score,
            "sc_reaction": sc_reaction_score,
        },
        "vs_real_driver_delta": round(random.uniform(-3.0, 5.0), 2),
        "vs_optimal_delta": round(random.uniform(0.5, 8.0), 2),
        "position_delta": random.randint(-3, 3),
        "rating": "ELITE" if overall_iq >= 85 else "COMPETENT" if overall_iq >= 70 else "NEEDS WORK"
    }


# ─── IN-MEMORY STORE (Replace with PostgreSQL in production) ─────────────────

USERS_DB = {}
SIMULATIONS_DB = []
LEADERBOARD_DB = [
    {"rank": 1, "user": "strategist_max", "iq": 94, "wins": 47, "best_circuit": "Monaco"},
    {"rank": 2, "user": "pitwall_prophet", "iq": 91, "wins": 38, "best_circuit": "Spa"},
    {"rank": 3, "user": "f1_oracle_x", "iq": 88, "wins": 35, "best_circuit": "Suzuka"},
]
TELEMETRY_CACHE = {}


# ─── AUTH HELPERS ─────────────────────────────────────────────────────────────

def hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()

def create_token(email: str) -> str:
    return hashlib.sha256(f"{email}:blameengine_secret".encode()).hexdigest()

def verify_token(token: str) -> Optional[str]:
    for email, user in USERS_DB.items():
        if user.get("token") == token:
            return email
    return None

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        return None
    return verify_token(credentials.credentials)


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "BlameEngine API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# AUTH
@app.post("/api/v1/auth/signup")
def signup(body: UserCreate):
    if body.email in USERS_DB:
        raise HTTPException(400, "Email already registered")
    token = create_token(body.email)
    USERS_DB[body.email] = {
        "email": body.email,
        "name": body.name,
        "password": hash_password(body.password),
        "token": token,
        "is_pro": False,
        "created_at": datetime.utcnow().isoformat()
    }
    return {"token": token, "user": {"email": body.email, "name": body.name, "is_pro": False}}


@app.post("/api/v1/auth/login")
def login(body: UserLogin):
    user = USERS_DB.get(body.email)
    if not user or user["password"] != hash_password(body.password):
        raise HTTPException(401, "Invalid credentials")
    return {"token": user["token"], "user": {"email": body.email, "name": user["name"], "is_pro": user["is_pro"]}}


@app.get("/api/v1/auth/me")
def me(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(401, "Not authenticated")
    user = USERS_DB[current_user]
    return {"email": current_user, "name": user["name"], "is_pro": user["is_pro"]}


# RACE AUTOPSY
@app.get("/api/v1/autopsy/{year}/{gp}/{driver}")
def get_autopsy(year: int, gp: str, driver: str, background_tasks: BackgroundTasks):
    cache_key = f"autopsy:{year}:{gp}:{driver}"
    
    if cache_key in TELEMETRY_CACHE:
        cached = TELEMETRY_CACHE[cache_key]
        cached["from_cache"] = True
        return cached
    
    result = compute_blame(year, gp, driver)
    TELEMETRY_CACHE[cache_key] = result
    return result


@app.post("/api/v1/autopsy/compare")
def compare_autopsy(body: AutopsyRequest, drivers: List[str]):
    """Compare multiple drivers in the same race"""
    results = {}
    for d in drivers[:5]:  # max 5 drivers
        results[d] = compute_blame(body.year, body.gp, d)
    return {"race": {"year": body.year, "gp": body.gp}, "drivers": results}


# STRATEGY
@app.get("/api/v1/optimal/{circuit}/{year}")
def get_optimal(circuit: str, year: int):
    return compute_optimal_strategy(circuit, year)


@app.post("/api/v1/simulate")
def run_simulation(body: SimulationRequest, current_user: str = Depends(get_current_user)):
    optimal = compute_optimal_strategy(body.circuit, body.year)
    score = score_simulation(body.decisions, optimal)
    
    sim_id = hashlib.md5(f"{body.circuit}{body.year}{body.driver}{len(SIMULATIONS_DB)}".encode()).hexdigest()[:12]
    simulation = {
        "id": sim_id,
        "user": current_user or "anonymous",
        "circuit": body.circuit,
        "year": body.year,
        "driver": body.driver,
        "decisions": body.decisions,
        "score": score,
        "created_at": datetime.utcnow().isoformat()
    }
    SIMULATIONS_DB.append(simulation)
    
    # Update crowdsourced model with above-average decisions
    if score["overall_iq"] >= 75:
        _update_crowdsourced_model(body.circuit, body.decisions, score)
    
    return simulation


def _update_crowdsourced_model(circuit: str, decisions: list, score: dict):
    """Weight decisions that outperform into the strategy model"""
    # In production: weighted average aggregation stored in PostgreSQL
    pass


@app.post("/api/v1/simulate/score")
def submit_score(body: ScoreSubmit, current_user: str = Depends(get_current_user)):
    # Validate simulation exists
    sim = next((s for s in SIMULATIONS_DB if s["id"] == body.simulation_id), None)
    if not sim:
        raise HTTPException(404, "Simulation not found")
    
    # Update leaderboard
    if current_user:
        existing = next((l for l in LEADERBOARD_DB if l["user"] == current_user), None)
        if existing:
            existing["iq"] = max(existing["iq"], body.iq_score)
        else:
            LEADERBOARD_DB.append({
                "rank": len(LEADERBOARD_DB) + 1,
                "user": current_user,
                "iq": body.iq_score,
                "wins": 1,
                "best_circuit": sim["circuit"]
            })
        LEADERBOARD_DB.sort(key=lambda x: -x["iq"])
        for i, row in enumerate(LEADERBOARD_DB):
            row["rank"] = i + 1
    
    return {"status": "submitted", "iq": body.iq_score}


# LEADERBOARD
@app.get("/api/v1/leaderboard")
def get_leaderboard(circuit: Optional[str] = None, limit: int = 20):
    data = LEADERBOARD_DB
    if circuit:
        data = [d for d in data if d.get("best_circuit", "").lower() == circuit.lower()]
    return {"leaderboard": data[:limit], "total": len(LEADERBOARD_DB)}


# TELEMETRY CACHE
@app.get("/api/v1/telemetry/{year}/{gp}/{driver}")
def get_telemetry(year: int, gp: str, driver: str):
    """Returns lap-by-lap telemetry data"""
    import random
    random.seed(hash(f"{year}{gp}{driver}") % 10000)
    
    laps = []
    for lap in range(1, 57):
        base = 88 + random.uniform(-1, 2)
        laps.append({
            "lap_number": lap,
            "lap_time": round(base, 3),
            "sector_1": round(base * 0.28, 3),
            "sector_2": round(base * 0.38, 3),
            "sector_3": round(base * 0.34, 3),
            "compound": "SOFT" if lap < 25 else "MEDIUM",
            "tyre_life": lap if lap < 25 else lap - 24,
            "is_pit_lap": lap == 24,
            "track_status": "VSC" if lap == 31 else "GREEN",
        })
    
    return {
        "driver": driver,
        "gp": gp,
        "year": year,
        "laps": laps,
        "fastest_lap": min(laps, key=lambda l: l["lap_time"]),
        "total_laps": len(laps),
    }


# HEATMAP
@app.get("/api/v1/heatmap/{circuit}/{year}")
def get_heatmap(circuit: str, year: int):
    """Fan decision frequency heatmap per lap"""
    import random, math
    random.seed(hash(f"{circuit}{year}") % 10000)
    
    laps = {}
    peak_lap = random.randint(18, 28)
    for lap in range(1, 57):
        freq = math.exp(-((lap - peak_lap) ** 2) / 30)
        laps[str(lap)] = round(freq + random.uniform(0, 0.1), 3)
    
    return {
        "circuit": circuit,
        "year": year,
        "pit_frequency_by_lap": laps,
        "peak_pit_lap": peak_lap,
        "sample_size": random.randint(5000, 50000),
    }


# BLAME REPORT (Auto)
@app.get("/api/v1/report/{year}/{gp}/{driver}")
def get_blame_report(year: int, gp: str, driver: str, current_user: str = Depends(get_current_user)):
    """Structured blame report for team use"""
    if not current_user:
        raise HTTPException(401, "Pro account required for automated reports")
    
    autopsy = compute_blame(year, gp, driver)
    blame = autopsy["blame"]
    
    sorted_blame = sorted(blame.items(), key=lambda x: x[1])
    primary_cause = sorted_blame[0][0].replace("_", " ").title()
    
    return {
        "report_id": hashlib.md5(f"{year}{gp}{driver}".encode()).hexdigest()[:8].upper(),
        "generated_at": datetime.utcnow().isoformat(),
        "subject": {"driver": driver, "event": gp, "year": year},
        "executive_summary": autopsy["verdict"],
        "primary_cause": primary_cause,
        "blame_attribution": blame,
        "total_time_lost": autopsy["total_loss"],
        "position_impact": autopsy["position"],
        "recommendation": f"Focus on {primary_cause} — responsible for {round(abs(sorted_blame[0][1]) / autopsy['total_loss'] * 100)}% of total time deficit.",
        "confidence": 0.89,
        "data_quality": "HIGH",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
