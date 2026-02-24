"""
BlameEngine Telemetry Engine — ML Edition
==========================================
Uses trained XGBoost model for all lap time predictions.
Falls back gracefully to polynomial curves if model not found.
Falls back to mock data if FastF1 not available.

The ML pipeline:
  1. LapTimeModel.predict(features) → predicted lap time delta
  2. CounterfactualEngine.simulate(race, alt_strategy) → alternative timeline
  3. blame = actual_time - counterfactual_time per factor
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List

CACHE_DIR  = os.environ.get("FASTF1_CACHE_DIR", "./cache")
MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# ── Load FastF1 ───────────────────────────────────────────────────────────────
try:
    import fastf1
    fastf1.Cache.enable_cache(CACHE_DIR)
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("⚠️  FastF1 not installed. Using mock data.")

# ── Load ML Model ─────────────────────────────────────────────────────────────
ML_MODEL   = None
ML_SCALER  = None
ML_META    = None
DEG_CURVES = None

try:
    import joblib
    model_path  = Path(MODELS_DIR) / "lap_time_model.joblib"
    scaler_path = Path(MODELS_DIR) / "feature_scaler.joblib"
    meta_path   = Path(MODELS_DIR) / "model_metadata.json"
    deg_path    = Path(MODELS_DIR) / "deg_curves.json"

    if model_path.exists():
        ML_MODEL  = joblib.load(model_path)
        ML_SCALER = joblib.load(scaler_path)
        with open(meta_path) as f:
            ML_META = json.load(f)
        with open(deg_path) as f:
            DEG_CURVES = json.load(f)
        print(f"✅ ML Model loaded — MAE={ML_META['mae_seconds']}s R²={ML_META['r2_score']}")
    else:
        print("⚠️  No ML model found. Run ml_pipeline.py first.")
except Exception as e:
    print(f"⚠️  ML model load failed: {e}")


# ─── ML LAP TIME PREDICTOR ────────────────────────────────────────────────────

COMPOUND_CODE = {"WET":0,"INTERMEDIATE":1,"HARD":2,"MEDIUM":3,"SOFT":4}
CIRCUIT_CLASSES = ML_META["circuit_classes"] if ML_META else []

def predict_lap_delta(
    tyre_age: float,
    compound: str,
    fuel_load: float,
    lap_number: float,
    track_temp: float,
    driver_rank: float,
    circuit: str,
    season: int,
) -> float:
    """
    Use XGBoost to predict lap time delta (seconds vs circuit median).
    
    This is the core ML inference call. The model learned from thousands
    of real laps how tyre age, compound, fuel, and temperature combine
    to affect lap time.
    
    Returns: float (seconds above/below median; positive = slower)
    """
    if ML_MODEL is None or ML_SCALER is None or ML_META is None:
        # Fallback: simple polynomial approximation
        return _polynomial_lap_delta(tyre_age, compound, fuel_load)

    # Encode circuit
    try:
        circuit_code = float(CIRCUIT_CLASSES.index(circuit))
    except ValueError:
        circuit_code = float(len(CIRCUIT_CLASSES) // 2)  # unknown circuit → midfield

    max_age_map = {"SOFT":25,"MEDIUM":35,"HARD":50,"INTERMEDIATE":30,"WET":40}
    cliff_map   = {"SOFT":20,"MEDIUM":30,"HARD":45,"INTERMEDIATE":25,"WET":35}
    comp_upper  = compound.upper()

    tyre_life_pct = tyre_age / max_age_map.get(comp_upper, 35)
    past_cliff    = int(tyre_age > cliff_map.get(comp_upper, 30))
    compound_code = float(COMPOUND_CODE.get(comp_upper, 3))

    features = np.array([[
        tyre_age,
        tyre_life_pct,
        past_cliff,
        compound_code,
        fuel_load,
        lap_number,
        track_temp,
        driver_rank,
        circuit_code,
        float(season),
    ]])

    features_scaled = ML_SCALER.transform(features)
    delta = float(ML_MODEL.predict(features_scaled)[0])
    return delta


def _polynomial_lap_delta(tyre_age: float, compound: str, fuel_load: float) -> float:
    """Fallback when ML model not available — simple polynomial deg curve."""
    deg_per_lap = {"SOFT":0.09,"MEDIUM":0.055,"HARD":0.03,"INTERMEDIATE":0.07,"WET":0.05}
    rate = deg_per_lap.get(compound.upper(), 0.055)
    fuel_effect = (1.0 - fuel_load) * 0.08  # heavier fuel = slower early
    return tyre_age * rate + fuel_effect


# ─── COUNTERFACTUAL STRATEGY ENGINE ──────────────────────────────────────────

def simulate_strategy(
    race_laps: pd.DataFrame,
    driver: str,
    alt_pit_lap: int,
    circuit: str,
    season: int,
    track_temp: float = 35.0,
) -> Dict:
    """
    Counterfactual simulation: what would have happened with a different pit lap?

    Method:
    1. Take actual race stint structure
    2. Replace pit lap with alt_pit_lap
    3. Re-predict lap times for each lap using ML model
    4. Compare total race time: actual vs counterfactual
    5. Delta = strategy gain or loss from the decision

    This is real causal inference — not just "their laps were slow",
    but "if they had pitted lap 28 instead of 32, here's the exact time delta."
    """
    try:
        driver_laps = race_laps[race_laps["Driver"] == driver].copy()
        if driver_laps.empty:
            return {"delta": 0.0, "method": "no_data"}

        total_laps    = int(driver_laps["LapNumber"].max())
        actual_pit_laps = driver_laps[driver_laps["PitOutTime"].notna()]["LapNumber"].tolist()

        if not actual_pit_laps:
            return {"delta": 0.0, "method": "no_pit_found"}

        actual_pit_lap = int(actual_pit_laps[0])
        driver_rank = 10.0  # default

        # ── Simulate ACTUAL strategy ──────────────────────────────────────────
        actual_time = 0.0
        tyre_age = 0
        compound = "MEDIUM"

        # Get starting compound
        first_lap = driver_laps[driver_laps["LapNumber"] == 1]
        if not first_lap.empty and "Compound" in first_lap.columns:
            compound = str(first_lap.iloc[0]["Compound"]).upper()

        for lap in range(1, total_laps + 1):
            fuel = 1.0 - (lap / total_laps) * 0.95
            delta = predict_lap_delta(
                tyre_age=float(tyre_age),
                compound=compound,
                fuel_load=fuel,
                lap_number=float(lap),
                track_temp=track_temp,
                driver_rank=driver_rank,
                circuit=circuit,
                season=season,
            )
            # Circuit median from metadata or hardcoded fallback
            circuit_median = _get_circuit_median(circuit)
            actual_time += circuit_median + delta

            tyre_age += 1
            if lap == actual_pit_lap:
                tyre_age = 0
                compound = _get_next_compound(driver_laps, actual_pit_lap)

        # ── Simulate COUNTERFACTUAL strategy ─────────────────────────────────
        counter_time = 0.0
        tyre_age = 0
        compound = "MEDIUM"

        first_lap = driver_laps[driver_laps["LapNumber"] == 1]
        if not first_lap.empty and "Compound" in first_lap.columns:
            compound = str(first_lap.iloc[0]["Compound"]).upper()

        for lap in range(1, total_laps + 1):
            fuel = 1.0 - (lap / total_laps) * 0.95
            delta = predict_lap_delta(
                tyre_age=float(tyre_age),
                compound=compound,
                fuel_load=fuel,
                lap_number=float(lap),
                track_temp=track_temp,
                driver_rank=driver_rank,
                circuit=circuit,
                season=season,
            )
            circuit_median = _get_circuit_median(circuit)
            counter_time += circuit_median + delta

            tyre_age += 1
            if lap == alt_pit_lap:
                tyre_age = 0
                compound = _get_next_compound(driver_laps, actual_pit_lap)

        # Positive delta = actual was slower (bad strategy)
        # Negative delta = actual was faster (good strategy)
        time_delta = actual_time - counter_time

        return {
            "actual_pit_lap":   actual_pit_lap,
            "alt_pit_lap":      alt_pit_lap,
            "actual_total_s":   round(actual_time, 3),
            "counter_total_s":  round(counter_time, 3),
            "delta_s":          round(time_delta, 3),
            "method":           "xgboost_counterfactual" if ML_MODEL else "polynomial_counterfactual",
        }

    except Exception as e:
        return {"delta": 0.0, "error": str(e), "method": "failed"}


def _get_circuit_median(circuit: str) -> float:
    """Get circuit median lap time from model metadata."""
    if ML_META and "circuit_classes" in ML_META:
        try:
            idx = ML_META["circuit_classes"].index(circuit)
            return ML_META["circuit_medians"].get(str(idx), 90.0)
        except (ValueError, KeyError):
            pass
    return 90.0  # fallback


def _get_next_compound(driver_laps: pd.DataFrame, pit_lap: int) -> str:
    """Get compound used after a pit stop."""
    try:
        post_pit = driver_laps[driver_laps["LapNumber"] > pit_lap]
        if not post_pit.empty and "Compound" in post_pit.columns:
            return str(post_pit.iloc[0]["Compound"]).upper()
    except Exception:
        pass
    return "HARD"


# ─── BLAME ATTRIBUTION (CAUSAL) ───────────────────────────────────────────────

def compute_strategy_error_ml(
    driver_laps: pd.DataFrame,
    race_laps: pd.DataFrame,
    driver: str,
    circuit: str,
    season: int,
    track_temp: float,
) -> float:
    """
    Real strategy error using counterfactual simulation.

    Instead of "how far from lap 42%" (arbitrary), we:
    1. Find actual pit lap
    2. Simulate 5 alternative pit laps (±3, ±6, ±9 laps)
    3. Find the best alternative
    4. Strategy error = actual total time - best alternative total time
    """
    try:
        pit_laps = driver_laps[driver_laps["PitOutTime"].notna()]["LapNumber"].tolist()
        if not pit_laps:
            return 0.0

        actual_pit = int(pit_laps[0])
        total_laps = int(driver_laps["LapNumber"].max())

        # Test alternative windows
        alternatives = [
            max(3, actual_pit - 9),
            max(3, actual_pit - 6),
            max(3, actual_pit - 3),
            min(total_laps - 5, actual_pit + 3),
            min(total_laps - 5, actual_pit + 6),
            min(total_laps - 5, actual_pit + 9),
        ]

        best_delta = 0.0  # 0 = actual was optimal
        for alt_lap in alternatives:
            result = simulate_strategy(
                race_laps, driver, alt_lap, circuit, season, track_temp
            )
            delta = result.get("delta_s", 0.0)
            if delta > best_delta:  # positive = actual was slower
                best_delta = delta

        return -round(best_delta, 3)  # negative = cost

    except Exception:
        return 0.0


def compute_qualifying_cost(race_session, quali_session, driver: str) -> float:
    try:
        quali_laps  = quali_session.laps.pick_driver(driver)
        driver_best = quali_laps.pick_fastest()["LapTime"].total_seconds()
        pole_time   = quali_session.laps.pick_fastest()["LapTime"].total_seconds()
        gap_to_pole = driver_best - pole_time
        race_laps   = race_session.laps.pick_driver(driver)
        grid_pos    = race_laps.iloc[0].get("GridPosition", 1)
        traffic     = max(0, (grid_pos - 1) * 0.12)
        return -round(gap_to_pole + traffic, 3)
    except Exception:
        return 0.0


def compute_tyre_degradation_cost_ml(
    driver_laps: pd.DataFrame,
    circuit: str,
    season: int,
    track_temp: float,
) -> float:
    """
    Compare actual per-lap degradation vs ML model's prediction of optimal.
    
    For each lap in each stint:
    - Predict what lap time SHOULD be (optimal management)  
    - Subtract actual lap time
    - Excess = driver pushing too hard / wrong management
    """
    try:
        cost = 0.0
        for stint_num in driver_laps["Stint"].unique():
            stint = driver_laps[driver_laps["Stint"] == stint_num].copy()
            stint = stint[stint["LapTime"].notna()]
            if len(stint) < 3:
                continue

            compound = str(stint["Compound"].iloc[0]).upper() if "Compound" in stint else "MEDIUM"
            total_laps = int(driver_laps["LapNumber"].max())

            for _, lap in stint.iterrows():
                lap_num  = int(lap["LapNumber"])
                tyre_age = int(lap["TyreLife"]) if "TyreLife" in lap else 0
                fuel     = 1.0 - (lap_num / total_laps) * 0.95
                actual_t = lap["LapTime"].total_seconds()

                # Predicted optimal time for this tyre state
                circuit_med = _get_circuit_median(circuit)
                predicted_delta = predict_lap_delta(
                    tyre_age=float(tyre_age),
                    compound=compound,
                    fuel_load=fuel,
                    lap_number=float(lap_num),
                    track_temp=track_temp,
                    driver_rank=5.0,  # assume average for prediction
                    circuit=circuit,
                    season=season,
                )
                predicted_t = circuit_med + predicted_delta

                # Excess = how much slower than model prediction
                excess = actual_t - predicted_t
                if excess > 0.3:  # only count genuine excess
                    cost += excess * 0.6  # discount factor (noise reduction)

        return -round(min(cost, 8.0), 3)  # cap at 8s total
    except Exception:
        return 0.0


def compute_pit_execution_cost(laps: pd.DataFrame, session) -> float:
    try:
        pit_laps  = laps[laps["PitOutTime"].notna() | laps["PitInTime"].notna()]
        if pit_laps.empty:
            return 0.0
        total_cost   = 0.0
        team_avg_stop = 2.4
        for _, lap in pit_laps.iterrows():
            if pd.notna(lap.get("PitInTime")) and pd.notna(lap.get("PitOutTime")):
                stop_time = (lap["PitOutTime"] - lap["PitInTime"]).total_seconds()
                excess    = max(0, stop_time - team_avg_stop)
                total_cost += excess
        return -round(total_cost, 3)
    except Exception:
        return 0.0


def compute_car_pace_deficit(race_session, driver: str) -> float:
    try:
        driver_laps = race_session.laps.pick_driver(driver)
        clean = driver_laps[
            (driver_laps["TrackStatus"] == "1") &
            (driver_laps["PitOutTime"].isna()) &
            (driver_laps["PitInTime"].isna())
        ]
        if clean.empty:
            return 0.0
        driver_median = clean["LapTime"].dt.total_seconds().median()
        all_laps      = race_session.laps[
            (race_session.laps["TrackStatus"] == "1") &
            (race_session.laps["PitOutTime"].isna())
        ]
        field_times     = all_laps.groupby("Driver")["LapTime"].median().dt.total_seconds()
        sorted_times    = field_times.sort_values()
        reference       = sorted_times.iloc[3:-3].median()
        deficit         = driver_median - reference
        return -round(max(0, deficit), 3)
    except Exception:
        return 0.0


def compute_incident_impact(race_session, driver: str) -> float:
    try:
        messages    = race_session.race_control_messages
        driver_laps = race_session.laps.pick_driver(driver)
        total_cost  = 0.0
        yellows     = messages[messages["Message"].str.contains("YELLOW", na=False)]
        total_cost += len(yellows) * 0.08
        sc_events   = messages[messages["Message"].str.contains("SAFETY CAR DEPLOYED", na=False)]
        if not sc_events.empty:
            pit_laps = driver_laps[driver_laps["PitOutTime"].notna()]["LapNumber"].tolist()
            for _, sc in sc_events.iterrows():
                matching = driver_laps[driver_laps["Time"] >= sc["Time"]]
                if not matching.empty:
                    sc_lap = matching.iloc[0]["LapNumber"]
                    if not any(abs(p - sc_lap) <= 2 for p in pit_laps):
                        total_cost += 1.2  # missed free pit
        return -round(total_cost, 3)
    except Exception:
        return 0.0


def compute_optimal_position(
    actual_pos: int,
    total_loss_seconds: float,
    car_pace_deficit: float,
    race_session,
    driver: str,
) -> int:
    try:
        recoverable       = max(0, total_loss_seconds - abs(car_pace_deficit))
        positions_recover = int(recoverable / 2.5)
        car_floor         = max(1, int(abs(car_pace_deficit) / 0.4) + 1)
        raw_optimal       = actual_pos - positions_recover
        optimal           = max(raw_optimal, car_floor)
        optimal           = min(optimal, actual_pos)  # never worse than actual
        return max(1, optimal)
    except Exception:
        return actual_pos


# ─── FULL AUTOPSY ─────────────────────────────────────────────────────────────

def get_session(year: int, gp: str, session_type: str = "R"):
    if not FASTF1_AVAILABLE:
        raise RuntimeError("FastF1 not available")
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, laps=True, weather=True)
    return session


def full_autopsy(year: int, gp: str, driver: str) -> dict:
    """
    Complete race autopsy using real FastF1 data + XGBoost ML model.

    Pipeline:
    1. Load FastF1 race + quali sessions
    2. Extract driver lap data
    3. For each blame factor, run real computation:
       - Qualifying cost: gap to pole × traffic model
       - Tyre degradation: actual laps vs ML model predictions
       - Pit execution: stationary time vs team benchmark
       - Strategy error: counterfactual simulation (actual pit vs optimal pit)
       - Car pace: driver median vs midfield reference pace
       - Incident impact: SC/yellow flag time cost model
    4. Sum to total loss, compute optimal position
    """
    if not FASTF1_AVAILABLE:
        from main import compute_blame
        return compute_blame(year, gp, driver)

    try:
        race  = get_session(year, gp, "R")
        quali = get_session(year, gp, "Q")

        driver_laps = race.laps.pick_driver(driver)
        circuit     = race.event["EventName"]

        # Get track temp
        track_temp = 35.0
        try:
            if not race.weather_data.empty and "TrackTemp" in race.weather_data.columns:
                track_temp = float(race.weather_data["TrackTemp"].median())
        except Exception:
            pass

        # ── Compute all blame factors ─────────────────────────────────────────
        qualifying  = compute_qualifying_cost(race, quali, driver)

        tyre_deg = compute_tyre_degradation_cost_ml(
            driver_laps, circuit, year, track_temp
        )

        pit_exec    = compute_pit_execution_cost(driver_laps, race)

        strategy = compute_strategy_error_ml(
            driver_laps, race.laps, driver, circuit, year, track_temp
        )

        car_pace    = compute_car_pace_deficit(race, driver)
        incident    = compute_incident_impact(race, driver)

        total_loss  = abs(qualifying + tyre_deg + pit_exec + strategy + car_pace + incident)

        blame = {
            "qualifying_cost": qualifying,
            "tyre_degradation": tyre_deg,
            "pit_execution":    pit_exec,
            "strategy_error":   strategy,
            "car_pace_deficit": car_pace,
            "incident_impact":  incident,
        }

        primary_cause = min(blame, key=blame.get)
        primary_pct   = round(abs(blame[primary_cause]) / total_loss * 100) if total_loss > 0 else 0
        actual_pos    = int(driver_laps.iloc[-1].get("Position", 10))
        optimal_pos   = compute_optimal_position(actual_pos, total_loss, car_pace, race, driver)
        positions_lost = actual_pos - optimal_pos

        model_info = (
            f"XGBoost (MAE={ML_META['mae_seconds']}s, R²={ML_META['r2_score']})"
            if ML_MODEL else "Polynomial fallback"
        )

        return {
            "driver":       driver,
            "gp":           gp,
            "year":         year,
            "blame":        blame,
            "total_loss":   round(total_loss, 3),
            "primary_cause": primary_cause,
            "position": {
                "actual":         actual_pos,
                "optimal":        optimal_pos,
                "positions_lost": positions_lost,
            },
            "verdict": (
                f"{driver} at {year} {gp}: "
                f"{primary_cause.replace('_',' ').title()} was the primary cost "
                f"({primary_pct}% of {round(total_loss,1)}s total deficit). "
                f"Counterfactual simulation shows optimal pit timing could have saved "
                f"{abs(strategy):.2f}s. With clean execution, P{optimal_pos} was achievable "
                f"vs actual P{actual_pos}."
            ),
            "telemetry_source": "FastF1 (live)",
            "ml_model":         model_info,
            "data_quality":     "HIGH",
            "track_temp":       track_temp,
        }

    except Exception as e:
        return {"error": str(e), "fallback": "mock_data"}