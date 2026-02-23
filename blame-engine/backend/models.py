"""
BlameEngine Database Schema — PostgreSQL + SQLAlchemy
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean,
    DateTime, JSON, ForeignKey, Text, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()


# ─── USERS ────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    password_hash = Column(String(64), nullable=False)
    is_pro = Column(Boolean, default=False)
    strategy_iq = Column(Integer, default=0)
    total_simulations = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    simulations = relationship("Simulation", back_populates="user")
    decisions = relationship("StrategyDecision", back_populates="user")


# ─── RACE METADATA ────────────────────────────────────────────────────────────
class Race(Base):
    __tablename__ = "races"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    year = Column(Integer, nullable=False, index=True)
    gp_name = Column(String(100), nullable=False)
    circuit = Column(String(100), nullable=False)
    round_number = Column(Integer)
    race_date = Column(DateTime)
    total_laps = Column(Integer)
    weather = Column(String(50))
    track_temp = Column(Float)

    __table_args__ = (UniqueConstraint("year", "gp_name"),)

    telemetry_cache = relationship("TelemetryCache", back_populates="race")
    autopsies = relationship("RaceAutopsy", back_populates="race")


# ─── TELEMETRY CACHE ─────────────────────────────────────────────────────────
class TelemetryCache(Base):
    __tablename__ = "telemetry_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    race_id = Column(UUID(as_uuid=True), ForeignKey("races.id"), nullable=False)
    driver_code = Column(String(3), nullable=False, index=True)
    laps_json = Column(JSON, nullable=False)  # Array of lap dicts
    sectors_json = Column(JSON)
    tyre_stints_json = Column(JSON)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    fastf1_version = Column(String(20))
    data_quality = Column(String(20), default="HIGH")

    race = relationship("Race", back_populates="telemetry_cache")

    __table_args__ = (UniqueConstraint("race_id", "driver_code"),)


# ─── RACE AUTOPSY ─────────────────────────────────────────────────────────────
class RaceAutopsy(Base):
    __tablename__ = "race_autopsies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    race_id = Column(UUID(as_uuid=True), ForeignKey("races.id"), nullable=False)
    driver_code = Column(String(3), nullable=False)
    
    # Blame attribution (seconds lost)
    qualifying_cost = Column(Float, default=0.0)
    tyre_degradation = Column(Float, default=0.0)
    pit_execution = Column(Float, default=0.0)
    strategy_error = Column(Float, default=0.0)
    car_pace_deficit = Column(Float, default=0.0)
    incident_impact = Column(Float, default=0.0)
    total_loss = Column(Float, default=0.0)
    
    # Position
    actual_position = Column(Integer)
    optimal_position = Column(Integer)
    position_delta = Column(Integer)  # positions lost vs optimal
    
    # AI verdict
    ai_verdict = Column(Text)
    primary_cause = Column(String(50))
    confidence = Column(Float, default=0.89)
    
    computed_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(20), default="1.0")

    race = relationship("Race", back_populates="autopsies")


# ─── SIMULATIONS ──────────────────────────────────────────────────────────────
class Simulation(Base):
    __tablename__ = "simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    race_id = Column(UUID(as_uuid=True), ForeignKey("races.id"), nullable=True)
    driver_code = Column(String(3), nullable=False)
    circuit = Column(String(100), nullable=False)
    year = Column(Integer, nullable=False)
    
    # Performance metrics
    strategy_iq = Column(Integer)
    pit_timing_score = Column(Integer)
    tyre_efficiency_score = Column(Integer)
    race_outcome_score = Column(Integer)
    sc_reaction_score = Column(Integer)
    
    # Race result
    final_position = Column(Integer)
    total_race_time = Column(Float)
    vs_real_driver_delta = Column(Float)
    vs_optimal_delta = Column(Float)
    
    # Strategy summary
    num_pit_stops = Column(Integer)
    compounds_used = Column(JSON)  # ["Soft", "Medium"]
    pit_laps = Column(JSON)        # [23, 45]
    push_laps_pct = Column(Float)  # percentage of laps on push mode
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed = Column(Boolean, default=False)

    user = relationship("User", back_populates="simulations")
    decisions = relationship("StrategyDecision", back_populates="simulation")


# ─── STRATEGY DECISIONS ───────────────────────────────────────────────────────
class StrategyDecision(Base):
    __tablename__ = "strategy_decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id = Column(UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    circuit = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    
    lap_number = Column(Integer, nullable=False)
    decision_type = Column(String(50), nullable=False)  # "pit", "compound", "push", "sc_react"
    decision_value = Column(String(50), nullable=False)  # "box", "Medium", "push", "stay_out"
    
    # Outcome scoring
    is_optimal = Column(Boolean)
    time_delta_vs_optimal = Column(Float)
    contributed_to_iq = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    simulation = relationship("Simulation", back_populates="decisions")
    user = relationship("User", back_populates="decisions")


# ─── OPTIMAL STRATEGY MODEL ───────────────────────────────────────────────────
class OptimalStrategyModel(Base):
    __tablename__ = "optimal_strategy_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    circuit = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False)
    
    # Crowdsourced pit windows
    pit_window_1_start = Column(Integer)
    pit_window_1_end = Column(Integer)
    pit_window_1_confidence = Column(Float)
    pit_window_1_compound = Column(String(20))
    
    pit_window_2_start = Column(Integer)
    pit_window_2_end = Column(Integer)
    pit_window_2_confidence = Column(Float)
    pit_window_2_compound = Column(String(20))
    
    # Strategy insights
    undercut_delta = Column(Float)
    sc_pit_threshold = Column(Float)
    
    # Model metadata
    sample_size = Column(Integer, default=0)
    above_average_decisions = Column(Integer, default=0)
    model_confidence = Column(Float, default=0.5)
    last_updated = Column(DateTime, default=datetime.utcnow)
    version = Column(Integer, default=1)

    __table_args__ = (UniqueConstraint("circuit", "year"),)


# ─── LEADERBOARD ─────────────────────────────────────────────────────────────
class LeaderboardEntry(Base):
    __tablename__ = "leaderboard"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    overall_iq = Column(Integer, default=0)
    best_iq = Column(Integer, default=0)
    average_iq = Column(Float, default=0.0)
    total_simulations = Column(Integer, default=0)
    winning_simulations = Column(Integer, default=0)  # beat real driver
    best_circuit = Column(String(100))
    
    global_rank = Column(Integer, index=True)
    weekly_rank = Column(Integer)
    monthly_rank = Column(Integer)
    
    last_updated = Column(DateTime, default=datetime.utcnow)


# ─── DATABASE SETUP ───────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@localhost/blameengine")

engine = create_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

import os

if __name__ == "__main__":
    init_db()
    print("✅ Database tables created")
