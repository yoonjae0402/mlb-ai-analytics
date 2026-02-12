"""SQLAlchemy ORM models for MLB Analytics."""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, SmallInteger, BigInteger, String, Float, Boolean, Date, DateTime,
    Text, ForeignKey, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlb_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    abbreviation = Column(String(10), nullable=False)
    league = Column(String(5))  # AL or NL
    division = Column(String(20))  # East, Central, West
    logo_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    players = relationship("Player", back_populates="team_rel", lazy="dynamic")


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (
        Index("ix_players_team_level", "team", "current_level"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlb_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    team = Column(String(100))
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    position = Column(String(20))
    bats = Column(String(5))
    throws = Column(String(5))
    headshot_url = Column(String(500))
    current_level = Column(String(10), default="MLB")  # MLB, AAA, AA, A+, A
    prospect_rank = Column(Integer, nullable=True)  # Top prospect ranking (1-100+)
    age = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    team_rel = relationship("Team", back_populates="players")
    stats = relationship("PlayerStat", back_populates="player", lazy="dynamic")
    predictions = relationship("Prediction", back_populates="player", lazy="dynamic")


class PlayerStat(Base):
    __tablename__ = "player_stats"
    __table_args__ = (
        UniqueConstraint("player_id", "game_date", name="uq_player_game_date"),
        Index("ix_player_stats_player_date", "player_id", "game_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_date = Column(Date, nullable=False)
    level = Column(String(10), default="MLB")  # MLB, AAA, AA, etc.

    # Core batting stats
    batting_avg = Column(Float)
    obp = Column(Float)
    slg = Column(Float)
    woba = Column(Float)

    # Statcast metrics
    barrel_rate = Column(Float)
    exit_velo = Column(Float)
    hard_hit_rate = Column(Float)
    launch_angle = Column(Float)

    # Plate discipline
    k_rate = Column(Float)
    bb_rate = Column(Float)

    # Speed / other
    sprint_speed = Column(Float)
    park_factor = Column(Float)

    # Game-level counting stats (targets) — SmallInteger saves space
    hits = Column(SmallInteger, default=0)
    home_runs = Column(SmallInteger, default=0)
    rbi = Column(SmallInteger, default=0)
    walks = Column(SmallInteger, default=0)

    # Additional context
    at_bats = Column(SmallInteger, default=0)
    plate_appearances = Column(SmallInteger, default=0)
    doubles = Column(SmallInteger, default=0)
    triples = Column(SmallInteger, default=0)
    strikeouts = Column(SmallInteger, default=0)
    stolen_bases = Column(SmallInteger, default=0)

    # Pitching stats (only populated for pitchers)
    innings_pitched = Column(Float, nullable=True)
    earned_runs = Column(SmallInteger, nullable=True)

    player = relationship("Player", back_populates="stats")


class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlb_game_id = Column(Integer, unique=True, nullable=False, index=True)
    game_date = Column(Date, nullable=False, index=True)
    away_team = Column(String(100), nullable=False)
    home_team = Column(String(100), nullable=False)
    away_score = Column(Integer)
    home_score = Column(Integer)
    status = Column(String(50))
    venue = Column(String(200))
    away_probable_pitcher = Column(String(200))
    home_probable_pitcher = Column(String(200))
    game_datetime = Column(DateTime, nullable=True)

    predictions = relationship("Prediction", back_populates="game", lazy="dynamic")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)  # "lstm" or "xgboost"
    version = Column(String(50), nullable=False)
    hyperparams_json = Column(JSON)
    train_mse = Column(Float)
    val_mse = Column(Float)
    train_r2 = Column(Float)
    val_r2 = Column(Float)
    trained_at = Column(DateTime, default=datetime.utcnow)
    checkpoint_path = Column(String(500))

    predictions = relationship("Prediction", back_populates="model_version", lazy="dynamic")
    metrics = relationship("ModelMetric", back_populates="model_version", lazy="dynamic")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_player_date", "player_id", "created_at"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    predicted_hits = Column(Float)
    predicted_hr = Column(Float)
    predicted_rbi = Column(Float)
    predicted_walks = Column(Float)
    confidence = Column(Float, nullable=True)  # Model confidence score
    created_at = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player", back_populates="predictions")
    game = relationship("Game", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    result = relationship("PredictionResult", back_populates="prediction", uselist=False)


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), unique=True, nullable=False)
    actual_hits = Column(Integer)
    actual_hr = Column(Integer)
    actual_rbi = Column(Integer)
    actual_walks = Column(Integer)
    mse = Column(Float)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="result")


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    metric_date = Column(Date, nullable=False)
    daily_mse = Column(Float)
    daily_mae = Column(Float)
    n_predictions = Column(Integer, default=0)

    model_version = relationship("ModelVersion", back_populates="metrics")
