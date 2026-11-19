import os
from sqlalchemy import Boolean, Column, Float, Integer, String, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

DEFAULT_DB_URL = os.getenv("IOT_DB_URL", "sqlite:///warehouse.db")

Base = declarative_base()


class TelemetryRecord(Base):
    __tablename__ = "telemetry_stream"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(String(32), nullable=False, index=True)
    roller_temp_c = Column(Float, nullable=False)
    vibration_hz = Column(Float, nullable=False)
    motor_current_a = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False, index=True)
    primary_driver = Column(String(32), nullable=False)
    inference_ms = Column(Float, nullable=False)
    model_version = Column(String(32), nullable=False)


def create_warehouse_engine(db_url: str = DEFAULT_DB_URL) -> Engine:
    """Create a SQLAlchemy engine with SQLite WAL mode for concurrent reader/writer safety."""
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, echo=False, connect_args=connect_args)

    if db_url.startswith("sqlite") and ":memory:" not in db_url:
        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

    Base.metadata.create_all(engine)
    return engine


def create_session_factory(db_url: str = DEFAULT_DB_URL):
    engine = create_warehouse_engine(db_url)
    return engine, sessionmaker(bind=engine)