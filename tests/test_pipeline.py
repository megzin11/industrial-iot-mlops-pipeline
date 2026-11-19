from pathlib import Path
from src.detector import ExplainableAnomalyDetector, generate_healthy_baseline
from src.models import TelemetryRecord, create_session_factory


def _fitted_detector() -> ExplainableAnomalyDetector:
    return ExplainableAnomalyDetector().fit(generate_healthy_baseline(n_samples=2000, seed=42))


def test_normal_operating_point_is_not_flagged():
    det = _fitted_detector()
    res = det.score_reading(68.2, 12.5, 18.1)
    assert not res.is_anomaly
    assert res.primary_driver == "normal"
    assert res.anomaly_score < 4.2


def test_thermal_runaway_attributed_to_roller_temp():
    det = _fitted_detector()
    res = det.score_reading(88.5, 12.6, 18.2)
    assert res.is_anomaly
    assert res.primary_driver == "roller_temp_c"


def test_bearing_vibration_attributed_to_vibration_hz():
    det = _fitted_detector()
    res = det.score_reading(68.0, 23.0, 18.1)
    assert res.is_anomaly
    assert res.primary_driver == "vibration_hz"


def test_cooling_decoupling_caught_as_correlation_drift_despite_low_zscores():
    det = _fitted_detector()
    res = det.score_reading(73.8, 10.6, 15.3)
    assert res.is_anomaly
    assert res.primary_driver == "correlation_drift"
    assert all(abs(z) < 3.5 for z in res.feature_zscores.values())
    assert res.anomaly_score >= 4.2


def test_model_artifact_save_and_reload_preserves_inference(tmp_path: Path):
    artifact = tmp_path / "detector.joblib"
    det = _fitted_detector()
    det.save(artifact)

    reloaded = ExplainableAnomalyDetector.load_or_train(artifact)
    r1 = det.score_reading(73.8, 10.6, 15.3)
    r2 = reloaded.score_reading(73.8, 10.6, 15.3)
    assert r1.is_anomaly == r2.is_anomaly
    assert r1.anomaly_score == r2.anomaly_score
    assert r1.primary_driver == r2.primary_driver
    assert r2.model_version == "mahalanobis-cov-v2.0"


def test_sqlite_wal_warehouse_persists_auditable_records(tmp_path: Path):
    db_file = tmp_path / "test_warehouse.db"
    engine, SessionLocal = create_session_factory(f"sqlite:///{db_file}")

    det = _fitted_detector()
    res = det.score_reading(88.5, 12.6, 18.2)

    session = SessionLocal()
    session.add(
        TelemetryRecord(
            machine_id="EXTRUDER-X1",
            timestamp="2026-11-19 12:00:00",
            roller_temp_c=88.5,
            vibration_hz=12.6,
            motor_current_a=18.2,
            anomaly_score=res.anomaly_score,
            is_anomaly=res.is_anomaly,
            primary_driver=res.primary_driver,
            inference_ms=res.inference_ms,
            model_version=res.model_version,
        )
    )
    session.commit()

    rows = session.query(TelemetryRecord).all()
    assert len(rows) == 1
    assert rows[0].is_anomaly is True
    assert rows[0].primary_driver == "roller_temp_c"

    session.close()
    engine.dispose()