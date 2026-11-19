from dataclasses import dataclass
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.covariance import EmpiricalCovariance

FEATURE_NAMES = ("roller_temp_c", "vibration_hz", "motor_current_a")
DEFAULT_MODEL_PATH = Path("artifacts/mahalanobis_detector_v2.joblib")
MODEL_VERSION = "mahalanobis-cov-v2.0"


@dataclass(frozen=True)
class InferenceResult:
    is_anomaly: bool
    anomaly_score: float
    primary_driver: str
    feature_zscores: dict[str, float]
    inference_ms: float
    model_version: str


def generate_healthy_baseline(n_samples: int = 2000, seed: int = 42) -> np.ndarray:
    """
    Generate correlated healthy industrial extruder telemetry.
    Under normal load, roller temperature, vibration, and motor current co-vary positively.
    """
    rng = np.random.default_rng(seed)
    load = rng.normal(0.0, 1.0, size=n_samples)

    temp = 68.0 + 2.0 * load + rng.normal(0.0, 0.5, size=n_samples)
    vib = 12.5 + 0.8 * load + rng.normal(0.0, 0.25, size=n_samples)
    current = 18.0 + 1.0 * load + rng.normal(0.0, 0.25, size=n_samples)
    return np.column_stack([temp, vib, current])


def sample_telemetry_frame(
    rng: np.random.Generator,
    machine_id: str = "EXTRUDER-X1",
    fault_prob: float = 0.10,
) -> tuple[dict, str]:
    """
    Generate one realistic sensor reading across 4 physical operating regimes:
      - normal: correlated Gaussian load variation
      - thermal_runaway: single-axis roller overheating (+12 to +22 C)
      - bearing_vibration: single-axis mechanical vibration spike (+5 to +12 Hz)
      - cooling_decoupling: subtle multivariate fault (hot roller + low load/current;
        each individual feature stays < 3.2 sigma, breaking the normal correlation manifold)
    """
    load = float(rng.normal(0.0, 1.0))
    temp = 68.0 + 2.0 * load + float(rng.normal(0.0, 0.5))
    vib = 12.5 + 0.8 * load + float(rng.normal(0.0, 0.25))
    current = 18.0 + 1.0 * load + float(rng.normal(0.0, 0.25))

    regime = "normal"
    if rng.random() < fault_prob:
        regime = str(rng.choice(["thermal_runaway", "bearing_vibration", "cooling_decoupling"]))
        if regime == "thermal_runaway":
            temp += float(rng.uniform(12.0, 22.0))
        elif regime == "bearing_vibration":
            vib += float(rng.uniform(5.0, 12.0))
        elif regime == "cooling_decoupling":
            temp = 68.0 + float(rng.uniform(5.2, 6.4))
            vib = 12.5 - float(rng.uniform(1.8, 2.4))
            current = 18.0 - float(rng.uniform(2.3, 3.0))

    payload = {
        "machine_id": machine_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "roller_temp_c": round(temp, 2),
        "vibration_hz": round(vib, 2),
        "motor_current_a": round(current, 2),
    }
    return payload, regime


class ExplainableAnomalyDetector:
    """
    Covariance-aware Mahalanobis anomaly detector (sklearn EmpiricalCovariance)
    paired with per-feature Z-score root-cause attribution.
    Threshold d_M >= 4.2 corresponds to the ~99.95% chi2(df=3) confidence boundary.
    """

    def __init__(
        self,
        mahalanobis_threshold: float = 4.2,
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.mahalanobis_threshold = mahalanobis_threshold
        self.model_version = model_version
        self.estimator = EmpiricalCovariance()
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, X_normal: np.ndarray) -> "ExplainableAnomalyDetector":
        X = np.asarray(X_normal, dtype=np.float64)
        self.estimator.fit(X)
        self.feature_means_ = self.estimator.location_.copy()
        self.precision_ = self.estimator.precision_.copy()
        stds = X.std(axis=0)
        self.feature_stds_ = np.where(stds == 0, 1.0, stds)
        return self

    def _attribute_driver(self, z_scores: np.ndarray, is_anomaly: bool) -> str:
        if not is_anomaly:
            return "normal"
        abs_z = np.abs(z_scores)
        elevated = [FEATURE_NAMES[i] for i, val in enumerate(abs_z) if val >= 3.5]
        if len(elevated) >= 2:
            return "multivariate"
        if len(elevated) == 1:
            return elevated[0]
        return "correlation_drift"

    def score_reading(self, temp_c: float, vibration_hz: float, current_a: float) -> InferenceResult:
        if self.feature_means_ is None or self.feature_stds_ is None or self.precision_ is None:
            raise RuntimeError("Detector must be fitted or loaded before scoring.")

        t0 = time.perf_counter()
        x = np.array([temp_c, vibration_hz, current_a], dtype=np.float64)
        diff = x - self.feature_means_
        mahal = float(np.sqrt(max(float(diff @ self.precision_ @ diff), 0.0)))
        z_vec = diff / self.feature_stds_

        is_anomaly = mahal >= self.mahalanobis_threshold
        driver = self._attribute_driver(z_vec, is_anomaly)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return InferenceResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(mahal, 4),
            primary_driver=driver,
            feature_zscores={name: round(float(z), 2) for name, z in zip(FEATURE_NAMES, z_vec)},
            inference_ms=round(elapsed_ms, 4),
            model_version=self.model_version,
        )

    def score_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Vectorized batch inference returning (flags, mahalanobis_scores, drivers)."""
        if self.feature_means_ is None or self.feature_stds_ is None or self.precision_ is None:
            raise RuntimeError("Detector must be fitted or loaded before scoring.")

        arr = np.asarray(X, dtype=np.float64)
        diffs = arr - self.feature_means_
        mahal_sq = np.einsum("ij,jk,ik->i", diffs, self.precision_, diffs)
        scores = np.sqrt(np.maximum(mahal_sq, 0.0))
        z_matrix = diffs / self.feature_stds_

        flags = scores >= self.mahalanobis_threshold
        drivers = [self._attribute_driver(z_matrix[i], bool(flags[i])) for i in range(len(arr))]
        return flags, scores, drivers

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_means": self.feature_means_,
                "feature_stds": self.feature_stds_,
                "precision": self.precision_,
                "mahalanobis_threshold": self.mahalanobis_threshold,
                "model_version": self.model_version,
            },
            path,
        )
        return path

    @classmethod
    def load_or_train(cls, path: Path = DEFAULT_MODEL_PATH) -> "ExplainableAnomalyDetector":
        if path.exists():
            bundle = joblib.load(path)
            if bundle.get("model_version") == MODEL_VERSION and "precision" in bundle:
                det = cls(
                    mahalanobis_threshold=bundle["mahalanobis_threshold"],
                    model_version=bundle["model_version"],
                )
                det.estimator = bundle["estimator"]
                det.feature_means_ = bundle["feature_means"]
                det.feature_stds_ = bundle["feature_stds"]
                det.precision_ = bundle["precision"]
                return det

        det = cls().fit(generate_healthy_baseline())
        det.save(path)
        return det