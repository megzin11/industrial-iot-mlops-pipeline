import tempfile
import time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

from src.detector import (
    ExplainableAnomalyDetector,
    generate_healthy_baseline,
    sample_telemetry_frame,
)
from src.models import TelemetryRecord, create_session_factory


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, regimes: np.ndarray) -> dict[str, float]:
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    regime_recalls = {}
    for reg in ("thermal_runaway", "bearing_vibration", "cooling_decoupling"):
        mask = regimes == reg
        regime_recalls[reg] = float((y_pred[mask] == 1).mean()) if mask.any() else 0.0

    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": fpr,
        **regime_recalls,
    }


def export_benchmark_figure(
    X: np.ndarray,
    regimes: np.ndarray,
    metrics_map: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=140)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Sensor Correlation Manifold (Roller Temp vs Motor Current)
    palette = {
        "normal": ("#94a3b8", 0.25, 10, "Normal (Correlated)"),
        "thermal_runaway": ("#dc2626", 0.75, 18, "Thermal Runaway"),
        "bearing_vibration": ("#2563eb", 0.65, 16, "Bearing Vibration"),
        "cooling_decoupling": ("#d97706", 0.85, 22, "Cooling Decoupling (Off-Manifold)"),
    }
    for reg, (color, alpha, size, label) in palette.items():
        idx = np.where(regimes == reg)[0][:600]
        ax1.scatter(X[idx, 0], X[idx, 2], c=color, alpha=alpha, s=size, label=label, edgecolors="none")

    ax1.set_title("Sensor Manifold: Roller Temp vs. Motor Current", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Roller Temperature (C)", fontsize=9.5)
    ax1.set_ylabel("Motor Current (A)", fontsize=9.5)
    ax1.grid(linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=True, facecolor="#f8fafc", fontsize=8)

    # Panel 2: 4-Way Model Selection Ablation (Precision, Recall, F1, Decoupling Recall)
    metric_keys = ["precision", "recall", "f1", "cooling_decoupling"]
    metric_labels = ["Precision", "Overall\nRecall", "F1 Score", "Decoupling\nRecall"]
    det_names = list(metrics_map.keys())
    colors = ["#64748b", "#ef4444", "#3b82f6", "#16a34a"]

    x_pos = np.arange(len(metric_keys))
    width = 0.19
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for det_name, offset, color in zip(det_names, offsets, colors):
        vals = [metrics_map[det_name][k] * 100.0 for k in metric_keys]
        bars = ax2.bar(x_pos + offset * width, vals, width=width, label=det_name, color=color, alpha=0.92)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2.0, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.0, color="#1e293b")

    ax2.set_ylim(0, 115)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylabel("Score (%)", fontsize=9.5)
    ax2.set_title("4-Way Model Selection Ablation (10,000 Frames)", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=True, facecolor="#f8fafc", fontsize=7.8, loc="lower left")

    plt.tight_layout(pad=2.0)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def main() -> None:
    n_frames = 10_000
    X_train = generate_healthy_baseline(n_samples=2000, seed=42)

    # Train promoted Mahalanobis detector + baseline Isolation Forest on identical healthy data
    promoted = ExplainableAnomalyDetector().fit(X_train)
    iforest = IsolationForest(n_estimators=100, contamination=0.005, random_state=42).fit(X_train)

    # Build held-out 10,000-frame evaluation stream
    rng = np.random.default_rng(2026)
    frames = []
    regimes = []
    for _ in range(n_frames):
        payload, regime = sample_telemetry_frame(rng, fault_prob=0.10)
        frames.append([payload["roller_temp_c"], payload["vibration_hz"], payload["motor_current_a"]])
        regimes.append(regime)

    X = np.array(frames, dtype=np.float64)
    regimes_arr = np.array(regimes)
    y_true = (regimes_arr != "normal").astype(int)

    # 1. Univariate 3.5-sigma baseline
    z_mat = (X - promoted.feature_means_) / promoted.feature_stds_
    pred_z = (np.max(np.abs(z_mat), axis=1) >= 3.5).astype(int)

    # 2. Standalone Isolation Forest (evaluated baseline)
    pred_if = (iforest.decision_function(X) < 0.0).astype(int)

    # 3. Promoted Mahalanobis Covariance Detector
    t_batch = time.perf_counter()
    pred_mahal_bool, mahal_scores, drivers = promoted.score_batch(X)
    batch_ms = (time.perf_counter() - t_batch) * 1000.0
    pred_mahal = pred_mahal_bool.astype(int)

    # 4. Hybrid IForest + Mahalanobis (intermediate ablation)
    pred_hybrid = np.maximum(pred_if, pred_mahal)

    metrics_map = {
        "Univariate Z (3.5s)": evaluate_predictions(y_true, pred_z, regimes_arr),
        "Standalone IForest": evaluate_predictions(y_true, pred_if, regimes_arr),
        "Hybrid IForest+Mahal": evaluate_predictions(y_true, pred_hybrid, regimes_arr),
        "Mahalanobis (Promoted)": evaluate_predictions(y_true, pred_mahal, regimes_arr),
    }

    print(f"\n=== 10,000-Frame Industrial MLOps Ablation Benchmark ({int(y_true.sum()):,} injected faults) ===")
    header = (
        f"  {'Detector':<23} | {'Prec.':>6} | {'Recall':>6} | {'F1':>6} | {'FPR':>6} | "
        f"{'Thermal':>7} | {'Bearing':>7} | {'Decoupl.':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, m in metrics_map.items():
        print(
            f"  {name:<23} | {m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['f1']:>6.4f} | "
            f"{m['fpr']*100:>5.2f}% | {m['thermal_runaway']*100:>6.1f}% | "
            f"{m['bearing_vibration']*100:>6.1f}% | {m['cooling_decoupling']*100:>7.1f}%"
        )

    # Single-frame latency comparison (200 readings): IForest vs Promoted Mahalanobis
    if_latencies = []
    for i in range(200):
        t0 = time.perf_counter()
        _ = iforest.decision_function(X[i : i + 1])
        if_latencies.append((time.perf_counter() - t0) * 1000.0)

    mahal_latencies = [promoted.score_reading(X[i, 0], X[i, 1], X[i, 2]).inference_ms for i in range(200)]
    if_p50, if_p95 = float(np.percentile(if_latencies, 50)), float(np.percentile(if_latencies, 95))
    m_p50, m_p95 = float(np.percentile(mahal_latencies, 50)), float(np.percentile(mahal_latencies, 95))

    # SQLite WAL bulk persistence throughput (10,000 ORM records)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{Path(tmpdir) / 'bench.db'}"
        engine, SessionLocal = create_session_factory(db_url)
        t_db = time.perf_counter()
        session = SessionLocal()
        records = [
            TelemetryRecord(
                machine_id="EXTRUDER-X1",
                timestamp="2026-11-19 12:00:00",
                roller_temp_c=float(X[i, 0]),
                vibration_hz=float(X[i, 1]),
                motor_current_a=float(X[i, 2]),
                anomaly_score=float(mahal_scores[i]),
                is_anomaly=bool(pred_mahal_bool[i]),
                primary_driver=drivers[i],
                inference_ms=m_p50,
                model_version=promoted.model_version,
            )
            for i in range(n_frames)
        ]
        session.bulk_save_objects(records)
        session.commit()
        session.close()
        engine.dispose()
        db_ms = (time.perf_counter() - t_db) * 1000.0

    chart_path = Path("docs/mlops_benchmark.png")
    export_benchmark_figure(X, regimes_arr, metrics_map, chart_path)
    size_kb = chart_path.stat().st_size / 1024.0

    print("\n=== Throughput & Latency Profile ===")
    print(f"  IForest Single-Frame Latency     : P50 = {if_p50:.3f} ms | P95 = {if_p95:.3f} ms")
    print(f"  Mahalanobis Single-Frame Latency : P50 = {m_p50:.4f} ms | P95 = {m_p95:.4f} ms ({if_p50 / max(m_p50, 1e-6):.0f}x faster)")
    print(f"  Mahalanobis Batch (10K rows)     : {batch_ms:.2f} ms ({n_frames / (batch_ms / 1000.0):,.0f} frames/sec)")
    print(f"  SQLite WAL Bulk Persist (10K)    : {db_ms:.1f} ms ({n_frames / (db_ms / 1000.0):,.0f} rows/sec)")
    print(f"  Benchmark Chart Exported         : {chart_path} ({size_kb:.1f} KB)\n")


if __name__ == "__main__":
    main()