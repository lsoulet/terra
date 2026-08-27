"""Pick REFLECTANCE_MAX by matching the calibrated real-scene tiles'
pixel-value distribution to EuroSAT's own (data/eurosat_stats.json), rather
than guessing a physically-plausible constant.

We have no ground truth for the real Sentinel-2 scene, so we can't measure
accuracy directly against different calibration values. What we DO have is
EuroSAT's own per-channel mean/std -- the distribution the model was
actually trained on. A calibration that makes real tiles statistically
resemble that distribution is a reasonable, label-free proxy for "this is
closer to what the model expects" -- not a guarantee of higher accuracy, but
a measurable signal where before there was only eyeballing.

Not parallelized with Ray: this reads a small random sample of already-local
tiles and does plain numpy arithmetic on them -- there's no meaningful
distributed-systems problem here to justify it, unlike the tiling/inference
scripts.
"""

import json
from pathlib import Path

import mlflow
import numpy as np

from distributed_inference import MLFLOW_TRACKING_DIR, TILES_DIR

STATS_PATH = Path("data/eurosat_stats.json")
MLFLOW_EXPERIMENT = "terra-reflectance-calibration"
CANDIDATES = [5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600, 6800]
SAMPLE_SIZE = 500
SEED = 42


def calibrate(tile, reflectance_max):
    return np.clip(tile.astype(np.float32) / reflectance_max, 0, 1)


if __name__ == "__main__":
    with open(STATS_PATH) as f:
        eurosat_stats = json.load(f)
    eurosat_mean = np.array(eurosat_stats["mean"])
    eurosat_std = np.array(eurosat_stats["std"])

    all_paths = sorted(TILES_DIR.glob("*.npy"))
    rng = np.random.default_rng(SEED)
    sample_paths = rng.choice(all_paths, size=min(SAMPLE_SIZE, len(all_paths)), replace=False)
    tiles = [np.load(p) for p in sample_paths]
    print(f"Sampled {len(tiles)} / {len(all_paths)} tiles for calibration search")

    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR.resolve()}/mlflow.db")
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT,
            artifact_location=str(MLFLOW_TRACKING_DIR.resolve() / "artifacts"),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []
    for reflectance_max in CANDIDATES:
        calibrated = np.stack([calibrate(t, reflectance_max) for t in tiles])  # (N, H, W, 3)
        mean = calibrated.mean(axis=(0, 1, 2))
        std = calibrated.std(axis=(0, 1, 2))
        distance = float(np.linalg.norm(np.concatenate([mean - eurosat_mean, std - eurosat_std])))

        with mlflow.start_run(run_name=f"reflectance_max_{reflectance_max}"):
            mlflow.log_param("reflectance_max", reflectance_max)
            mlflow.log_metrics({
                "mean_r": mean[0], "mean_g": mean[1], "mean_b": mean[2],
                "std_r": std[0], "std_g": std[1], "std_b": std[2],
                "distance_to_eurosat": distance,
            })

        results.append((reflectance_max, mean, std, distance))
        print(f"reflectance_max={reflectance_max:5d}  mean={mean.round(3)}  std={std.round(3)}  "
              f"distance={distance:.4f}")

    best = min(results, key=lambda r: r[3])
    print(f"\nEuroSAT reference: mean={eurosat_mean.round(3)}  std={eurosat_std.round(3)}")
    print(f"Best match: reflectance_max={best[0]} (distance={best[3]:.4f})")
