"""Classify Sentinel-2 tiles (produced by distributed_tiling.py) with the
Phase 1 EuroSAT ResNet18, using a Ray actor that loads the model once and
processes tiles in batches -- rather than a task that reloads the model on
every call.

With a single GPU on this cluster, "distributed" here means keeping the
loaded model resident and streaming batches of work through it, not fanning
out across multiple GPU workers.

Calibration: EuroSAT's `ToTensor()` maps its rendered 8-bit JPEGs (0-255) to
[0,1]. Our tiles are raw 16-bit Sentinel-2 L1C reflectance, with no direct
JPEG equivalent. Rather than the per-tile percentile stretch tried in
notebook 03 (which broke pixel-value consistency across tiles -- the same
real reflectance mapped to different apparent brightness depending on what
else was in that tile), tiles are divided by one FIXED reflectance constant
and clipped to [0,1] -- an approximation of EuroSAT's unknown original scale,
but applied identically to every tile.
"""

import time
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import ray

MLFLOW_TRACKING_DIR = Path("mlruns")
MLFLOW_EXPERIMENT = "terra-land-use-inference"

TILES_DIR = Path("data/sentinel2_tiles")
# Grid outputs live inside TILES_DIR (not a sibling of it) deliberately: on
# KubeRay, only TILES_DIR is backed by the shared PVC -- saving here rather
# than in TILES_DIR.parent keeps the result visible from every pod instead of
# stuck on whichever one happened to run the driver.
MAPS_DIR = TILES_DIR / "land_use_maps"
CHECKPOINT_PATH = Path("models/resnet18_eurosat_best.pt")
STATS_PATH = Path("data/eurosat_stats.json")
REFLECTANCE_MAX = 3000
BATCH_SIZE = 64
SMOOTHING_WINDOW = 3


def calibrate(tile):
    return np.clip(tile.astype(np.float32) / REFLECTANCE_MAX, 0, 1)


def majority_filter(grid, size=SMOOTHING_WINDOW, center_weight=3):
    """Replace each cell with the most common class in its `size`x`size`
    neighborhood -- smooths isolated misclassifications into the surrounding
    region rather than treating every tile as an independent decision.

    The center tile's own vote is counted `center_weight` times (instead of
    once, as a plain vote among its neighbors) so it only gets overridden
    when the surrounding tiles clearly disagree, not on a narrow majority --
    less aggressive smoothing without shrinking the window."""
    pad = size // 2
    n_rows, n_cols = grid.shape
    smoothed = np.empty_like(grid)
    for i in range(n_rows):
        for j in range(n_cols):
            i0, i1 = max(0, i - pad), min(n_rows, i + pad + 1)
            j0, j1 = max(0, j - pad), min(n_cols, j + pad + 1)
            window = list(grid[i0:i1, j0:j1].flatten())
            window += [grid[i, j]] * (center_weight - 1)
            smoothed[i, j] = Counter(window).most_common(1)[0][0]
    return smoothed


@ray.remote(num_gpus=1)
class TileClassifier:
    def __init__(self):
        # Imports deferred to inside the actor: importing torch/torchvision at
        # module level made Ray fail to serialize the actor class definition
        # (a non-picklable internal torch object gets captured in the
        # closure) -- keeping them out of the module's global scope avoids it.
        import json

        import torch
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classes come from eurosat_stats.json, not EuroSAT(...).classes --
        # the latter re-derives the list by scanning the downloaded dataset's
        # folders, which requires the full ~224MB dataset to be present. It
        # isn't (data/eurosat/ is excluded from the image, same reasoning as
        # the Sentinel-2 scene/tiles), so the class list -- a fixed property
        # of the dataset -- is read from the stats file instead, saved
        # alongside mean/std in notebook 01.
        with open(STATS_PATH) as f:
            stats = json.load(f)
        self.classes = stats["classes"]
        self.mean = torch.tensor(stats["mean"]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(stats["std"]).view(3, 1, 1).to(self.device)

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
        self.model = model.to(self.device).eval()

    def classify_batch(self, paths):
        import torch

        tiles = [calibrate(np.load(p)) for p in paths]
        batch = np.stack(tiles)  # (B, H, W, 3)

        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(self.device)
        tensor = (tensor - self.mean) / self.std

        with torch.no_grad():
            preds = self.model(tensor).argmax(dim=1).cpu().numpy()

        names = [Path(p).stem for p in paths]
        return list(zip(names, (self.classes[p] for p in preds)))


if __name__ == "__main__":
    ray.init(address="auto")

    # Plain filesystem tracking ("file:...") is in maintenance mode as of
    # MLflow 3.x -- SQLite is the lightweight alternative: still a single
    # local file, no separate server, but on the actively maintained path.
    MLFLOW_TRACKING_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR.resolve()}/mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_params({
            "reflectance_max": REFLECTANCE_MAX,
            "smoothing_window": SMOOTHING_WINDOW,
            "batch_size": BATCH_SIZE,
        })

        tile_paths = sorted(str(p) for p in TILES_DIR.glob("*.npy"))
        print(f"Classifying {len(tile_paths)} tiles...")
        mlflow.log_param("n_tiles", len(tile_paths))

        classifier = TileClassifier.remote()

        start = time.time()
        futures = [
            classifier.classify_batch.remote(tile_paths[i : i + BATCH_SIZE])
            for i in range(0, len(tile_paths), BATCH_SIZE)
        ]
        results = ray.get(futures)
        elapsed = time.time() - start
        predictions = [item for batch in results for item in batch]

        counts = {}
        for _, cls in predictions:
            counts[cls] = counts.get(cls, 0) + 1

        print(f"Classified {len(predictions)} tiles in {elapsed:.1f}s. Predicted class counts:")
        for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {cls:25s} {count}")
            mlflow.log_metric(f"raw_count_{cls}", count)
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.log_metric("tiles_per_second", len(predictions) / elapsed)

        # Reassemble a 2D (row x col) map from "tile_ROW_COL" filenames, and
        # smooth it with a majority filter to unify tiles into coherent regions.
        parsed = []
        for name, cls in predictions:
            _, row_str, col_str = name.split("_")
            parsed.append((int(row_str), int(col_str), cls))

        n_rows = max(row for row, _, _ in parsed) + 1
        n_cols = max(col for _, col, _ in parsed) + 1
        grid = np.full((n_rows, n_cols), "", dtype=object)
        for row, col, cls in parsed:
            grid[row, col] = cls

        smoothed = majority_filter(grid)

        changed = (grid != smoothed).sum()
        mlflow.log_metric("pct_changed_by_smoothing", 100 * changed / grid.size)
        smoothed_counts = Counter(smoothed.flatten())
        for cls, count in smoothed_counts.items():
            mlflow.log_metric(f"smoothed_count_{cls}", count)

        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        grid_path = MAPS_DIR / "land_use_grid.npy"
        smoothed_path = MAPS_DIR / "land_use_grid_smoothed.npy"
        np.save(grid_path, grid)
        np.save(smoothed_path, smoothed)
        mlflow.log_artifact(grid_path)
        mlflow.log_artifact(smoothed_path)
        print(f"Saved {n_rows}x{n_cols} prediction grid to {MAPS_DIR}/land_use_grid.npy "
              f"(+ land_use_grid_smoothed.npy, {SMOOTHING_WINDOW}x{SMOOTHING_WINDOW} majority filter)")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
