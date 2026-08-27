"""POC: detect forest loss by comparing two classifications of the same area,
three years apart, with the existing Phase 1 EuroSAT ResNet18.

Standalone script, deliberately NOT importing from distributed_tiling.py /
distributed_inference.py -- this is an experiment, not a change to the
production pipeline, so it duplicates the small pieces it needs (STAC
lookup, row-strip reads, calibration, majority filter) rather than reusing
or modifying those scripts.

Area: the Harz mountains, Germany -- hit by a well-documented, large-scale
bark-beetle/drought dieback in 2018-2020. Comparing an August 2017 scene
(before) against a September 2020 scene (after), same MGRS tile (32UPC),
both near-zero cloud cover, same season (avoids seasonal false positives).

Not Ray-parallelized: cropped to just the area of interest (via the STAC
item's own georeferencing) rather than the full ~110x110km scene, small
enough (~2600 tiles per date) that a plain sequential run finishes in
well under a minute -- no meaningful distributed-systems problem here to
justify the complexity, same reasoning as calibrate_reflectance.py.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import torch

# GDAL's vsicurl has no timeout by default -- a stalled/rate-limited S3
# connection can hang a worker thread forever instead of erroring out.
# Found the hard way: the concurrent tiling below silently stalled at 85%
# with zero further CPU activity, no exception, no progress. These make a
# stuck request fail (and get retried) instead of hanging indefinitely.
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "20")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l1c"
BBOX = [10.4, 51.65, 10.9, 51.95]  # Harz mountains, Germany
TILE_SIZE = 64

# Exact scenes, both near-zero cloud cover, same MGRS tile, same season.
SCENES = {
    "before": "S2B_32UPC_20170823_0_L1C",
    "after": "S2A_32UPC_20200921_0_L1C",
}

OUTPUTS_DIR = Path("data/outputs/deforestation_poc")
CHECKPOINT_PATH = Path("models/resnet18_eurosat_best.pt")
STATS_PATH = Path("data/eurosat_stats.json")
REFLECTANCE_MAX = 5600  # same calibration as distributed_inference.py
BATCH_SIZE = 64
SMOOTHING_WINDOW = 3


def s3_to_https(href):
    if href.startswith("s3://"):
        bucket, key = href.removeprefix("s3://").split("/", 1)
        # Earth Search's catalog has a metadata bug on some older L1C items:
        # the href's bucket says "sentinel-s2-l2a" even though the key uses
        # the legacy L1C tile path (tiles/{utm}/.../B0X.jp2), which only
        # exists in the public "sentinel-s2-l1c" bucket -- correct it when
        # that path shape is detected, rather than trusting the bucket field.
        if key.startswith("tiles/") and bucket != "sentinel-s2-l1c":
            bucket = "sentinel-s2-l1c"
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def find_scene_urls(item_id):
    client = Client.open(STAC_URL)
    search = client.search(collections=[COLLECTION], ids=[item_id])
    item = next(search.items())
    return {c: s3_to_https(item.assets[c].href) for c in ("red", "green", "blue")}


def compute_pixel_window(url, bbox):
    """Convert the (lon/lat) area of interest into a pixel window in this
    scene's own CRS/grid -- clamped to the scene's bounds, since the bbox
    can extend slightly past one edge of the tile."""
    with rasterio.open(f"/vsicurl/{url}") as src:
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bbox)
        row_start, col_start = src.index(minx, maxy)
        row_end, col_end = src.index(maxx, miny)
        row_start, row_end = sorted((max(0, row_start), min(src.height, row_end)))
        col_start, col_end = sorted((max(0, col_start), min(src.width, col_end)))
        # Snap to a whole number of tiles.
        n_rows = (row_end - row_start) // TILE_SIZE
        n_cols = (col_end - col_start) // TILE_SIZE
        return row_start, col_start, n_rows, n_cols


def read_band_window(url, window):
    with rasterio.open(f"/vsicurl/{url}") as src:
        return src.read(1, window=window)


def tile_scene(label, item_id):
    tiles_dir = OUTPUTS_DIR / label / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    urls = find_scene_urls(item_id)
    row_offset, col_offset, n_rows, n_cols = compute_pixel_window(urls["red"], BBOX)
    width = n_cols * TILE_SIZE
    height = n_rows * TILE_SIZE
    expected = n_rows * n_cols

    existing = len(list(tiles_dir.glob("*.npy")))
    if existing >= expected:
        print(f"[{label}] {item_id}: already tiled ({existing} tiles), skipping")
        return n_rows, n_cols

    print(f"[{label}] {item_id}: {n_rows}x{n_cols} tiles ({expected} total)")

    # One windowed read per band for the whole crop, rather than one read
    # per tile-row -- ~53 small per-row requests turned out to be fragile
    # against this particular S3 path (some individual row reads would hang
    # well past GDAL_HTTP_TIMEOUT and stall the whole run, even with a
    # thread pool). Fewer, larger requests: 3 total instead of ~160.
    window = Window(col_offset, row_offset, width, height)
    start = time.time()
    scene_rgb = np.stack(
        [read_band_window(urls[c], window) for c in ("red", "green", "blue")],
        axis=-1,
    )
    for row in range(n_rows):
        for col in range(n_cols):
            tile = scene_rgb[
                row * TILE_SIZE : (row + 1) * TILE_SIZE,
                col * TILE_SIZE : (col + 1) * TILE_SIZE,
                :,
            ]
            np.save(tiles_dir / f"tile_{row:03d}_{col:03d}.npy", tile)
    elapsed = time.time() - start
    print(f"[{label}] tiled in {elapsed:.1f}s")
    return n_rows, n_cols


def calibrate(tile):
    return np.clip(tile.astype(np.float32) / REFLECTANCE_MAX, 0, 1)


def majority_filter(grid, size=SMOOTHING_WINDOW, center_weight=3):
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


def classify_scene(label, n_rows, n_cols, model, classes, mean, std, device):
    tiles_dir = OUTPUTS_DIR / label / "tiles"
    grid = np.full((n_rows, n_cols), "", dtype=object)

    paths, coords = [], []
    for row in range(n_rows):
        for col in range(n_cols):
            paths.append(tiles_dir / f"tile_{row:03d}_{col:03d}.npy")
            coords.append((row, col))

    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[i : i + BATCH_SIZE]
            batch = np.stack([calibrate(np.load(p)) for p in batch_paths])
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device)
            tensor = (tensor - mean) / std
            preds = model(tensor).argmax(dim=1).cpu().numpy()
            for (row, col), pred in zip(coords[i : i + BATCH_SIZE], preds):
                grid[row, col] = classes[pred]

    smoothed = majority_filter(grid)
    print(f"[{label}] classified {len(paths)} tiles")
    return smoothed


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(STATS_PATH) as f:
        stats = json.load(f)
    classes = stats["classes"]
    mean = torch.tensor(stats["mean"]).view(3, 1, 1).to(device)
    std = torch.tensor(stats["std"]).view(3, 1, 1).to(device)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device).eval()

    for label, item_id in SCENES.items():
        n_rows, n_cols = tile_scene(label, item_id)

        grid_path = OUTPUTS_DIR / label / "land_use_grid_smoothed.npy"
        if grid_path.exists():
            smoothed = np.load(grid_path, allow_pickle=True)
            print(f"[{label}] already classified ({smoothed.shape}), skipping")
        else:
            smoothed = classify_scene(label, n_rows, n_cols, model, classes, mean, std, device)
            np.save(grid_path, smoothed)

        counts = Counter(smoothed.flatten())
        print(f"[{label}] class counts: {dict(counts)}")

    print(f"\nDone. Outputs under {OUTPUTS_DIR}/{{before,after}}/")
