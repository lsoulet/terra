"""Split a Sentinel-2 L1C scene into 64x64 tiles in parallel with Ray.

Each tile is read directly from its source on AWS (HTTP range requests via
GDAL's /vsicurl/ virtual filesystem) rather than from a local download --
this avoids needing shared storage between pods when running distributed
across multiple nodes (e.g. on the KubeRay cluster), since every task fetches
what it needs straight from the source.

Tiles are saved as raw .npy arrays (uint16, RGB) -- no normalization or
rescaling here, that's inference's job (Phase 4). Row/col are encoded in the
filename, which is enough to place each tile back in the scene later.

One task per tile (3 HTTP requests each) was tried first and was far too slow
(network latency per request dominates) -- instead, each Ray task reads one
full tile-row per band (3 requests for a whole row of tiles) and slices it
into individual tiles in memory, cutting the request count by ~n_cols. Tasks
request a fraction of a CPU each since they're network-bound, not purely
compute-bound, letting Ray overlap more of them than there are cores -- but
not too many: each task's JP2000 decompression genuinely uses ~100-200MB, and
since these tasks don't request a GPU, Ray is free to schedule them on the
lightweight head node (4Gi) as well as the worker (8Gi), so concurrency is
kept modest enough to avoid OOM on the smaller of the two.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
import ray
from pystac_client import Client
from rasterio.windows import Window

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l1c"
BBOX = [2.0, 48.6, 2.7, 49.0]  # Paris area
TILE_SIZE = 64
OUTPUT_DIR = Path("data/sentinel2_tiles")


def s3_to_https(href):
    if href.startswith("s3://"):
        bucket, key = href.removeprefix("s3://").split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def find_scene_urls():
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=BBOX,
        query={"eo:cloud_cover": {"lt": 10}},
        limit=20,
    )
    item = min(search.items(), key=lambda i: i.properties["eo:cloud_cover"])
    print(f"Selected scene: {item.id} (cloud cover: {item.properties['eo:cloud_cover']:.1f}%)")
    return {c: s3_to_https(item.assets[c].href) for c in ("red", "green", "blue")}


def read_band_row(url, row, width):
    window = Window(0, row * TILE_SIZE, width, TILE_SIZE)
    with rasterio.open(f"/vsicurl/{url}") as src:
        return src.read(1, window=window)


@ray.remote(num_cpus=0.25)
def extract_tile_row(urls, row, n_cols):
    # Each task may land on a different pod/node than the driver, with its own
    # local disk -- create the output dir here too rather than assuming the
    # driver's mkdir() is visible.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    width = n_cols * TILE_SIZE
    row_rgb = np.stack(
        [read_band_row(urls[c], row, width) for c in ("red", "green", "blue")],
        axis=-1,
    )

    names = []
    total_bytes = 0
    for col in range(n_cols):
        tile = row_rgb[:, col * TILE_SIZE : (col + 1) * TILE_SIZE, :]
        out_path = OUTPUT_DIR / f"tile_{row:03d}_{col:03d}.npy"
        np.save(out_path, tile)
        names.append(out_path.name)
        total_bytes += out_path.stat().st_size
    return names, total_bytes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Limit the number of tile-rows processed (for quick/demo runs); default processes the full scene.",
    )
    args = parser.parse_args()

    ray.init(address="auto")

    urls = find_scene_urls()

    with rasterio.open(f"/vsicurl/{urls['red']}") as src:
        n_rows = src.height // TILE_SIZE
        n_cols = src.width // TILE_SIZE

    if args.max_rows is not None:
        n_rows = min(n_rows, args.max_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {n_rows * n_cols} tiles ({n_rows}x{n_cols}) of {TILE_SIZE}x{TILE_SIZE}px, "
          f"{n_rows} row-tasks...")

    start = time.time()
    futures = [extract_tile_row.remote(urls, row, n_cols) for row in range(n_rows)]
    results = ray.get(futures)
    elapsed = time.time() - start

    all_names = [name for row_names, _ in results for name in row_names]
    total_size = sum(row_bytes for _, row_bytes in results)
    print(f"Done: {len(all_names)} tiles in {elapsed:.1f}s ({total_size / 1e6:.1f} MB total)")
