"""Download a single Sentinel-2 L1C scene (RGB bands only) from the public
Earth Search STAC API on AWS -- no AWS account required.

L1C (not L2A) is used deliberately: EuroSAT, the dataset the classifier was
trained on, is itself derived from Sentinel-2 L1C imagery, so tiles cut from
an L1C scene stay consistent with what the model has seen.
"""

from pathlib import Path

import requests
from pystac_client import Client

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l1c"
BBOX = [2.0, 48.6, 2.7, 49.0]  # Paris area
ROOT = Path(__file__).parent / "sentinel2_scene"
BANDS = {"red": "B04.jp2", "green": "B03.jp2", "blue": "B02.jp2"}

if __name__ == "__main__":
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=BBOX,
        query={"eo:cloud_cover": {"lt": 10}},
        limit=20,
    )
    item = min(search.items(), key=lambda i: i.properties["eo:cloud_cover"])
    print(f"Selected scene: {item.id} (cloud cover: {item.properties['eo:cloud_cover']:.1f}%)")

    ROOT.mkdir(exist_ok=True)
    for band_name, filename in BANDS.items():
        href = item.assets[band_name].href
        if href.startswith("s3://"):
            bucket, key = href.removeprefix("s3://").split("/", 1)
            href = f"https://{bucket}.s3.amazonaws.com/{key}"
        response = requests.get(href)
        response.raise_for_status()
        out_path = ROOT / filename
        out_path.write_bytes(response.content)
        print(f"{filename}: {len(response.content) / 1e6:.1f} MB")
