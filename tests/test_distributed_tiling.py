"""Unit test for the one pure function in distributed_tiling.py -- everything
else in that script either calls out to the network (STAC/S3) or needs a Ray
cluster, neither of which belongs in CI.
"""

from distributed_tiling import s3_to_https


def test_s3_to_https_converts_s3_uri():
    assert (
        s3_to_https("s3://sentinel-s2-l1c/tiles/32/U/PC/2020/9/21/0/B04.jp2")
        == "https://sentinel-s2-l1c.s3.amazonaws.com/tiles/32/U/PC/2020/9/21/0/B04.jp2"
    )


def test_s3_to_https_passes_through_non_s3_urls():
    url = "https://already-https.example.com/file.jp2"
    assert s3_to_https(url) == url
