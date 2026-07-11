"""Download the EuroSAT (RGB) dataset into this directory."""

from pathlib import Path

from torchvision.datasets import EuroSAT

ROOT = Path(__file__).parent

if __name__ == "__main__":
    dataset = EuroSAT(root=ROOT, download=True)
    print(f"{len(dataset)} images, {len(dataset.classes)} classes: {dataset.classes}")
