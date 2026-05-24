# One-time script to download the Kaggle fashion dataset into data/raw/.
# Uses kagglehub for reproducibility — anyone with ~/.kaggle/kaggle.json can run this.
# Run manually: python data/download_kaggle.py

from __future__ import annotations

import os
import shutil
from pathlib import Path

DATASET = "hiteshsuthar101/myntra-fashion-product-dataset"
RAW_DIR = Path(__file__).parent / "raw"


def download() -> Path:
    import kagglehub

    print(f"Downloading {DATASET}...")
    src = Path(kagglehub.dataset_download(DATASET))

    RAW_DIR.mkdir(exist_ok=True)

    # Copy only the CSV — we don't need the Images folder for text RAG
    for f in src.rglob("*.csv"):
        dest = RAW_DIR / f.name
        shutil.copy(f, dest)
        print(f"Copied {f.name} → {dest}")

    return RAW_DIR


if __name__ == "__main__":
    download()
