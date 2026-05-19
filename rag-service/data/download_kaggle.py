# One-time script to download the Kaggle fashion dataset into data/raw/.
# Requires ~/.kaggle/kaggle.json credentials. Run manually before the ingestion pipeline.

from __future__ import annotations

# One-time script: downloads the Kaggle fashion dataset and writes raw CSVs to data/raw/.
# Run manually: python data/download_kaggle.py
# Requires ~/.kaggle/kaggle.json credentials.


def download() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    download()
