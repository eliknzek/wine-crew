"""Download the raw Wine Reviews CSV from Kaggle into data/raw/ and print its sha256.

Usage:
    python scripts/download_data.py           # skip the download if the file exists
    python scripts/download_data.py --force   # download again

Kaggle credentials are not read from .env. Use any method the kaggle package
supports: `kaggle auth login`, KAGGLE_API_TOKEN, ~/.kaggle/access_token, or
~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(dataset: str, file_name: str, dest_dir: Path) -> None:
    # The kaggle package authenticates on import, so import it only when needed.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(dataset, file_name, path=str(dest_dir), force=True, quiet=False, unzip=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download even if the file already exists")
    args = parser.parse_args(argv)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    raw_path = REPO_ROOT / config["paths"]["raw_csv"]
    source = config["data_source"]

    if raw_path.exists() and not args.force:
        print(f"Already present, skipping download: {raw_path.relative_to(REPO_ROOT)}")
    else:
        print(f"Downloading {source['file']} from kaggle dataset {source['kaggle_dataset']} ...")
        download(source["kaggle_dataset"], source["file"], raw_path.parent)
        if not raw_path.exists():
            print(f"Download finished but {raw_path} was not found.", file=sys.stderr)
            return 1

    print(f"sha256 {sha256_of(raw_path)}  {raw_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
