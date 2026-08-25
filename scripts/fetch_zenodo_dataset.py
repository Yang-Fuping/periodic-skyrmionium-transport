"""Download, verify, and extract the complete paper dataset."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "F7AA9D88E3B0198D01994DC9DBB505AF997187F8BABA9C6C4999D06412C93666"
ZENODO_DOI = "10.5281/zenodo.22092300"
DEFAULT_URL = (
    "https://zenodo.org/api/records/22092300/files/"
    "periodic-skyrmionium-transport-data-v1.1.0.zip/content"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        help=("Dataset URL; defaults to Zenodo version 1.1.0 and may also "
              "be set as SKYRMIONIUM_DATASET_URL"),
    )
    parser.add_argument("--archive", type=Path,
                        help="Use an already downloaded v1.1.0 ZIP")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()

    if args.url and args.archive:
        parser.error("--url and --archive are mutually exclusive")
    url = args.url or os.environ.get("SKYRMIONIUM_DATASET_URL") or DEFAULT_URL

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="skyrmionium_dataset_") as temporary:
        downloaded = Path(temporary) / "dataset.zip"
        if args.archive:
            shutil.copyfile(args.archive.resolve(), downloaded)
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "paper-reproduction/1.1"})
            with urllib.request.urlopen(request) as response, downloaded.open("wb") as handle:
                shutil.copyfileobj(response, handle)

        actual = sha256(downloaded)
        expected = args.sha256.upper()
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch: expected {expected}, received {actual}")

        with zipfile.ZipFile(downloaded) as archive:
            names = set(archive.namelist())
            required = {
                "results/disorder_topology_comparison_v1/assessment.json",
                "results/peer_review_convergence/assessment.json",
                "results/probe_width_crossover_v2/assessment.json",
                "results/kwant_validation/full_with_array_hall.json",
            }
            missing = sorted(required - names)
            if missing:
                raise SystemExit("Incomplete dataset archive; missing: " + ", ".join(missing))
            archive.extractall(args.output)

    print(f"Verified and extracted complete dataset to {args.output / 'results'}")


if __name__ == "__main__":
    main()
