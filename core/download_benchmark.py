from __future__ import annotations

import sys
import zipfile
import argparse
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmark_tubingen"

FILES = {
    "pairs.zip": "https://webdav.tuebingen.mpg.de/cause-effect/pairs.zip",
    "pairmeta.txt": "https://webdav.tuebingen.mpg.de/cause-effect/pairmeta.txt",
    "README": "https://webdav.tuebingen.mpg.de/cause-effect/README",
}


def download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"Already exists: {target}")
        return

    print(f"Downloading: {url}")
    target.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target)
    print(f"Saved: {target}")


def unzip_pairs(zip_path: Path, output_dir: Path, *, force: bool = False) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    marker_files = [
        output_dir / "pair0001.txt",
        output_dir / "pair0001_des.txt",
    ]
    if not force and all(path.exists() for path in marker_files):
        print(f"Pairs already extracted in: {output_dir}")
        return

    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    print(f"Extracted to: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Tübingen cause-effect pairs.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Extract pairs.zip even if pair files already exist.",
    )
    args = parser.parse_args()

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        download_file(url, BENCHMARK_DIR / filename)

    unzip_pairs(BENCHMARK_DIR / "pairs.zip", BENCHMARK_DIR, force=args.force)

    print(f"\nDone. Benchmark directory: {BENCHMARK_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
