from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.env  # noqa: F401 - loads .env
from core.config import get_config
from core.utils import configure_logging
from rag.rag_pipeline import RAG


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the RAG index.")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not prepend metadata from metadata.csv to chunks",
    )
    args = parser.parse_args()

    cfg = get_config()
    configure_logging(
        cfg.pipeline_log_path,
        cfg.logging,
        add_stdout=True,
        stdout_format="{level} | {message}",
    )

    rag = RAG()
    rag.run_rag_pipeline(use_metadata=not args.no_metadata)
    print("RAG index built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
