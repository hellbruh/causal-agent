"""Check the RAG text corpus for obvious benchmark leakage markers."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import get_config


FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"t[üu]bingen",
        r"cause\s*effect\s*pairs?",
        r"cause-effect\s+pairs?",
        r"pairmeta",
        r"pair\d{4}",
        r"ground\s+truth",
        r"x\s*-->\s*y",
        r"y\s*-->\s*x",
        r"<--",
        r"-->",
        r"mooij",
        r"peters",
        r"janzing",
        r"zscheischler",
        r"sch[öo]lkopf",
        r"\bdwd\b",
        r"\babalone\b",
        r"auto-mpg",
        r"\bbafu\b",
        r"\bundata\b",
        r"\bgagurine\b",
        r"ncep-ncar",
    )
)


def main() -> int:
    cfg = get_config()
    corpus_dir = cfg.document_corpus_dir

    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        return 2

    hits: list[tuple[Path, int, str, str]] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN_PATTERNS:
                match = pattern.search(line)
                if match:
                    hits.append((path, line_no, pattern.pattern, line.strip()))

    if not hits:
        print(f"OK: no obvious leakage markers found in {corpus_dir}")
        return 0

    print(f"Found {len(hits)} possible leakage marker(s):")
    for path, line_no, pattern, line in hits:
        print(f"{path}:{line_no}: /{pattern}/ -> {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
