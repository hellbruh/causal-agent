from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_GOLD_PATH = ROOT / "data" / "benchmark_tubingen" / "pairmeta.txt"
DEFAULT_PRED_PATH = ROOT / "artifacts" / "dev_pairmeta.txt"
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "benchmark_score_report.csv"
DEFAULT_RAG_PRED_PATH = ROOT / "artifacts" / "dev_rag_pairmeta.txt"
DEFAULT_RAG_REPORT_PATH = ROOT / "artifacts" / "dev_rag_benchmark_score_report.csv"


@dataclass(frozen=True)
class PairMeta:
    pair_id: str
    cause_start: int
    cause_end: int
    effect_start: int
    effect_end: int
    weight: float

    @property
    def direction(self) -> tuple[int, int, int, int]:
        return (
            self.cause_start,
            self.cause_end,
            self.effect_start,
            self.effect_end,
        )


def read_pairmeta(path: Path) -> dict[str, PairMeta]:
    if not path.exists():
        raise FileNotFoundError(f"pairmeta file not found: {path}")

    rows: dict[str, PairMeta] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Invalid line {line_no} in {path}: {raw_line!r}")

        pair_id, cs, ce, es, ee, weight = parts[:6]
        rows[pair_id] = PairMeta(
            pair_id=pair_id,
            cause_start=int(cs),
            cause_end=int(ce),
            effect_start=int(es),
            effect_end=int(ee),
            weight=float(weight),
        )
    return rows


def weighted_accuracy(rows: list[dict[str, object]], weight_key: str) -> float | None:
    total_weight = sum(float(row[weight_key]) for row in rows)
    if total_weight == 0:
        return None
    correct_weight = sum(
        float(row[weight_key]) for row in rows if bool(row["correct"])
    )
    return correct_weight / total_weight


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "correct",
        "gold_direction",
        "pred_direction",
        "gold_weight",
        "pred_weight",
        "weight_equal",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Tübingen benchmark predictions.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--pred", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--rag", action="store_true", help="Score default RAG output files")
    args = parser.parse_args()

    pred_path = args.pred or (DEFAULT_RAG_PRED_PATH if args.rag else DEFAULT_PRED_PATH)
    report_path = args.report or (
        DEFAULT_RAG_REPORT_PATH if args.rag else DEFAULT_REPORT_PATH
    )

    gold = read_pairmeta(args.gold)
    pred = read_pairmeta(pred_path)

    rows: list[dict[str, object]] = []
    missing_predictions: list[str] = []
    extra_predictions: list[str] = []

    for pair_id in sorted(gold.keys(), key=lambda value: int(value)):
        gold_row = gold[pair_id]
        pred_row = pred.get(pair_id)
        if pred_row is None:
            missing_predictions.append(pair_id)
            continue

        correct = gold_row.direction == pred_row.direction
        rows.append(
            {
                "pair_id": pair_id,
                "correct": correct,
                "gold_direction": " ".join(map(str, gold_row.direction)),
                "pred_direction": " ".join(map(str, pred_row.direction)),
                "gold_weight": gold_row.weight,
                "pred_weight": pred_row.weight,
                "weight_equal": abs(gold_row.weight - pred_row.weight) < 1e-12,
            }
        )

    for pair_id in sorted(set(pred.keys()).difference(gold.keys()), key=lambda value: int(value)):
        extra_predictions.append(pair_id)

    evaluated = len(rows)
    correct_count = sum(1 for row in rows if bool(row["correct"]))
    accuracy = correct_count / evaluated if evaluated else None
    weighted_gold = weighted_accuracy(rows, "gold_weight") if rows else None
    weighted_pred = weighted_accuracy(rows, "pred_weight") if rows else None

    write_report(report_path, rows)

    print(f"Gold file: {args.gold}")
    print(f"Pred file: {pred_path}")
    print(f"Report: {report_path}")
    print()
    print(f"Gold pairs: {len(gold)}")
    print(f"Predicted pairs: {len(pred)}")
    print(f"Evaluated pairs: {evaluated}")
    print(f"Correct pairs: {correct_count}")
    print(f"Missing predictions: {len(missing_predictions)}")
    print(f"Extra predictions: {len(extra_predictions)}")
    print()
    print(f"Accuracy: {accuracy:.6f}" if accuracy is not None else "Accuracy: n/a")
    print(
        f"Weighted accuracy by gold weights: {weighted_gold:.6f}"
        if weighted_gold is not None
        else "Weighted accuracy by gold weights: n/a"
    )
    print(
        f"Weighted accuracy by dev weights: {weighted_pred:.6f}"
        if weighted_pred is not None
        else "Weighted accuracy by dev weights: n/a"
    )

    unequal_weights = [row["pair_id"] for row in rows if not bool(row["weight_equal"])]
    print(f"Weights differ on evaluated pairs: {len(unequal_weights)}")

    if missing_predictions:
        print("Missing pair ids:", ", ".join(missing_predictions[:20]))
        if len(missing_predictions) > 20:
            print(f"... and {len(missing_predictions) - 20} more")

    if extra_predictions:
        print("Extra pair ids:", ", ".join(extra_predictions[:20]))
        if len(extra_predictions) > 20:
            print(f"... and {len(extra_predictions) - 20} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
