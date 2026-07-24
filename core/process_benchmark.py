from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.env  # noqa: F401 - loads .env
from agent.agent_service import CausalAgent
from core.config import get_config


DEFAULT_BENCHMARK_DIR = ROOT / "data" / "benchmark_tubingen"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
REQUEST_SLEEP_SECONDS = 1.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5.0


@dataclass(frozen=True)
class MetaRow:
    pair_id: str
    cause_start: int
    cause_end: int
    effect_start: int
    effect_end: int
    weight: str

    @property
    def x_range(self) -> tuple[int, int]:
        """Column range for the X variable group.

        In the Tübingen CEP files, X is the first variable group and Y is the
        second variable group. pairmeta stores cause/effect ranges, so we recover
        X/Y ranges by column order after the model has already predicted.
        """
        ranges = [
            (self.cause_start, self.cause_end),
            (self.effect_start, self.effect_end),
        ]
        return min(ranges, key=lambda item: item[0])

    @property
    def y_range(self) -> tuple[int, int]:
        ranges = [
            (self.cause_start, self.cause_end),
            (self.effect_start, self.effect_end),
        ]
        return max(ranges, key=lambda item: item[0])


@dataclass(frozen=True)
class PairPayload:
    x_name: str
    y_name: str
    description: str
    data_summary: str


def read_meta(meta_path: Path) -> dict[str, MetaRow]:
    rows: dict[str, MetaRow] = {}
    for raw_line in meta_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        pair_id, cs, ce, es, ee, weight = parts[:6]
        rows[pair_id] = MetaRow(
            pair_id=pair_id,
            cause_start=int(cs),
            cause_end=int(ce),
            effect_start=int(es),
            effect_end=int(ee),
            weight=weight,
        )
    return rows


def description_without_ground_truth(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    description = re.split(r"(?im)^\s*ground\s*truth\b.*$", text, maxsplit=1)[0]
    description = re.sub(
        r"(?im)^\s*information\s+for\s+pairs?\d+\s*:\s*\n?",
        "",
        description,
    )
    return description.strip()


def extract_variable_name(description: str, variable: str) -> str:
    patterns = [
        rf"(?im)^\s*{variable}\s*[:=]\s*(.+)$",
        rf"(?im)^\s*{variable.upper()}\s*[:=]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            value = match.group(1).strip()
            if value:
                return re.sub(r"\s+", " ", value)
    return f"variable {variable.upper()}"


def summarize_pair_data(path: Path, *, max_rows: int = 5, max_cols: int = 12) -> str:
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        raw_head = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:max_rows])
        return f"Could not parse numeric table: {exc}\nRaw first rows:\n{raw_head}"

    shape = f"rows={len(df)}, columns={len(df.columns)}"
    preview = df.iloc[:max_rows, :max_cols].to_string(index=False, header=False)

    numeric = df.iloc[:, :max_cols].apply(pd.to_numeric, errors="coerce")
    stats = numeric.agg(["mean", "std", "min", "max"]).round(4)
    stats_text = stats.to_string()

    suffix = ""
    if len(df.columns) > max_cols:
        suffix = f"\nOnly the first {max_cols} columns are shown/summarized."

    return (
        f"Data shape: {shape}\n"
        f"First rows:\n{preview}\n\n"
        f"Column summary statistics:\n{stats_text}"
        f"{suffix}"
    )


def build_pair_payload(data_path: Path, desc_path: Path) -> PairPayload:
    description = description_without_ground_truth(desc_path)
    x_name = extract_variable_name(description, "x")
    y_name = extract_variable_name(description, "y")
    data_summary = summarize_pair_data(data_path)

    return PairPayload(
        x_name=x_name,
        y_name=y_name,
        description=description,
        data_summary=data_summary,
    )


def parse_answer(text: str) -> str | None:
    match = re.search(r"<Answer>\s*([AB])\s*</Answer>", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallbacks for imperfect model formatting.
    match = re.search(r"\bAnswer\s*[:=]\s*([AB])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"\b([AB])\b", text)
    return matches[-1].upper() if matches else None


def prediction_to_meta_line(meta: MetaRow, prediction: str) -> str:
    x_start, x_end = meta.x_range
    y_start, y_end = meta.y_range

    if prediction == "A":
        cause_start, cause_end = x_start, x_end
        effect_start, effect_end = y_start, y_end
    elif prediction == "B":
        cause_start, cause_end = y_start, y_end
        effect_start, effect_end = x_start, x_end
    else:
        raise ValueError(f"Unsupported prediction: {prediction!r}")

    return (
        f"{meta.pair_id} {cause_start} {cause_end} "
        f"{effect_start} {effect_end} {meta.weight}"
    )


def iter_pair_ids(benchmark_dir: Path) -> Iterable[str]:
    for path in sorted(benchmark_dir.glob("pair[0-9][0-9][0-9][0-9].txt")):
        yield path.stem.replace("pair", "")


def read_done_pair_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            done.add(parts[0])
    return done


def ask_agent_with_retry(
    agent: CausalAgent,
    payload: PairPayload,
    *,
    use_rag: bool,
    rag_top_k: int,
    use_recommendation: bool,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = agent.ask_causal_pair(
                x_name=payload.x_name,
                y_name=payload.y_name,
                description=payload.description,
                data_summary=payload.data_summary,
                use_rag=use_rag,
                rag_top_k=rag_top_k,
                use_recommendation=use_recommendation,
            )
            if REQUEST_SLEEP_SECONDS > 0:
                time.sleep(REQUEST_SLEEP_SECONDS)
            return result.answer
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break

            sleep_seconds = RETRY_SLEEP_SECONDS * (2 ** (attempt - 1))
            print(
                f"LLM request failed on attempt {attempt}/{MAX_RETRIES}: {exc}. "
                f"Retrying in {sleep_seconds:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the LLM agent over Tübingen cause-effect pairs."
    )
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--responses-output", type=Path, default=None)
    parser.add_argument("--errors-output", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None, help="First pair id, e.g. 1")
    parser.add_argument("--end", type=int, default=None, help="Last pair id, e.g. 108")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rag", action="store_true", help="Enable agent RAG")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument(
        "--no-recommendations",
        action="store_true",
        help="Disable recommendation generation in agent responses",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build pair inputs and metadata only; do not call the LLM.",
    )
    args = parser.parse_args()

    benchmark_dir: Path = args.benchmark_dir
    artifacts_dir = get_config().full_artifacts_dir or DEFAULT_ARTIFACTS_DIR
    if args.rag:
        default_output_path = artifacts_dir / "dev_rag_pairmeta.txt"
        default_responses_path = artifacts_dir / "dev_rag_predictions.jsonl"
        default_errors_path = artifacts_dir / "dev_rag_errors.jsonl"
    else:
        default_output_path = artifacts_dir / "dev_pairmeta.txt"
        default_responses_path = artifacts_dir / "dev_predictions.jsonl"
        default_errors_path = artifacts_dir / "dev_errors.jsonl"

    output_path = args.output or default_output_path
    responses_path = args.responses_output or default_responses_path
    errors_path = args.errors_output or default_errors_path
    meta_path = benchmark_dir / "pairmeta.txt"

    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"pairmeta.txt not found: {meta_path}")

    meta_by_id = read_meta(meta_path)
    done_ids = set() if args.no_resume else read_done_pair_ids(output_path)

    cfg = get_config()
    if not args.dry_run and not cfg.effective_llm_api_key:
        raise RuntimeError("LLM_API_KEY is not configured.")

    agent = None if args.dry_run else CausalAgent(model=cfg.llm.model_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with output_path.open("a", encoding="utf-8") as out_f, responses_path.open(
        "a", encoding="utf-8"
    ) as resp_f, errors_path.open("a", encoding="utf-8") as err_f:
        for pair_id in iter_pair_ids(benchmark_dir):
            pair_num = int(pair_id)
            if args.start is not None and pair_num < args.start:
                continue
            if args.end is not None and pair_num > args.end:
                continue
            if pair_id in done_ids:
                print(f"Skipping pair{pair_id}: already in {output_path.name}")
                continue
            if args.limit is not None and processed >= args.limit:
                break

            data_path = benchmark_dir / f"pair{pair_id}.txt"
            desc_path = benchmark_dir / f"pair{pair_id}_des.txt"
            if pair_id not in meta_by_id:
                print(f"Skipping pair{pair_id}: no metadata row")
                continue
            if not desc_path.exists():
                print(f"Skipping pair{pair_id}: no description file")
                continue

            print(f"Processing pair{pair_id}...")
            payload = build_pair_payload(data_path, desc_path)

            try:
                if args.dry_run:
                    answer_text = ""
                    parsed = "A"
                else:
                    assert agent is not None
                    answer_text = ask_agent_with_retry(
                        agent,
                        payload,
                        use_rag=args.rag,
                        rag_top_k=args.rag_top_k,
                        use_recommendation=not args.no_recommendations,
                    )
                    parsed = parse_answer(answer_text)

                if parsed not in {"A", "B"}:
                    raise ValueError(f"Could not parse <Answer>A/B</Answer>: {answer_text}")

                meta_line = prediction_to_meta_line(meta_by_id[pair_id], parsed)
                out_f.write(meta_line + "\n")
                out_f.flush()

                resp_f.write(
                    json.dumps(
                        {
                            "pair_id": pair_id,
                            "prediction": parsed,
                            "dev_pairmeta_line": meta_line,
                            "input": {
                                "x_name": payload.x_name,
                                "y_name": payload.y_name,
                                "description": payload.description,
                                "data_summary": payload.data_summary,
                            },
                            "response": answer_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                resp_f.flush()
                processed += 1

            except Exception as exc:
                error_payload = {
                    "pair_id": pair_id,
                    "error": str(exc),
                    "input": {
                        "x_name": payload.x_name,
                        "y_name": payload.y_name,
                        "description": payload.description,
                        "data_summary": payload.data_summary,
                    },
                }
                resp_f.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                resp_f.flush()
                err_f.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                err_f.flush()
                print(f"ERROR pair{pair_id}: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    raise

    print(f"Done. Processed pairs: {processed}")
    print(f"Predicted metadata: {output_path}")
    print(f"Responses/debug log: {responses_path}")
    print(f"Error report: {errors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
