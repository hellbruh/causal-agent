from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.env  # noqa: F401 - loads .env
from agent.agent_service import CausalAgent
from core.config import get_config
from core.utils import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the configured LLM agent.")
    parser.add_argument("question", nargs="*", help="Question for the LLM agent")
    parser.add_argument("--use-rag", action="store_true", help="Use RAG context")
    parser.add_argument("--rag-top-k", type=int, default=3, help="Number of RAG chunks")
    args = parser.parse_args()

    cfg = get_config()
    configure_logging(
        cfg.pipeline_log_path,
        cfg.logging,
        add_stdout=True,
        stdout_format="{level} | {message}",
    )

    question = " ".join(args.question).strip()
    if not question:
        question = input("Введите запрос: ").strip()

    if not question:
        print("Запрос пустой.", file=sys.stderr)
        return 1

    if not cfg.effective_llm_api_key:
        print("Не задан LLM_API_KEY в .env или переменных окружения.", file=sys.stderr)
        return 1

    agent = CausalAgent(model=cfg.llm.model_name)
    result = agent.ask(
        question,
        use_rag=args.use_rag,
        rag_top_k=args.rag_top_k,
    )

    print("\nОтвет:\n")
    print(result.answer)

    if result.used_rag:
        print(f"\nRAG-фрагментов использовано: {len(result.rag_chunks)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
