from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from core.config import get_config
from core.llm import invoke_with_fallback
from rag.rag_pipeline import RAG


def _supports_sampling_params(model_name: str) -> bool:
    """GPT-5 proxy models reject explicit temperature/seed in the current API."""
    return not model_name.lower().startswith("openai/gpt-5")


@dataclass
class AgentAnswer:
    """Result returned by CausalAgent."""

    answer: str
    used_rag: bool = False
    rag_chunks: list[str] = field(default_factory=list)


class CausalAgent:
    """LLM agent with optional RAG context."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        api_key: Optional[str] = None,
    ) -> None:
        cfg = get_config()
        self.model_name = model or cfg.llm.model_name
        self.api_key = api_key or cfg.effective_llm_api_key
        self.top_p = top_p

        if _supports_sampling_params(self.model_name):
            self.temperature = (
                float(temperature)
                if temperature is not None
                else float(cfg.llm.temperature)
            )
            self.seed = seed
        else:
            self.temperature = None
            self.seed = None

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant for causal reasoning. "
                    "You answer causal questions carefully, identify the more plausible "
                    "direction of cause and effect, and separate evidence from assumptions. "
                    "If RAG context is provided, use it as the primary source of evidence. "
                    "If the context does not contain enough information, say so and rely on "
                    "general causal reasoning. "
                    "For A/B causal choice tasks, include a clear structural conclusion "
                    "immediately before the final answer tag in this format: "
                    "Conclusion: <cause> causes <effect>. "
                    "For A/B causal choice tasks, end your response with exactly one final tag: "
                    "<Answer>A</Answer> or <Answer>B</Answer>.",
                ),
                (
                    "user",
                    "Question:\n{question}\n\n"
                    "RAG context:\n{rag_context}\n\n"
                    "Which cause-and-effect relationship is more likely?\n\n"
                    "If the question provides options A and B, compare those two options. "
                    "Let's work this out in a step by step way to be sure that we have "
                    "the right answer. Then write a separate line in the format "
                    "Conclusion: <cause> causes <effect>. "
                    "Finally, provide your final answer within the tags <Answer>A/B</Answer>.",
                ),
            ]
        )

    def ask(
        self,
        question: str,
        *,
        use_rag: bool = False,
        rag_top_k: int = 3,
    ) -> AgentAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Empty question.")

        rag_chunks: list[str] = []
        if use_rag:
            logger.info("RAG enabled for query: {}", question)
            rag_chunks = RAG().perform_query(question, top_k=rag_top_k)

        rag_context = "\n\n---\n\n".join(rag_chunks) if rag_chunks else "Not provided."
        messages = self._prompt.format_messages(
            question=question,
            rag_context=rag_context,
        )

        content, _, used_json = invoke_with_fallback(
            messages,
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            seed=self.seed,
            top_p=self.top_p,
        )
        logger.info(
            "LLM answer generated: model={}, json_mode={}, rag_chunks={}",
            self.model_name,
            used_json,
            len(rag_chunks),
        )
        return AgentAnswer(
            answer=str(content).strip(),
            used_rag=use_rag,
            rag_chunks=rag_chunks,
        )
