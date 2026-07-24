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


def _recommendation_instruction(use_recommendation: bool) -> str:
    if use_recommendation:
        return (
            "Добавь отдельный раздел «Рекомендации» с 3-5 практическими "
            "рекомендациями на основе причинно-следственного вывода и доступного "
            "RAG-контекста. Если RAG-контекста недостаточно, явно отметь, что "
            "рекомендации являются выводом модели, а не фактом из источников."
        )
    return "Не добавляй раздел «Рекомендации»."


@dataclass
class AgentAnswer:
    """Result returned by CausalAgent."""

    answer: str
    used_rag: bool = False
    used_recommendation: bool = True
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
                    "Ты помощник для причинно-следственного анализа. "
                    "Отвечай на русском языке. Не галлюцинируй: не выдумывай "
                    "факты, источники, числа или причинные механизмы, которых нет "
                    "в вопросе, данных или RAG-контексте. "
                    "Аккуратно отвечай на вопросы о причинности, определяй более "
                    "правдоподобное направление причины и следствия, отделяй "
                    "доказательства от предположений. "
                    "Если предоставлен RAG-контекст, используй его как основной "
                    "источник доказательств. Если в контексте недостаточно "
                    "информации, прямо скажи об этом и опирайся на общие "
                    "причинно-следственные рассуждения. "
                    "Следуй инструкции по рекомендациям, указанной пользователем. "
                    "Всегда добавляй отдельную строку с самооценкой уверенности "
                    "модели в формате: Уверенность: <0-100>% - <краткая причина>. "
                    "Для задач выбора A/B добавляй ясный структурный вывод "
                    "непосредственно перед финальным тегом ответа в формате: "
                    "Вывод: <причина> вызывает <следствие>. "
                    "Для задач выбора A/B заканчивай ответ ровно одним финальным "
                    "тегом: <Answer>A</Answer> или <Answer>B</Answer>.",
                ),
                (
                    "user",
                    "Вопрос:\n{question}\n\n"
                    "RAG-контекст:\n{rag_context}\n\n"
                    "Инструкция по рекомендациям:\n{recommendation_instruction}\n\n"
                    "Какая причинно-следственная связь более вероятна?\n\n"
                    "Если в вопросе есть варианты A и B, сравни именно эти два "
                    "варианта. Дай краткое пошаговое обоснование, чтобы проверить "
                    "выбор направления причинности. Затем выполни инструкцию по "
                    "рекомендациям. Затем напиши отдельную строку «Уверенность». "
                    "Затем напиши отдельную строку в формате "
                    "Вывод: <причина> вызывает <следствие>. "
                    "В конце дай финальный ответ в тегах <Answer>A/B</Answer>.",
                ),
            ]
        )
        self._causal_pair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты помощник для причинно-следственного анализа. "
                    "Отвечай на русском языке. Не галлюцинируй: не выдумывай "
                    "факты, источники, числа или причинные механизмы, которых нет "
                    "в вопросе, данных или RAG-контексте. "
                    "Аккуратно отвечай на вопросы о причинности, определяй более "
                    "правдоподобное направление причины и следствия, отделяй "
                    "доказательства от предположений. "
                    "Если предоставлен RAG-контекст, используй его как основной "
                    "источник доказательств. Если в контексте недостаточно "
                    "информации, прямо скажи об этом и опирайся на общие "
                    "причинно-следственные рассуждения. "
                    "Следуй инструкции по рекомендациям, указанной пользователем. "
                    "Всегда добавляй отдельную строку с самооценкой уверенности "
                    "модели в формате: Уверенность: <0-100>% - <краткая причина>. "
                    "Для задач выбора A/B добавляй ясный структурный вывод "
                    "непосредственно перед финальным тегом ответа в формате: "
                    "Вывод: <причина> вызывает <следствие>. "
                    "Заканчивай ответ ровно одним финальным тегом: "
                    "<Answer>A</Answer> или <Answer>B</Answer>.",
                ),
                (
                    "user",
                    "Переменная X:\n{x_name}\n\n"
                    "Переменная Y:\n{y_name}\n\n"
                    "Описание:\n{description}\n\n"
                    "Сводка наблюдательных данных:\n{data_summary}\n\n"
                    "RAG-контекст:\n{rag_context}\n\n"
                    "Инструкция по рекомендациям:\n{recommendation_instruction}\n\n"
                    "Какая причинно-следственная связь более вероятна?\n\n"
                    "A. изменение {x_name} вызывает изменение {y_name}.\n"
                    "B. изменение {y_name} вызывает изменение {x_name}.\n\n"
                    "Дай краткое пошаговое обоснование, чтобы проверить выбор "
                    "направления причинности. Затем выполни инструкцию по "
                    "рекомендациям. Затем напиши отдельную строку «Уверенность». "
                    "Затем напиши отдельную строку в формате "
                    "Вывод: <причина> вызывает <следствие>. "
                    "В конце дай финальный ответ в тегах <Answer>A/B</Answer>.",
                ),
            ]
        )

    def ask(
        self,
        question: str,
        *,
        use_rag: bool = False,
        rag_top_k: int = 3,
        use_recommendation: bool = True,
    ) -> AgentAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Empty question.")

        rag_chunks: list[str] = []
        if use_rag:
            logger.info("RAG enabled for query: {}", question)
            rag_chunks = RAG().perform_query(question, top_k=rag_top_k)

        rag_context = "\n\n---\n\n".join(rag_chunks) if rag_chunks else "Не предоставлен."
        messages = self._prompt.format_messages(
            question=question,
            rag_context=rag_context,
            recommendation_instruction=_recommendation_instruction(use_recommendation),
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
            used_recommendation=use_recommendation,
            rag_chunks=rag_chunks,
        )

    def ask_causal_pair(
        self,
        *,
        x_name: str,
        y_name: str,
        description: str,
        data_summary: str,
        use_rag: bool = False,
        rag_top_k: int = 3,
        use_recommendation: bool = True,
    ) -> AgentAnswer:
        x_name = x_name.strip()
        y_name = y_name.strip()
        description = description.strip()
        data_summary = data_summary.strip()

        if not x_name or not y_name:
            raise ValueError("Empty variable name.")

        retrieval_query = "\n\n".join(
            part
            for part in (
                f"Переменная X: {x_name}",
                f"Переменная Y: {y_name}",
                description,
            )
            if part.strip()
        )

        rag_chunks: list[str] = []
        if use_rag:
            logger.info("RAG enabled for causal pair query: {}", retrieval_query)
            rag_chunks = RAG().perform_query(retrieval_query, top_k=rag_top_k)

        rag_context = "\n\n---\n\n".join(rag_chunks) if rag_chunks else "Не предоставлен."
        messages = self._causal_pair_prompt.format_messages(
            x_name=x_name,
            y_name=y_name,
            description=description or "Не предоставлено.",
            data_summary=data_summary or "Не предоставлена.",
            rag_context=rag_context,
            recommendation_instruction=_recommendation_instruction(use_recommendation),
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
            "LLM causal-pair answer generated: model={}, json_mode={}, rag_chunks={}",
            self.model_name,
            used_json,
            len(rag_chunks),
        )
        return AgentAnswer(
            answer=str(content).strip(),
            used_rag=use_rag,
            used_recommendation=use_recommendation,
            rag_chunks=rag_chunks,
        )
