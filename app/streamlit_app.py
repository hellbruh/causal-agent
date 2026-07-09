from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.env  # noqa: F401 - loads .env
from agent.agent_service import CausalAgent
from core.config import get_config
from core.llm_clients.factory import reset_llm_client
from core.utils import configure_logging


cfg = get_config()
configure_logging(cfg.streamlit_log_path, cfg.logging)

st.set_page_config(
    page_title=cfg.streamlit.page_title,
    page_icon=cfg.streamlit.page_icon,
    layout=cfg.streamlit.layout,
)

st.title("LLM + RAG")


def ensure_api_key() -> bool:
    if cfg.effective_llm_api_key:
        return True
    st.error("Не задан LLM_API_KEY в .env или переменных окружения.")
    return False


@st.cache_resource(show_spinner=False)
def get_agent(model_name: str, temperature: float) -> CausalAgent:
    reset_llm_client()
    return CausalAgent(model=model_name, temperature=temperature)


with st.sidebar:
    st.header("Настройки")
    model_name = st.text_input("Модель", value=cfg.llm.model_name)
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(cfg.llm.temperature),
        step=0.05,
    )
    rag_top_k = st.slider(
        "RAG top-k",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
    if model_name.lower().startswith("openai/gpt-5"):
        st.caption("Для gpt-5* temperature не отправляется в API.")


question = st.text_area(
    "Введите запрос",
    height=180,
    placeholder="Например: какие риски есть при запуске нового продукта?",
)
use_rag = st.checkbox("Использовать RAG", value=False)
submit = st.button("Отправить", type="primary")

if submit:
    cleaned_question = question.strip()
    if not cleaned_question:
        st.warning("Введите запрос.")
        st.stop()

    if not ensure_api_key():
        st.stop()

    try:
        agent = get_agent(model_name, float(temperature))
        with st.spinner("Генерирую ответ..."):
            result = agent.ask(
                cleaned_question,
                use_rag=use_rag,
                rag_top_k=int(rag_top_k),
            )

        st.markdown("### Ответ")
        st.write(result.answer)

        if result.used_rag:
            st.markdown("### RAG-контекст")
            if result.rag_chunks:
                for idx, chunk in enumerate(result.rag_chunks, start=1):
                    with st.expander(f"Фрагмент {idx}"):
                        st.write(chunk)
            else:
                st.info("RAG включен, но релевантные фрагменты не найдены.")

    except Exception as exc:
        logger.exception("Streamlit request failed")
        st.error(f"Ошибка: {exc}")
