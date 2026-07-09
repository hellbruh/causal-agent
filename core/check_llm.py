"""Простая интерактивная проверка LLM через OpenAI-совместимый API."""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"В файле .env не задана переменная {name}")
    return value


def chat_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def ask_model(question: str) -> str:
    base_url = required_env("LLM_BASE_URL")
    api_key = required_env("LLM_API_KEY")
    model = os.getenv("LLM_MODEL_NAME", "gpt-5-nano").strip() or "gpt-5-nano"

    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": question}],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        chat_url(base_url),
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    try:
        choice = result["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "Сервер вернул ответ в неожиданном формате:\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
        ) from error

    if not content:
        finish_reason = choice.get("finish_reason", "не указан")
        usage = result.get("usage", {})
        raise RuntimeError(
            "Модель вернула пустой ответ. "
            f"Причина завершения: {finish_reason}. "
            f"Использование токенов: {json.dumps(usage, ensure_ascii=False)}"
        )

    return content


def main() -> int:
    load_dotenv()

    try:
        question = input("Что спросить у модели? ").strip()
        if not question:
            print("Вопрос не введён.")
            return 1

        print("\nОтвет модели:\n")
        print(ask_model(question))
        return 0
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"\nОшибка API: HTTP {error.code}\n{details}", file=sys.stderr)
    except URLError as error:
        print(f"\nНе удалось подключиться к LLM: {error.reason}", file=sys.stderr)
    except (ValueError, RuntimeError) as error:
        print(f"\nОшибка: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nПроверка прервана.")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
