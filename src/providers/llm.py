"""LLM provider interface. Providers may propose text but never verify claims."""

from abc import ABC, abstractmethod
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    text: str
    model: str
    raw_response_id: str | None = None


class LLMProvider(ABC):
    """Text generation only; no verification-status method exists."""

    provider_name: str

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """Text-only provider. It cannot set verification statuses."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0,
    ) -> LLMResponse:
        if not self.api_key:
            raise NotImplementedError(
                "An OpenAI API key is required for optional Phase 2 extraction."
            )

        from openai import OpenAI

        response = OpenAI(api_key=self.api_key).chat.completions.create(
            model=self.model,
            messages=[message.model_dump() for message in messages],
            temperature=temperature,
        )

        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            raw_response_id=response.id,
        )


class OllamaProvider(LLMProvider):
    """Native Ollama HTTP client."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:0.5b",
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0,
    ) -> LLMResponse:
        prompt = "\n\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in messages
        )

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
        ).encode("utf-8")

        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=300) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )
            raise RuntimeError(
                f"Ollama native generation failed "
                f"({error.code}): {detail}"
            ) from error

        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Ollama native generation failed: {error}"
            ) from error

        output = data.get("response")

        if not isinstance(output, str):
            raise RuntimeError(
                "Ollama native generation returned no string "
                "'response' field."
            )

        return LLMResponse(
            text=output,
            model=self.model,
        )


class GeminiProvider(LLMProvider):
    """Google Gemini text generation; never performs verification."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0,
    ) -> LLMResponse:
        if not self.api_key:
            raise NotImplementedError(
                "A Gemini API key is required for Gemini extraction."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        prompt = "\n\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in messages
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )

        output = response.text or ""

        if not output:
            raise RuntimeError(
                "Gemini generation returned no text."
            )

        return LLMResponse(
            text=output,
            model=self.model,
        )


def create_llm_provider(
    *,
    provider: str,
    openai_api_key: str | None,
    openai_model: str,
    ollama_base_url: str,
    ollama_model: str,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-2.5-flash",
) -> LLMProvider:
    """Construct the configured extraction-only provider."""

    if provider.lower() == "ollama":
        return OllamaProvider(
            base_url=ollama_base_url,
            model=ollama_model,
        )

    if provider.lower() == "openai":
        return OpenAIProvider(
            api_key=openai_api_key,
            model=openai_model,
        )

    if provider.lower() == "gemini":
        return GeminiProvider(
            api_key=gemini_api_key,
            model=gemini_model,
        )

    raise ValueError(
        "LLM_PROVIDER must be 'ollama', 'openai', or 'gemini'."
    )