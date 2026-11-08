from typing import Optional
import logging
import time

from RoleRAG.models.models_base import BaseModel

logger = logging.getLogger(__name__)


def _retryable_errors() -> tuple:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    return (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


class _ChatAPIGenerator(BaseModel):

    capabilities = frozenset({"generate"})
    default_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.model_name = model_name if model_name is not None else self.default_model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            for noisy_logger in ["httpx", "httpcore", "openai", "urllib3", "faiss", "faiss.loader", "sentence_transformers"]:
                logging.getLogger(noisy_logger).setLevel(logging.WARNING)
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=0,
            )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        retryable = _retryable_errors()

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
            except retryable as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "%s(%s) attempt %d/%d failed: %s; retrying in %ss",
                    type(self).__name__,
                    self.model_name,
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
            except Exception as exc:
                raise RuntimeError(
                    f"{type(self).__name__}({self.model_name}) generation failed "
                    f"(not retryable): {exc}"
                ) from exc

        raise RuntimeError(
            f"{type(self).__name__}({self.model_name}) generation failed after "
            f"{self.max_retries} attempt(s): {last_error}"
        ) from last_error

    def clone(self):
        return type(self)(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )


class OpenAIGenerator(_ChatAPIGenerator):
    pass


class DeepSeekGenerator(_ChatAPIGenerator):

    pass
