from typing import Optional
import logging

from RoleRAG.models.models_base import BaseModel


class OpenAIGenerator(BaseModel):
    def __init__(
        self,
        model_name: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs
    ):
        self.model_name = model_name
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
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Error: {str(e)}"
        return "Error: Max retries exceeded"

    def clone(self) -> "OpenAIGenerator":
        return type(self)(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )


class DeepSeekGenerator(BaseModel):
    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs
    ):
        self.model_name = model_name
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
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"Error: {str(e)}"
        return "Error: Max retries exceeded"

    def clone(self) -> "DeepSeekGenerator":
        return type(self)(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
