from typing import Any, Dict, List, Optional


class LLMLinguaModel:
    
    requires_question = True
    _METHODS = {"llmlingua", "longllmlingua"}

    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        model_path: Optional[str] = None,
        device_map: str = "cpu",
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.device_map = device_map
        self.model_config = dict(model_config or {})
        self._compressor = None

    def load(self) -> None:
        if self._compressor is None:
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name=self.model_path,
                device_map=self.device_map,
                model_config=self.model_config.copy(),
            )

    def compress(
        self,
        prompt: List[str],
        question: str,
        method: str = "llmlingua",
        **kwargs,
    ) -> Dict[str, Any]:
        if method not in self._METHODS:
            raise ValueError(f"Unsupported LLMLingua method: {method}")
        if not question:
            raise ValueError("LLMLingua compression requires a non-empty question")

        self.load()
        return self._compressor.compress_prompt(
            context=prompt,
            question=question,
            rank_method=method,
            concate_question=False,
            **kwargs,
        )

    def clone(self) -> "LLMLinguaModel":
        return type(self)(
            model_name=self.model_name,
            model_path=self.model_path,
            device_map=self.device_map,
            model_config=self.model_config.copy(),
        )
