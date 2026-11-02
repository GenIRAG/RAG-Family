from typing import Any, Dict, List, Optional


class LLMLingua2Model:

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        model_path: Optional[str] = None,
        device_map: str = "cpu",
        use_llmlingua2: bool = True,
        model_config: Optional[Dict[str, Any]] = None,
        llmlingua2_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.device_map = device_map
        self.use_llmlingua2 = use_llmlingua2
        self.model_config = dict(model_config or {})
        self.llmlingua2_config = dict(llmlingua2_config or {})
        self._compressor = None

    def load(self) -> None:
        if self._compressor is None:
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name=self.model_path,
                device_map=self.device_map,
                use_llmlingua2=self.use_llmlingua2,
                model_config=self.model_config.copy(),
                llmlingua2_config=self.llmlingua2_config.copy(),
            )

    def compress(self, prompt: List[str], **kwargs) -> Dict[str, Any]:
        self.load()
        return self._compressor.compress_prompt("\n\n".join(prompt), **kwargs)

    def clone(self) -> "LLMLingua2Model":
        return type(self)(
            model_name=self.model_name,
            model_path=self.model_path,
            device_map=self.device_map,
            use_llmlingua2=self.use_llmlingua2,
            model_config=self.model_config.copy(),
            llmlingua2_config=self.llmlingua2_config.copy(),
        )
