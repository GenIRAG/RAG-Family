from typing import List, Union, Optional


class HFEncoderDecoder:

    capabilities = frozenset({"generate"})

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        model_path: Optional[str] = None,
        max_length: int = 512,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_fp16: bool = True,
        mirror_url: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_fp16 = use_fp16
        self.mirror_url = mirror_url
        self._model = None
        self._tokenizer = None

    @classmethod
    def from_config(cls, config) -> "HFEncoderDecoder":
        enc_dec_config = config.get("encoder_decoder", {})
        global_cfg = config.get("global", {}) or {}
        return cls(
            model_name=config.get("llm_model", "google/flan-t5-base"),
            model_path=config.get("llm_model_path"),
            max_length=enc_dec_config.get("max_length", 512),
            max_new_tokens=enc_dec_config.get("max_new_tokens", 256),
            temperature=enc_dec_config.get("temperature", 0.0),
            use_fp16=enc_dec_config.get("use_fp16", True),
            mirror_url=global_cfg.get("hf_mirror"),
        )

    def _load_model(self):
        if self.mirror_url:
            import os
            os.environ["HF_ENDPOINT"] = self.mirror_url

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        if self.use_fp16:
            self._model = self._model.half()
        else:
            self._model = self._model.float()

        self._model.eval()

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def generate(
        self,
        input_text: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        import torch

        if isinstance(input_text, str):
            input_text = [input_text]
            single_input = True
        else:
            single_input = False

        max_new_tokens = kwargs.pop("max_tokens", self.max_new_tokens)
        temperature = kwargs.pop("temperature", self.temperature)

        responses = []
        for text in input_text:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    **kwargs
                )

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            responses.append(generated_text.strip())

        return responses[0] if single_input else responses

    def encode(self, texts: Union[str, List[str]], **kwargs):
        raise NotImplementedError("Encoder-Decoder models use generate() not encode()")

    def encode_queries(self, queries: Union[str, List[str]], **kwargs):
        raise NotImplementedError("Encoder-Decoder models use generate() not encode_queries()")

    def encode_documents(self, documents: List[str], **kwargs):
        raise NotImplementedError("Encoder-Decoder models use generate() not encode_documents()")
