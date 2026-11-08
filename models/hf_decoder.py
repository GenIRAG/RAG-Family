from typing import Optional

from RoleRAG.models.models_base import BaseModel


class HFDecoder(BaseModel):

    capabilities = frozenset({"generate"})

    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        max_length: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_fp16: bool = True,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        use_chat_template: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_fp16 = use_fp16
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token_id is not None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            self._model.generation_config.pad_token_id = self._tokenizer.pad_token_id
        if self.use_fp16:
            self._model = self._model.half()
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

    def generate(self, prompt: str, **kwargs) -> str:
        import torch
        max_tokens = kwargs.pop("max_tokens", self.max_new_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        pad_token_id = kwargs.pop("pad_token_id", self.tokenizer.pad_token_id)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                return_dict=True,
            )
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )

        model_device = getattr(self.model, "device", None)
        if model_device is not None and model_device.type != "meta":
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
        elif inputs["input_ids"].device.type == "cpu" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=pad_token_id,
                **kwargs
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def clone(self) -> "HFDecoder":
        return type(self)(
            model_name=self.model_name,
            model_path=self.model_path,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            use_fp16=self.use_fp16,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
            use_chat_template=self.use_chat_template,
        )
