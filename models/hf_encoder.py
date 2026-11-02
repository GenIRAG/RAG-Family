from typing import List, Union, Optional
import numpy as np

from RoleRAG.models.models_base import BaseEncoder, BaseCrossEncoder


class HFEncoder(BaseEncoder):

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        model_path: Optional[str] = None,
        max_length: int = 512,
        use_fp16: bool = False,
        batch_size: int = 32,
        instruction: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        if instruction is not None:
            self.instruction = instruction
        elif "e5" in self.model_name.lower():
            self.instruction = "query: "
        elif "bge" in self.model_name.lower():
            self.instruction = "Represent this sentence for searching relevant passages: "
        else:
            self.instruction = ""

    def _load_model(self):
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_path, device='cpu')
        if self.use_fp16:
            self._model = self._model.half()
        self._tokenizer = self._model.tokenizer

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

    def encode(self, texts: Union[str, List[str]], is_query: bool = False, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if is_query and self.instruction:
            texts = [self.instruction + t for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            **kwargs,
        )

    def get_embedding_dimension(self) -> int:
        if self._model is None:
            self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def clone(self) -> "HFEncoder":
        return type(self)(
            model_name=self.model_name,
            model_path=self.model_path,
            max_length=self.max_length,
            use_fp16=self.use_fp16,
            batch_size=self.batch_size,
            instruction=self.instruction,
        )


class HFReranker(BaseCrossEncoder):

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        model_path: Optional[str] = None,
        use_fp16: bool = False,
        max_length: int = 512,
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        if self.use_fp16:
            self._model = self._model.half()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device).eval()

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

    def rerank(self, query: str, documents: List[str], top_k: int = 10, **kwargs) -> List[tuple]:
        import torch
        pairs = [[query, doc] for doc in documents]
        device = next(self.model.parameters()).device

        with torch.inference_mode():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(device)
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu().numpy()

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [(documents[idx], float(score)) for idx, score in indexed_scores[:top_k]]

    def clone(self) -> "HFReranker":
        return type(self)(
            model_name=self.model_name,
            model_path=self.model_path,
            use_fp16=self.use_fp16,
            max_length=self.max_length,
        )
