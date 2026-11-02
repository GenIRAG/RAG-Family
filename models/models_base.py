from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def generate(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        pass


class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, texts: Union[str, List[str]], is_query: bool = False, **kwargs) -> np.ndarray:
        pass


class BaseCrossEncoder(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 10, **kwargs) -> List[Tuple[int, float]]:
        pass


class BaseSeq2Seq(ABC):
    @abstractmethod
    def generate(self, input_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        pass
