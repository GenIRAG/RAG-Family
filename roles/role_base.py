from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from RoleRAG.config.types import Document, RetrieverResult


class BaseRetrieverContract(ABC):

    @abstractmethod
    def search(self, query: str, **kwargs) -> RetrieverResult:

        pass


class BaseRerankerContract(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Union[List[Document], List[str]],
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[Union[Document, str], float]]:

        pass


class BaseGeneratorContract(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:

        pass


class BaseRewriterContract(ABC):

    @abstractmethod
    def rewrite(self, query: str, **kwargs) -> str:

        pass


class BaseRefinerContract(ABC):


    @abstractmethod
    def refine(self, context: Union[str, List[str]], **kwargs) -> str:

        pass
