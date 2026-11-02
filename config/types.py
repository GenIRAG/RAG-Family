from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def serialize_for_eval(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if value.__class__.__module__ == "numpy":
        if hasattr(value, "tolist"):
            return serialize_for_eval(value.tolist())
        if hasattr(value, "item"):
            return serialize_for_eval(value.item())
    if isinstance(value, dict):
        return {str(key): serialize_for_eval(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_for_eval(item) for item in value]
    if hasattr(value, "to_eval_dict") and callable(value.to_eval_dict):
        return value.to_eval_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return serialize_for_eval(value.to_dict())
    if hasattr(value, "__dict__"):
        return serialize_for_eval(vars(value))
    return str(value)


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = None
    score: Optional[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "contents": self.content,
            "metadata": self.metadata
        }

    def to_eval_dict(self) -> Dict[str, Any]:
        data = self.to_dict()
        if self.score is not None:
            data["score"] = self.score
        return serialize_for_eval(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        content = data.get("contents", data.get("content", ""))
        return cls(
            id=data.get("id", ""),
            content=content,
            metadata=data.get("metadata", {})
        )


@dataclass
class RetrieverResult:
    documents: List[Document]
    scores: List[float]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if len(self.documents) != len(self.scores):
            raise ValueError("Number of documents must match number of scores")

    def to_eval_dict(self) -> Dict[str, Any]:
        return {
            "documents": [document.to_eval_dict() for document in self.documents],
            "scores": serialize_for_eval(self.scores),
            "metadata": serialize_for_eval(self.metadata),
        }



@dataclass
class RAGOutput:
    query: str
    answer: str
    retrieval_result: Optional[RetrieverResult] = None
    metrics: Dict[str, float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.metadata is None:
            self.metadata = {}

    def get_retrieved_contexts(self) -> List[Document]:
        if self.retrieval_result and hasattr(self.retrieval_result, "documents"):
            return self.retrieval_result.documents
        return []


@dataclass
class EvalItem:
    query: str
    golden_answers: List[str]
    metadata: Dict[str, Any] = None
    id: Optional[str] = None
    choices: List[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_eval_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.query,
            "golden_answers": serialize_for_eval(self.golden_answers),
            "metadata": serialize_for_eval(self.metadata),
            "choices": serialize_for_eval(self.choices),
        }


@dataclass
class EvalPrediction:
    answer: str
    retrieval_result: Any = None
    metadata: Dict[str, Any] = None
    metric_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.metric_scores is None:
            self.metric_scores = {}

    def to_eval_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieval_result": serialize_for_eval(self.retrieval_result),
            "metadata": serialize_for_eval(self.metadata),
            "metric_scores": serialize_for_eval(self.metric_scores),
        }


@dataclass
class EvalRecord:
    sample: EvalItem
    prediction: EvalPrediction

    def to_eval_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample.id,
            "question": self.sample.query,
            "golden_answers": serialize_for_eval(self.sample.golden_answers),
            "pred": self.prediction.answer,
            "metric_scores": serialize_for_eval(self.prediction.metric_scores),
            "retrieval_result": serialize_for_eval(self.prediction.retrieval_result),
            "metadata": serialize_for_eval(self.sample.metadata),
            "output_metadata": serialize_for_eval(self.prediction.metadata),
            "sample": self.sample.to_eval_dict(),
            "prediction": self.prediction.to_eval_dict(),
        }
