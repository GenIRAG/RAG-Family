import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from RoleRAG.config.types import EvalItem, EvalPrediction, EvalRecord, RAGOutput, serialize_for_eval
from RoleRAG.rolerag.evaluator.metrics import BaseMetric, get_metric


class RAGEvaluator:

    def __init__(
        self,
        pipeline,
        metrics: Optional[List[str]] = None,
        show_progress: bool = True,
        config: Dict[str, Any] = None,
        save_dir: str = None,
        save_metric: bool = True,
        verbose: bool = False,
    ):
        self.pipeline = pipeline
        self.config = config or {}
        self.metrics_names = metrics or ["em", "f1"]
        self.show_progress = show_progress
        self.save_dir = save_dir
        self.save_metric_flag = save_metric
        self.verbose = verbose

        self.avaliable_metrics = self._collect_metrics()
        self.metric_class: Dict[str, BaseMetric] = {}
        for metric in self.metrics_names:
            metric_lower = metric.lower()
            try:
                if metric_lower in self.avaliable_metrics:
                    self.metric_class[metric] = self.avaliable_metrics[metric_lower](self.config)
                else:
                    self.metric_class[metric] = get_metric(metric, self.config)
            except Exception as exc:
                raise ValueError(f"Failed to initialize metric '{metric}': {exc}") from exc

    def _collect_metrics(self) -> Dict[str, type]:

        def find_descendants(base_class, subclasses=None):
            if subclasses is None:
                subclasses = set()
            for subclass in base_class.__subclasses__():
                if subclass not in subclasses:
                    subclasses.add(subclass)
                    find_descendants(subclass, subclasses)
            return subclasses

        avaliable_metrics = {}
        for cls in find_descendants(BaseMetric):
            metric_name = getattr(cls, "metric_name", cls.__name__.lower())
            avaliable_metrics[metric_name] = cls
        return avaliable_metrics

    def run(self, samples: List[EvalItem]) -> Dict[str, Any]:
        records: List[EvalRecord] = []
        log_records: List[Dict[str, Any]] = []
        overall_scores: Dict[str, float] = {}

        iterator = list(enumerate(samples))
        if self.show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Running pipeline")
            except ImportError:
                pass

        start_time = time.time()

        for idx, sample in iterator:
            self.pipeline.log_begin(sample)
            t_start = time.time()
            try:
                prediction = self._predict_sample(sample)
            finally:
                sample_elapsed = time.time() - t_start
                log_data = self.pipeline.log_end()

            record = EvalRecord(sample=sample, prediction=prediction)
            records.append(record)
            log_records.append({"record": record, "logs": log_data, "per_elapsed": sample_elapsed})

            if self.verbose:
                self._print_sample_result(idx, sample, prediction.answer, samples)

        elapsed = time.time() - start_time

        for metric_name, metric in self.metric_class.items():
            try:
                metric_result, metric_scores = metric.calculate_metric(records)
                overall_scores.update(metric_result)

                for record, score in zip(records, metric_scores):
                    record.prediction.metric_scores[metric_name] = score
            except Exception as e:
                raise RuntimeError(
                    f"Metric '{metric_name}' failed while evaluating "
                    f"{len(records)} samples: {e}"
                ) from e

        if self.save_dir and self.save_metric_flag:
            self._save_metric_score(overall_scores, num_samples=len(samples), elapsed=elapsed)

        self._save_intermediate_logs(
            log_records=log_records,
        )

        details = [
            RAGOutput(
                query=record.sample.query,
                answer=record.prediction.answer,
                retrieval_result=record.prediction.retrieval_result,
                metrics=record.prediction.metric_scores,
                metadata=record.prediction.metadata,
            )
            for record in records
        ]

        return {
            "summary": overall_scores,
            "details": details,
            "num_samples": len(samples),
            "elapsed_time": elapsed,
            "records": records,
            "eval_dataset": samples,
        }

    def _predict_sample(self, sample: EvalItem) -> EvalPrediction:
        try:
            output = self.pipeline.predict(sample)
        except Exception as e:
            sample_id = getattr(sample, "id", None) or "<unknown>"
            raise RuntimeError(
                f"Pipeline prediction failed for sample id={sample_id!r}, "
                f"query={sample.query!r}: {e}"
            ) from e

        if hasattr(output, "answer"):
            return EvalPrediction(
                answer=output.answer,
                retrieval_result=getattr(output, "retrieval_result", None),
                metadata=getattr(output, "metadata", {}) or {},
            )

        return EvalPrediction(answer=str(output))

    def _print_sample_result(self, idx: int, sample: EvalItem, prediction: str, dataset: List[EvalItem]):
        golden = sample.golden_answers[0] if sample.golden_answers else "N/A"
        query_text = sample.query[:60] + "..." if len(sample.query) > 60 else sample.query

        print()
        print("-" * 60)
        print(f"[{idx + 1}/{len(dataset)}] {query_text}")
        print("-" * 60)
        print(f"Prediction: {prediction[:100]}..." if len(prediction) > 100 else f"Prediction: {prediction}")
        print(f"Golden:     {golden[:100]}..." if len(golden) > 100 else f"Golden:     {golden}")

    def _save_metric_score(
        self,
        result_dict: Dict[str, float],
        num_samples: int = None,
        elapsed: float = None,
        filename: str = "metric_score.json",
    ):
        save_dir = self._resolve_log_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        payload_data = {"metrics": {k: round(v, 4) for k, v in result_dict.items()}}
        if num_samples is not None:
            payload_data["num_samples"] = num_samples
        if elapsed is not None:
            payload_data["elapsed_time"] = round(elapsed, 4)
        payload = serialize_for_eval(payload_data)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _resolve_log_save_dir(self) -> str:
        global_cfg = self.config.get("global", {}) or {}
        evaluation_cfg = global_cfg.get("evaluation", {}) or {}

        if self.save_dir:
            base_dir = self.save_dir
        else:
            if isinstance(evaluation_cfg, dict) and evaluation_cfg.get("save_dir"):
                base_dir = evaluation_cfg["save_dir"]
            else:
                base_dir = os.path.join("output", "evaluation_logs")

        data_file = evaluation_cfg.get("data_file") if isinstance(evaluation_cfg, dict) else None
        dataset_name = os.path.splitext(os.path.basename(data_file or "dataset"))[0]
        pipeline_name = global_cfg.get("pipeline_type") or self.pipeline.__class__.__name__

        folder_name = f"{self._slug_path_part(pipeline_name)}_{self._slug_path_part(dataset_name)}"
        return os.path.join(base_dir, folder_name)

    @staticmethod
    def _slug_path_part(value: Any) -> str:
        text = str(value or "unknown").strip().lower()
        text = re.sub(r"[^a-z0-9._-]+", "_", text)
        return text.strip("_") or "unknown"

    def _save_intermediate_logs(
        self,
        log_records: List[Dict[str, Any]],
        filename: str = "intermediate_logs.json",
    ):
        save_dir = self._resolve_log_save_dir()
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        samples = []
        for item in log_records:
            record = item["record"]
            samples.append(
                {
                    "id": record.sample.id,
                    "question": record.sample.query,
                    "golden_answers": record.sample.golden_answers,
                    "pred": record.prediction.answer,
                    "metric_scores": {
                        k: round(v, 4) for k, v in record.prediction.metric_scores.items()
                    } if record.prediction.metric_scores else {},
                    "logs": item.get("logs") or {},
                    "per_elapsed": round(item.get("per_elapsed"), 4) if item.get("per_elapsed") is not None else None,
                }
            )

        payload = serialize_for_eval(samples)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def print_results(self, results: Dict[str, Any]):
        from RoleRAG.rolerag.evaluator.metrics import MetricCategory

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        summary = results.get("summary", {})

        generation_scores = {}
        retrieval_scores = {}
        for metric_name, score in summary.items():
            if metric_name in self.metric_class:
                metric = self.metric_class[metric_name]
                if hasattr(metric, "category"):
                    if metric.category == MetricCategory.GENERATION:
                        generation_scores[metric_name] = score
                    else:
                        retrieval_scores[metric_name] = score
                elif "retrieval" in metric_name or "@" in metric_name:
                    retrieval_scores[metric_name] = score
                else:
                    generation_scores[metric_name] = score

        if generation_scores:
            print("\n  [Generation Metrics]")
            for metric_name, score in generation_scores.items():
                print(f"    {metric_name:24s}: {score:.4f}")

        if retrieval_scores:
            print("\n  [Retrieval Metrics]")
            for metric_name, score in retrieval_scores.items():
                print(f"    {metric_name:24s}: {score:.4f}")

        print("-" * 60)
        print(f"  Samples:     {results.get('num_samples', 0)}")
        print(f"  Time:        {results.get('elapsed_time', 0):.2f}s")
        print("=" * 60)
