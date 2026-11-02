# 🚀 RoleRAG

<div align="center">

![Python](https://img.shields.io/badge/python-3.11.15-blue)
![RAG](https://img.shields.io/badge/RAG-Modular-brightgreen)
![Config](https://img.shields.io/badge/config-YAML%20%2B%20CLI-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

**RoleRAG** is a highly modular, configuration-driven Retrieval-Augmented Generation (RAG) experimentation and development framework. It fully decouples low-level model capabilities from application roles: the same `HFEncoder` can serve as the underlying model for a dense retriever or a graph retriever; the same `OpenAIGenerator` can take on the responsibilities of a reasoner, rewriter, or graph-extraction LLM.

The framework organizes experiment configuration through `global.yaml`, `index.yaml`, `config/methods/*.yaml`, and CLI override mechanisms, with 10 built-in pipelines including Naive RAG, Hybrid, Rerank, LLMLingua, LLMLingua-2, PERQA, HippoRAG, GraphRAG Local/Global, and UCGR, making it well suited for quickly reproducing experiments, combining new retrieval strategies, and validating cutting-edge RAG algorithms.

## 🧭 Architecture Overview

```text
CLI / YAML configuration
  -> Config
  -> create_pipeline()
  -> get_model() (create the underlying model)
  -> assist_role() (bind the role adapter)
  -> Pipeline (retrieval / generation strategy)
  -> RAGOutput
```

## ✨ Core Features

- 🧩 **Model / Role decoupled architecture**: `get_model()` returns only the underlying model, while `assist_role()` binds roles such as `retriever`, `reasoner`, `reranker`, `rewriter`, and `refiner` at runtime.
- ⚙️ **YAML + CLI configuration system**: multi-layer configuration is read from `config/` by default, and command-line arguments can temporarily override the corpus, pipeline, index model, evaluation dataset, and metrics.
- 🔎 **Configurable retrieval-only mode**: except for `ucgr`, a pipeline can return immediately after retrieval via the `retrieval_only` flag in its method YAML, making it convenient to evaluate retrieval performance independently.
- 📊 **Built-in evaluation pipeline**: unified `EvalItem` / `RAGOutput` / `EvalRecord` data structures, supporting metrics such as `em`, `f1`, `acc`, `recall@k`, `precision@k`, `mrr@k`, `ndcg@k`, `plan_score`, `step_score`, `pse_g`, and `acc4r`.

## 🤖 Supported Pipelines

`rolerag/pipelines/__init__.py` currently registers 10 pipelines:

| Pipeline name     | Description                                                  | Index / key dependency                                     |
| ----------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| `naive`           | Basic RAG: BM25 retrieval + HyDE query rewrite + LLM generation | `BM25Model`, `bm25_index_path`                            |
| `hybrid`          | Fuses BM25 and dense retrieval, merging results with RRF     | `BM25Model` + `HFEncoder`, BM25 index + dense FAISS index |
| `rerank`          | Re-ranks BM25 candidates with a cross-encoder                | `BM25Model` + `HFReranker`                                |
| `llmlingua`       | Compresses context with LLMLingua/LongLLMLingua after BM25 retrieval before generation | `BM25Model`, `bm25_index_path`, `LLMLinguaModel`           |
| `llmlingua2`      | Compresses context with LLMLingua-2 after BM25 retrieval before generation | `BM25Model`, `bm25_index_path`, `LLMLingua2Model`          |
| `perqa`           | Plan, execute, and candidate-graph aggregation flow for multi-hop questions | `BM25Model`, `bm25_index_path`                             |
| `hipporag`        | HippoRAG retrieval based on entity-relation graph and PPR, followed by ordinary generation | graph index, `HFEncoder`, `OpenAIGenerator`               |
| `graphrag_local`  | GraphRAG local retrieval, generating around local entities, text units, and community context | GraphRAG graph index, entity vector DB                    |
| `graphrag_global` | GraphRAG global retrieval, global aggregation and generation based on community reports | GraphRAG graph index, community reports                   |
| `ucgr`            | Constrained generative retrieval based on FMIndex, directly generating candidate passages/titles | SEAL FMIndex, `HFDecoder`                                 |

## 🚀 Quick Start

### Environment Setup

```bash
git clone <your-RoleRAG-repo-url>
cd RoleRAG
pip install -r requirements.txt
```

Building the FMIndex for UCGR depends on SEAL, so `seal.index.FMIndex` must be importable. `requirements.txt` already lists `SEAL==1.0`; if it cannot be installed directly in the current environment, refer to the installation instructions of [LogiCGR](https://github.com/GenIRAG/LogiCGR) to install SEAL separately and add the SEAL repository path to `PYTHONPATH`.

It is recommended to check the following configuration files before running:


| File                             | Purpose                                                                     |
| -------------------------------- | -------------------------------------------------------------------------- |
| `config/global.yaml`             | Global settings, model default parameters, default `pipeline_type`, common evaluation parameters |
| `config/index.yaml`              | Shared index construction parameters, including corpus, chunking, BM25, dense, FMIndex, and graph building |
| `config/methods/<pipeline>.yaml` | Index paths, `inference` parameters, `roles` parameters, and evaluation dataset for a single pipeline |

### Build Index

Build a dense index:

```bash
python main.py --build_index dense
python main.py --build_index dense --corpus data/hotpot_hippo.jsonl --indexing_model HFEncoder
```

Build a GraphRAG graph index:

```bash
python main.py --build_index graphrag
python main.py --build_index graphrag --corpus data/hotpot_hippo.jsonl --indexing_model HFEncoder
```

<details>
<summary>Build a BM25 Index</summary>

```bash
python main.py --build_index bm25
python main.py --build_index bm25 --corpus data/2wikimultihopqa_t1.jsonl
```

</details>

<details>
<summary>Build the FMIndex used by UCGR</summary>

```bash
python main.py --build_index fmindex
python main.py --build_index fmindex --corpus data/topiocqa_grpo_test_index.jsonl
```

</details>

Common build override parameters:

```bash
python main.py --build_index dense --storage indexes/storage --chunk --chunk_size 1200 --chunk_overlap 100
python main.py --build_index hipporag --indexing_model HFEncoder
```

Ordinary index names are generated by `make_index_name()`, such as `HFEncoder_hotpot_hippo_c1200_flat` and `BM25Model_hotpot_hippo_c1200`.

The default output root directory comes from `index.paths.storage_dir`, currently configured as `indexes/storage`.

### Run a Single Query

The index paths required to run a query must be configured in the
`index.paths` section of the corresponding `config/methods/<pipeline>.yaml` method configuration file; evaluation data is specified by `evaluation.data_file`, and the `--data_file` argument is only used to temporarily override that configuration.

Run Naive RAG:

```bash
python main.py --pipeline naive --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
```

Run GraphRAG Local:

```bash
python main.py --pipeline graphrag_local --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
```

More available examples:

```bash
python main.py --pipeline perqa --run_pipeline "What is the cause of death of director of film Vrijdag?"
python main.py --pipeline hipporag --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
python main.py --pipeline graphrag_global --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
python main.py --pipeline llmlingua --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
python main.py --pipeline llmlingua2 --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
python main.py --pipeline ucgr --run_pipeline "Did the company announce it's merger with any other company?"
```

Except for `ucgr`, if you only want to perform retrieval and skip generation, enable `retrieval_only` in the current method configuration:

```yaml
inference:
  naive:
    retrieval_only: true
```

GraphRAG Local/Global share the `graphrag` inference block:

```yaml
inference:
  graphrag:
    retrieval_only: true
```

After enabling this, the returned `RAGOutput.answer` is an empty string, `retrieval_result` retains the retrieval results, and `metadata.retrieval_only` is `true`. For evaluation, it is recommended to use retrieval metrics such as `recall@k`, `precision@k`, `mrr@k`, and `ndcg@k`.

### Evaluation

The index paths required for batch evaluation must be configured in the
`index.paths` section of the corresponding `config/methods/<pipeline>.yaml` method configuration file; evaluation data is specified by `evaluation.data_file`, and the `--data_file` argument is only used to temporarily override that configuration.

Use the default evaluation dataset in the pipeline configuration:

```bash
python main.py --evaluate
python main.py --pipeline graphrag_local --evaluate
python main.py --pipeline llmlingua --evaluate
python main.py --pipeline llmlingua2 --evaluate
python main.py --pipeline ucgr --evaluate
```

Temporarily override the sample count, evaluation dataset, and metrics:

```bash
python main.py --pipeline naive --evaluate --sample_num 10
python main.py --pipeline naive --evaluate --data_file data/evalute/hotpot_hippotest.jsonl --metrics em f1 acc recall@4
python main.py --pipeline ucgr --evaluate --sample_num 10 --metrics acc4r recall@5 mrr@5 ndcg@3
```

Evaluation output is saved by default to `global.evaluation.save_dir` (currently `results`), with actual directories named by pipeline and dataset:

```text
results/<pipeline>_<dataset>/metric_score.json
results/<pipeline>_<dataset>/intermediate_logs.json
```

`metric_score.json` is only generated when `save_metric: true`.

## ⚙️ Architecture and Configuration System

The core design idea of RoleRAG is to separate "what a model is" from "what role the model plays in a pipeline":

- **Model**: a low-level capability object, such as `BM25Model`, `HFEncoder`, `HFReranker`, `OpenAIGenerator`, `DeepSeekGenerator`, `HFDecoder`, `LLMLinguaModel`, or `LLMLingua2Model`.
- **Role**: a unified application-side interface, such as `retriever`, `reranker`, `reasoner`, `rewriter`, or `refiner`.

The current model implementations are split by capability under `models/`:


| File                            | Purpose                                                               |
| ------------------------------- | -------------------------------------------------------------------- |
| `models/BM25_model.py`          | BM25 low-level model                                                 |
| `models/hf_encoder.py`          | `HFEncoder` and `HFReranker`                                         |
| `models/api_decoder.py`         | API-based generation models, e.g. `OpenAIGenerator`, `DeepSeekGenerator` |
| `models/hf_decoder.py`          | HuggingFace decoder-only generation model                            |
| `models/hf_encoder_decoder.py`  | Reserved HuggingFace encoder-decoder generation model (not yet added to `_MODEL_REGISTRY`) |
| `models/llmlingua_model.py`     | `LLMLinguaModel`, LLMLingua/LongLLMLingua context compression        |
| `models/llmlingua2_model.py`    | `LLMLingua2Model`, LLMLingua-2 context compression                   |
| `models/models_base.py`         | Abstract model base class                                            |
| `models/utils.py`               | Model registry and `get_model()`                                     |

Prompt templates are managed by `rolerag/pipelines/prompts/prompt_manager.py`. Each prompt file only exposes a `PROMPTS` dictionary, and `PromptManager` handles automatic loading and template lookup by name.

The configuration entry point is `Config` in `config/config.py`. The main flow always reads the `config/` directory inside the project:

```text
CLI overrides > config/methods/<pipeline>.yaml > config/index.yaml > config/global.yaml
```

During initialization, `Config(config_dir, pipeline_name=args.pipeline)` merges the YAML files; `main.py` then writes the explicitly passed CLI arguments back into `config.final_config`.

The following shows how `get_model()` and `assist_role()` work together:

```python
from RoleRAG.models import get_model
from RoleRAG.roles.utils import assist_role

bm25_model = get_model("BM25Model")
retriever = assist_role(bm25_model, "retriever", {
    "index_path": "indexes/storage/BM25Model_hotpot_hippo_c1200",
})

reasoner_model = get_model("OpenAIGenerator")
reasoner = assist_role(reasoner_model, "reasoner", {
    "prompt_name": "naive_rag",
})
```

In a pipeline, the role parameters in the method YAML are usually read first in `__init__`, and model creation and role binding are then completed in `create()`:

```python
self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator")

pipeline.reasoner = assist_role(
    get_model("OpenAIGenerator"),
    "reasoner",
    pipeline.reasoner_cfg,
)
```

Model construction parameters are handled by `get_model()`: it reads `global.models` from `config/global.yaml` and overrides parameters with the same name using the explicitly passed kwargs. `assist_role()` does not read `global.yaml`; instead, it clones the underlying model and constructs the role adapter using the parameters in the method YAML `roles` block (such as `top_k`, `prompt_name`, `temperature`), without writing role parameters back into the model construction configuration.

Role parameters no longer use a separate `role_config.yaml`; instead, they are placed in the `roles` block of the corresponding `config/methods/<pipeline>.yaml`. Index paths remain under `index.paths` and are written into the corresponding role configuration by the pipeline's `__init__`.

## 🛠️ Extension Guide (Build Your Own)

<details>
<summary>👉 How to Add a New Model</summary>

1. Implement the low-level model class under `models/`. A generation model implements `generate()`, an encoder implements `encode()`, and a cross-encoder implements `rerank()`.
2. Import the class in `models/utils.py` and add it to `_MODEL_REGISTRY`.
3. Add a configuration block with the same name under `global.models` in `config/global.yaml`.
4. If the model will serve a role, add the model configuration under the corresponding role in the `roles` block of `config/methods/<pipeline>.yaml`.
5. If the model should be exported publicly, add the import and update `__all__` in `models/__init__.py`.

Minimal example:

```python
class MyEncoder:
    def __init__(self, model_name: str = "my-encoder", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size

    def encode(self, texts, is_query: bool = False, **kwargs):
        ...
```

Register it in `models/utils.py` (do not reassign the entire dictionary, or you will overwrite existing models):

```python
_MODEL_REGISTRY["MyEncoder"] = MyEncoder
```

Configuration example (placed under `global.models` in `config/global.yaml`):

```yaml
global:
  models:
    MyEncoder:
      model_name: "my-encoder"
      batch_size: 32
```

</details>

<details>
<summary>👉 How to Add a New Role</summary>

1. Add a new abstract contract in `roles/role_base.py`, defining the unified interface that pipelines will depend on.
2. Create the corresponding adapter file, e.g. `roles/my_role.py`, wrapping the low-level model into the role interface.
3. Import the adapter in `roles/utils.py`, add the role name to `_ALLOWED_ROLES`, and add a branch in `assist_role()`.
4. Add default role configuration under the `roles` block in `config/methods/<pipeline>.yaml`.
5. If the role should be exported publicly, add the import and update `__all__` in `roles/__init__.py`.

The currently available roles include:

```text
retriever, reranker, reasoner, rewriter, refiner
```

`assist_role()` branch illustration:

```python
if role == "reranker":
    return RerankerRoleAdapter(
        model=model,
        top_k=merged_config.get("top_k", 10),
        **kwargs
    )
```

</details>

<details>
<summary>👉 How to Create a New Pipeline</summary>

1. Create a new pipeline file under `rolerag/pipelines/`.
2. Inherit from `BasePipeline` and implement `create()` and `_run_item()`.
3. Call `register_pipeline("my_pipeline", MyPipeline)` at the bottom of the file.
4. Import the new pipeline in `rolerag/pipelines/__init__.py` to ensure the registration logic runs.
5. Add `config/methods/my_pipeline.yaml` and ensure the `inference` key matches the registered name.

Minimal example:

```python
from typing import Any, Dict

from RoleRAG.config.types import RAGOutput
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.models import get_model
from RoleRAG.roles.utils import assist_role, get_method_role_config


class MyPipeline(BasePipeline):
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.index_cfg = config["index"]
        self.paths_cfg = self.index_cfg["paths"]
        self.bm25_index_path = self.paths_cfg["bm25_index_path"]

        self.retriever_cfg = get_method_role_config(config, "retriever", "BM25Model")
        self.retriever_cfg["index_path"] = self.bm25_index_path
        self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator")

        self.retriever = None
        self.reasoner = None
        self.create()

    def create(self) -> None:
        self.retriever = assist_role(
            get_model("BM25Model"),
            "retriever",
            self.retriever_cfg,
        )
        self.reasoner = assist_role(
            get_model("OpenAIGenerator"),
            "reasoner",
            self.reasoner_cfg,
        )

    def _run_item(self, query: str, **kwargs) -> RAGOutput:
        retrieval_result = self.retriever.search(query)
        if self._is_retrieval_only():
            return self._retrieval_only_output(query, retrieval_result)

        # Minimal example: a real pipeline usually composes the prompt and context through a GenerationStrategy
        answer = self.reasoner.generate(query)
        return RAGOutput(query=query, answer=answer, retrieval_result=retrieval_result)


register_pipeline("my_pipeline", MyPipeline)
```

The corresponding `config/methods/my_pipeline.yaml` must provide the three blocks `index`, `inference`, and `evaluation`; `roles` is provided as the pipeline requires:

```yaml
index:
  paths:
    bm25_index_path: "indexes/storage/BM25Model_hotpot_hippo_c1200"

inference:
  my_pipeline:
    return_top_k: 5
    retrieval_only: false

roles:
  retriever:
    BM25Model:
      top_k: 5
  reasoner:
    OpenAIGenerator:
      prompt_name: "naive_rag"
      max_tokens: 256
      temperature: 0.0

evaluation:
  data_file: "data/evalute/hotpot_hippotest.jsonl"
  sample_num: 4
  random_sample: false
  metrics:
    - "em"
    - "f1"
    - "acc"
    - "recall@4"
```

</details>

## 📂 Data Formats

### Ordinary Corpus Format

Ordinary indexes and graph indexes read `.json`, `.jsonl`, or `.txt` files through `load_corpus_records()`. JSON/JSONL supports fields such as `title + context`, `content`, `contents`, and `text`.

```json
{"title": "Scott Derrickson", "context": "Scott Derrickson is an American director, screenwriter and producer."}
```

Or:

```json
{"id": "doc_1", "contents": "A standalone passage for retrieval.", "metadata": {"source": "demo"}}
```

### UCGR / FMIndex Data Format

FMIndex construction extracts `id` and `title` from the JSONL. `title` is encoded as constrained-generation candidates; if a title contains `" [SEP] "`, it is converted to `": "` during construction.

```json
{"id": "10178697", "title": "Dunkirk (2017 film): Release"}
```

### PERQA Data Format

The evaluation dataset is read from JSONL by `load_eval_samples()`, which always uses `question`, `golden_answers`, `metadata`, and `id`. PERQA corpus extraction and evaluation samples may include multi-hop supporting documents in `metadata.graph`; `plan_score`, `step_score`, and `pse_g` additionally depend on `metadata.plan`, and graph nodes must contain fields such as `template` and `answer`.

```json
{
  "id": "test_0",
  "question": "What is the cause of death of director of film Vrijdag?",
  "golden_answers": ["Euthanasia", "euthanasia"],
  "metadata": {
    "type": "compositional",
    "plan": {
      "Q1": ["Who is the director of the film Vrijdag?", "<A1>"],
      "Q2": ["What is the cause of death of <A1>?", "<A2>"]
    },
    "graph": [
      {
        "Q1": {
          "template": "Who is the director of the film Vrijdag?",
          "previous": [],
          "query": "Who is the director of the film Vrijdag?",
          "supports": [
            {"title": "Vrijdag", "contents": ["Vrijdag is a 1980 Belgian-Dutch drama film directed by Hugo Claus."]}
          ],
          "tag": "<A1>",
          "answer": "Hugo Claus"
        },
        "Q2": {
          "template": "What is the cause of death of <A1>?",
          "previous": ["Q1"],
          "query": "What is the cause of death of Hugo Claus?",
          "supports": [
            {"title": "Hugo Claus", "contents": ["His death by euthanasia, which is legal in Belgium, led to considerable controversy."]}
          ],
          "tag": "<A2>",
          "answer": "euthanasia"
        }
      }
    ]
  }
}
```

### Common Evaluation Format

```json
{"id": "test_0", "question": "Were Scott Derrickson and Ed Wood of the same nationality?", "golden_answers": ["yes"]}
```

`golden_answers` can be a string or a list; the loader normalizes it to `List[str]`.

## 🤝 Contribution and Acknowledgments

Contributions of new models, roles, or pipelines are welcome through issues / PRs. Before extending, please read the "Extension Guide" and ensure the new method has a complete configuration under `config/methods/` and can be verified through a single query and a minimal evaluation.

This project uses open-source libraries such as BM25, FAISS, Hugging Face Transformers, and Sentence Transformers in its implementation, and thanks open-source works such as HippoRAG, GraphRAG, PERQA, UCGR, and SEAL for the algorithms and implementation references they provide.
