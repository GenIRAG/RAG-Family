# Project Description

The following content is generated automatically from the project's current files. Its purpose is to help an external LLM that cannot access the local repository understand RoleRAG and assist with code changes after the user confirms a plan.

## Instructions

You are a senior code modification assistant. You cannot access my local project directly and must work only from the project context and file contents I provide.

Follow these rules strictly:

1. Understand the project before offering code.
2. First provide a "Project Understanding Summary" and a "Change Plan."
3. Wait for my confirmation before outputting code.
4. Group code by file path so I can place it back into the local project.
5. If required file contents are missing, explicitly ask me for the relevant paths and explain why they are needed. Do not invent them.
6. Prefer the project's existing structure, naming, configuration conventions, and dependencies.
7. Do not perform unrelated refactoring.
8. If the change spans several files, explain why each file must change.
9. After outputting code, provide executable verification commands.
10. Unless explicitly required, do not add compatibility for legacy configuration or interfaces, automatically switch strategies, roles, models, or prompts, or fall back to another implementation on exceptions or empty results. Explicit parameter defaults within the same component may remain.
11. Unless the user explicitly requests reuse, do not use HyDE, the naive prompt, `NaiveGenerationStrategy`, or another existing implementation as a default substitute. Any reuse must be described in the plan with its purpose and necessity, and must wait for user confirmation.

## Project Name

🚀 RoleRAG

## Current Project Tree

```
- config/
  - config/methods/
    - config/methods/graphrag_global.yaml
    - config/methods/graphrag_local.yaml
    - config/methods/hipporag.yaml
    - config/methods/hybrid.yaml
    - config/methods/llmlingua.yaml
    - config/methods/llmlingua2.yaml
    - config/methods/naive.yaml
    - config/methods/perqa.yaml
    - config/methods/rerank.yaml
    - config/methods/ucgr.yaml
  - config/__init__.py
  - config/config.py
  - config/global.yaml
  - config/index.yaml
  - config/types.py
- indexes/
  - indexes/chunking/
    - indexes/chunking/__init__.py
    - indexes/chunking/chunking_base.py
    - indexes/chunking/fixedsize_chunking.py
    - indexes/chunking/sentence_chunking.py
    - indexes/chunking/tokensize_chunking.py
  - indexes/indexing/
    - indexes/indexing/chunk_indexing/
      - indexes/indexing/chunk_indexing/__init__.py
      - indexes/indexing/chunk_indexing/bm25_indexing.py
      - indexes/indexing/chunk_indexing/chunk_indexing_base.py
      - indexes/indexing/chunk_indexing/dense_indexing.py
      - indexes/indexing/chunk_indexing/fm_indexing.py
    - indexes/indexing/graph_indexing/
      - indexes/indexing/graph_indexing/extraction/
        - indexes/indexing/graph_indexing/extraction/__init__.py
        - indexes/indexing/graph_indexing/extraction/graphrag_extraction.py
        - indexes/indexing/graph_indexing/extraction/hipporag_extraction.py
      - indexes/indexing/graph_indexing/__init__.py
      - indexes/indexing/graph_indexing/community.py
      - indexes/indexing/graph_indexing/extraction_schema.py
      - indexes/indexing/graph_indexing/graph_builder.py
      - indexes/indexing/graph_indexing/graph_indexer.py
      - indexes/indexing/graph_indexing/universal_graph.py
  - indexes/loading/
    - indexes/loading/__init__.py
    - indexes/loading/chunk_loading.py
    - indexes/loading/graph_loading.py
  - indexes/__init__.py
  - indexes/utils.py
- models/
  - models/__init__.py
  - models/api_decoder.py
  - models/BM25_model.py
  - models/hf_decoder.py
  - models/hf_encoder.py
  - models/hf_encoder_decoder.py
  - models/llmlingua2_model.py
  - models/llmlingua_model.py
  - models/models_base.py
  - models/utils.py
- rolerag/
  - rolerag/evaluator/
    - rolerag/evaluator/datasets/
      - rolerag/evaluator/datasets/__init__.py
    - rolerag/evaluator/evaluators/
      - rolerag/evaluator/evaluators/__init__.py
      - rolerag/evaluator/evaluators/rag_evaluator.py
    - rolerag/evaluator/metrics/
      - rolerag/evaluator/metrics/__init__.py
    - rolerag/evaluator/__init__.py
  - rolerag/pipelines/
    - rolerag/pipelines/generation_strategy/
      - rolerag/pipelines/generation_strategy/__init__.py
      - rolerag/pipelines/generation_strategy/graphrag_generation_strategy.py
      - rolerag/pipelines/generation_strategy/naive_generation_strategy.py
      - rolerag/pipelines/generation_strategy/perqa_generation_strategy.py
      - rolerag/pipelines/generation_strategy/refine_generation_strategy.py
    - rolerag/pipelines/prompts/
      - rolerag/pipelines/prompts/__init__.py
      - rolerag/pipelines/prompts/graphrag_prompt.py
      - rolerag/pipelines/prompts/hipporag_prompt.py
      - rolerag/pipelines/prompts/hyde_prompt.py
      - rolerag/pipelines/prompts/naive_prompt.py
      - rolerag/pipelines/prompts/perqa_prompt.py
      - rolerag/pipelines/prompts/prompt_manager.py
      - rolerag/pipelines/prompts/ucgr_prompt.py
    - rolerag/pipelines/retrieval_strategy/
      - rolerag/pipelines/retrieval_strategy/__init__.py
      - rolerag/pipelines/retrieval_strategy/basic_retrieval_strategy.py
      - rolerag/pipelines/retrieval_strategy/graphrag_retrieval_strategy.py
      - rolerag/pipelines/retrieval_strategy/hipporag_retrieval_strategy.py
      - rolerag/pipelines/retrieval_strategy/hybrid_retrieval_strategy.py
      - rolerag/pipelines/retrieval_strategy/perqa_retrieval_strategy.py
      - rolerag/pipelines/retrieval_strategy/ucgr_retrieval_strategy.py
    - rolerag/pipelines/__init__.py
    - rolerag/pipelines/graphrag_global_pipeline.py
    - rolerag/pipelines/graphrag_local_pipeline.py
    - rolerag/pipelines/hipporag_pipeline.py
    - rolerag/pipelines/hybrid_pipeline.py
    - rolerag/pipelines/llmlingua2_pipeline.py
    - rolerag/pipelines/llmlingua_pipeline.py
    - rolerag/pipelines/naive_pipeline.py
    - rolerag/pipelines/perqa_pipeline.py
    - rolerag/pipelines/pipelines_base.py
    - rolerag/pipelines/rerank_pipeline.py
    - rolerag/pipelines/ucgr_pipeline.py
- roles/
  - roles/__init__.py
  - roles/reasoner.py
  - roles/refiner.py
  - roles/reranker.py
  - roles/retriever.py
  - roles/rewriter.py
  - roles/role_base.py
  - roles/utils.py
- COMMANDS.md
- README.md
- main.py
```

## Architecture Overview

RoleRAG is a configuration-driven RAG experimentation framework. When an external model reads this document, it should first understand the relationships among the entry point, configuration, pipeline, model, role, strategy, and evaluator layers before deciding how to change the code.

### Core Execution Flow

```text
main.py
-> Config(config_dir, pipeline_name)
-> config/global.yaml + config/index.yaml + config/methods/<pipeline>.yaml
-> create_pipeline(config.final_config)
-> Pipeline.__init__ / create()
-> get_model(model_name)
-> assist_role(model, role, role_config)
-> pipeline.run(query) / pipeline.evaluate(dataset)
```

### Module Layers

- `main.py`: The command-line entry point. It dispatches index construction, single-query inference, and batch evaluation.
- `config/`: Configuration loading, configuration merging, and cross-module data structures.
- `models/`: Low-level model capabilities. This layer expresses only what a model can do, such as `encode`, `generate`, or `rerank`.
- `roles/`: The role adapter layer. It wraps low-level models in application-facing interfaces such as retriever, reasoner, reranker, rewriter, and refiner. The refiner compresses or cleans retrieved context before generation.
- `rolerag/pipelines/`: Composes roles and strategies to define the execution flow of each RAG method.
- `rolerag/pipelines/retrieval_strategy/`: Retrieval strategy implementations.
- `rolerag/pipelines/generation_strategy/`: Generation strategy implementations.
- `rolerag/pipelines/prompts/`: Prompt template registration and lookup.
- `indexes/`: Construction and loading of standard indexes, graph indexes, and FMIndex artifacts.
- `rolerag/evaluator/`: Evaluation data loading, metric computation, and log persistence.

### Registered Pipelines

- Core and context-processing pipelines: `naive`, `hybrid`, `rerank`, `llmlingua`, and `llmlingua2`.
- Multi-hop, graph, and generative-retrieval pipelines: `perqa`, `hipporag`, `graphrag_local`, `graphrag_global`, and `ucgr`.
- When changing a pipeline, treat its import in `rolerag/pipelines/__init__.py`, its trailing `register_pipeline()` call, and its matching `config/methods/<pipeline>.yaml` as authoritative. Do not infer role composition or strategy behavior from a pipeline name alone.


## Project Directory and File Guide

RoleRAG is a modular, configuration-driven RAG experimentation framework. It separates low-level model implementations from application roles, builds or loads BM25, dense, graph, and FM indexes, assembles one of ten registered pipelines through YAML plus CLI configuration, runs retrieval/generation or evaluation, and persists metrics and intermediate logs. The collected snapshot includes the CLI entry point, configuration system, index construction and loading code, model wrappers, role adapters, retrieval/generation strategies, prompt templates, evaluator, metrics, and documentation. No web server or REST service appears in the snapshot; the API-facing component is the OpenAI-compatible generator client in `models/api_decoder.py`.

### Top-Level Files

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `main.py` | CLI entry point. Parses arguments, constructs `Config`, applies CLI overrides, dispatches index building, single-query pipeline execution, and evaluation. Also registers graph extraction methods and orchestrates graph-index construction. | New CLI flags, build targets, override semantics, graph-building workflow, pipeline dispatch, evaluation dispatch, import path setup. |
| `README.md` | User-facing architecture and usage documentation: config layers, model/role decoupling, pipeline list, index commands, evaluation commands, extension guides, and data formats. | Public behavior, documented commands, extension contracts, data-format expectations, pipeline descriptions. |
| `COMMANDS.md` | Concise command cookbook for building indexes, running pipelines, and evaluating methods. | Quick-start examples and CLI regression checks. |

### `config/` — Configuration Layer

The configuration layer merges YAML files and receives CLI overrides. `Config` reads `global.yaml`, `index.yaml`, and `config/methods/<pipeline>.yaml`, then exposes a dot-path mapping used by pipelines, roles, models, and evaluation.

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `config/__init__.py` | Re-exports `Config`, `Document`, `RetrieverResult`, `RAGOutput`, and `get_model`. | Public config/type exports. |
| `config/config.py` | Implements recursive dictionary merging and `Config`. Loads global/index/method YAML, requires method `index`, `inference`, and `evaluation` blocks, merges method index over shared index config, merges method evaluation over global evaluation, sets `pipeline_type`, supports dot-path `get`/`__setitem__`, and sets `HF_ENDPOINT` from `global.hf_mirror`. | Config precedence, required config keys, method-config schema, CLI override behavior, Hugging Face mirror behavior. |
| `config/global.yaml` | Global settings: seed, HF mirror placeholder, model defaults for BM25, HF encoder, OpenAI/DeepSeek generators, HF decoder, reranker, LLMLingua models, default `pipeline_type`, and evaluation output/verbosity settings. | Model default parameters, API keys/base URLs, mirror URL, default pipeline, evaluation save directory and display behavior. |
| `config/index.yaml` | Shared index configuration: storage root, corpus path, chunking, indexing model, dense/FAISS settings, BM25 parameters, FMIndex settings, and graph-building extraction/method configuration for HippoRAG and GraphRAG. | Corpus/storage paths, chunk size/overlap, FAISS type and search parameters, FMIndex tokenizer settings, graph extraction concurrency/retries, HippoRAG/GraphRAG index options. |
| `config/types.py` | Dataclasses for `Document`, `RetrieverResult`, `RAGOutput`, `EvalItem`, `EvalPrediction`, and `EvalRecord`; serialization helper for evaluation logs. | Data contracts between loaders, retrievers, pipelines, evaluator, and log persistence. |
| `config/methods/naive.yaml` | Naive RAG index path, inference return count, BM25 retriever, HyDE rewriter, naive reasoner, and HotpotQA evaluation. | Naive pipeline role parameters and evaluation dataset/metrics. |
| `config/methods/hybrid.yaml` | BM25 and dense index paths, RRF parameters, BM25/HFEncoder retrievers, reasoner, and evaluation. | Hybrid retrieval fusion and index-path changes. |
| `config/methods/rerank.yaml` | BM25 path, rerank return counts, BM25 retriever, HF reranker, reasoner, and evaluation. | Rerank candidate counts and reranker role settings. |
| `config/methods/llmlingua.yaml` | BM25 path, LLMLingua inference config, BM25 retriever, LLMLingua refiner with compression rate and method, reasoner, and evaluation. | Switching LLMLingua vs LongLLMLingua and compressor parameters. |
| `config/methods/llmlingua2.yaml` | BM25 path, LLMLingua-2 inference config, BM25 retriever, LLMLingua-2 refiner, reasoner, and evaluation. | LLMLingua-2 compression settings. |
| `config/methods/perqa.yaml` | BM25 index for 2WikiMultiHopQA, PERQA inference settings, BM25 retriever, planner/executor/answer reasoner prompt-name groups, and PERQA evaluation metrics. | PERQA planning, execution, aggregation, prompt grouping, and metric selection. |
| `config/methods/hipporag.yaml` | Graph and dense index paths, PPR settings, HFEncoder retriever, entity-extraction LLM and answer reasoner sub-configs, and evaluation. | HippoRAG graph/dense paths, PPR damping/top-k, entity LLM behavior. |
| `config/methods/graphrag_local.yaml` | GraphRAG graph index path, shared `inference.graphrag` block, HFEncoder retriever, local QA reasoner, and evaluation. | Local GraphRAG retrieval budgets, retriever top-k, local QA prompt. |
| `config/methods/graphrag_global.yaml` | GraphRAG graph index path, shared `inference.graphrag` block, global reduce-QA reasoner, and evaluation. | Global community budgets, response format, global reduce prompt. |
| `config/methods/ucgr.yaml` | FMIndex path, UCGR inference beam/token settings, HFDecoder reasoner, and retrieval-oriented evaluation metrics. | UCGR constrained-generation settings and FMIndex path. |

### `indexes/` — Loading, Chunking, and Index Construction

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `indexes/__init__.py` | Re-exports loaders, chunkers, `BaseIndexing`, `make_index_name`, and lazy `BM25Indexing`/`DenseIndexing`/`FMIndexing`. | Public index package API and lazy-import behavior. |
| `indexes/utils.py` | Builds standard index names from model, corpus, chunk size, and FAISS type. | Index directory naming and path compatibility. |
| `indexes/chunking/chunking_base.py` | Abstract chunking interface. | New chunking implementations. |
| `indexes/chunking/fixedsize_chunking.py` | Character/paragraph-based fixed-size chunking with overlap and parent/chunk metadata. | Fixed-size chunking behavior. |
| `indexes/chunking/sentence_chunking.py` | NLTK sentence tokenization and sentence-group chunking. | Sentence-based chunking and NLTK dependency behavior. |
| `indexes/chunking/tokensize_chunking.py` | Token-based chunking using tiktoken when available, with overlap, forced splits, and fallback character approximation. | Default chunking behavior used by `main.py`, chunk size/overlap semantics. |
| `indexes/indexing/chunk_indexing/chunk_indexing_base.py` | Abstract `build_index`, `search`, `save`, and `load` interface. | Chunk-index backend contracts. |
| `indexes/indexing/chunk_indexing/bm25_indexing.py` | BM25 index build/search/save/load using `bm25s` and Stemmer; saves corpus and BM25 params. | BM25 index format, k1/b, search behavior. |
| `indexes/indexing/chunk_indexing/dense_indexing.py` | Dense FAISS index build/search/save/load using an encoder; supports FLAT, IVF, and HNSW; stores documents and metadata. | Dense index format, FAISS behavior, encoder batching, score normalization. |
| `indexes/indexing/chunk_indexing/fm_indexing.py` | SEAL FMIndex construction for UCGR from JSONL `id`/`title` entries; tokenizer worker setup, multiprocessing, save/load, FM path helper. | FMIndex corpus format, tokenization, SEAL dependency, UCGR index paths. |
| `indexes/indexing/chunk_indexing/__init__.py` | Exports `BaseIndexing` and lazily imports BM25/Dense/FM indexers. | Chunk-index package exports. |
| `indexes/indexing/graph_indexing/extraction_schema.py` | Dataclasses for entities, relationships, chunk extraction, and extraction result. | Graph extraction data contracts. |
| `indexes/indexing/graph_indexing/extraction/hipporag_extraction.py` | HippoRAG NER/OpenIE extraction, JSON parsing, retry handling, concurrency, graph-item construction, and empty-extraction validation. | HippoRAG extraction prompts, retry/concurrency, entity/triple parsing. |
| `indexes/indexing/graph_indexing/extraction/graphrag_extraction.py` | GraphRAG entity/relationship extraction with delimited records, gleaning, retries, parsing, and diagnostics. | GraphRAG tuple/record format, gleaning count, extraction prompts. |
| `indexes/indexing/graph_indexing/extraction/__init__.py` | Exports HippoRAG and GraphRAG extraction classes and errors. | Extraction registry imports. |
| `indexes/indexing/graph_indexing/universal_graph.py` | Unified NetworkX-backed graph wrapper with node/edge defaults, accessors, topology helpers, pickle, and GraphML export. | Graph APIs used by builders, indexers, retrieval strategies, and serialization. |
| `indexes/indexing/graph_indexing/graph_builder.py` | Merges extraction entities and relationships into a `UniversalGraph`, deduplicating descriptions/source IDs and grouping edges. | Entity/relationship merge rules and graph construction. |
| `indexes/indexing/graph_indexing/graph_indexer.py` | Builds entity/relation vector DBs, graph augmentation, e2r/r2c sparse matrices, community structures, summaries, and save/load bundles. | Graph index persistence, VDB settings, PPR matrices, community support. |
| `indexes/indexing/graph_indexing/community.py` | GraphRAG community clustering, schema construction, community report generation, context packing, CSV/token truncation helpers. | Community clustering config, report prompts, community context budgets. |
| `indexes/indexing/graph_indexing/__init__.py` | Exports graph extraction schema, graph builder, universal graph, and graph indexer. | Graph index package API. |
| `indexes/loading/chunk_loading.py` | JSON/JSONL/TXT corpus loading, dataset graph-support extraction, record normalization, document ID generation. | Corpus format support, PERQA graph-support extraction, corpus record conversion. |
| `indexes/loading/graph_loading.py` | Loads serialized `GraphIndexer` and derives graph index path from method and dataset. | Graph index loading and path conventions. |
| `indexes/loading/__init__.py` | Re-exports corpus and graph loaders. | Loading package API. |

### `models/` — Low-Level Model Implementations

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `models/__init__.py` | Exports model classes and `get_model`. | Public model API. |
| `models/models_base.py` | Abstract contracts for generators, encoders, cross-encoders, and seq2seq models. | Adding model capability types. |
| `models/BM25_model.py` | Standalone BM25 model wrapper over `bm25s` with document initialization, scoring, top-k retrieval, and cloning. | BM25 model behavior outside indexers. |
| `models/hf_encoder.py` | `HFEncoder` using SentenceTransformers with instruction prefixes and cloning; `HFReranker` cross-encoder with rerank scoring. | Dense retrieval embeddings, query instruction behavior, reranker implementation. |
| `models/api_decoder.py` | `OpenAIGenerator` and `DeepSeekGenerator` using the OpenAI client, retries, generation parameters, and cloning. | API model configuration, retry/error behavior, request parameters. |
| `models/hf_decoder.py` | Hugging Face causal decoder wrapper with lazy loading, chat-template generation, dtype/device options, and cloning. | UCGR decoder model behavior or local HF generation. |
| `models/hf_encoder_decoder.py` | Hugging Face encoder-decoder wrapper, currently not in `_MODEL_REGISTRY` according to README. | Reserved seq2seq model support. |
| `models/llmlingua_model.py` | LLMLingua/LongLLMLingua compressor wrapper requiring a question and supporting rank methods. | LLMLingua compression behavior. |
| `models/llmlingua2_model.py` | LLMLingua-2 compressor wrapper. | LLMLingua-2 compression behavior. |
| `models/utils.py` | Model registry, global-model-config cache, and `get_model()` factory that merges `global.yaml` defaults with explicit kwargs. | Registering new models, model factory overrides, global model defaults. |

### `roles/` — Application Role Adapters

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `roles/role_base.py` | Abstract contracts for retriever, reranker, generator, rewriter, and refiner roles. | Adding new application roles or adapting pipelines to new contracts. |
| `roles/retriever.py` | Wraps BM25 or dense models as retrievers; detects backend by model methods; loads BM25/Dense indexers; implements `search` and `encode`. | Retriever index setup, top-k behavior, graph-retriever encoding. |
| `roles/reranker.py` | Wraps rerank models, converts documents to strings, maps scores back to `Document` objects. | Reranker adapter behavior. |
| `roles/reasoner.py` | Wraps generator models, holds prompt names/templates, supports system prompts, token/temperature overrides, answer prefixes, graph/FM paths. | Reasoner prompt routing and generation-parameter handling. |
| `roles/rewriter.py` | Implements HyDE rewriting through a nested reasoner and `hyde_*` prompts; `RewriterRoleAdapter` selects the method. | Query rewriting behavior. |
| `roles/refiner.py` | Wraps compression models and calls `compress`, passing rate/method/instruction and question when required. | Context compression adapter behavior. |
| `roles/utils.py` | Role factory `assist_role()`, model cloning, allowed-role validation, `get_method_role_config()`, model-default loading from `global.yaml`, and attribute configuration helper. | New roles, role-config extraction, model/role binding semantics. |
| `roles/__init__.py` | Re-exports contracts, adapters, and role utilities. | Role package API. |

### `rolerag/pipelines/` — Pipeline Framework and Concrete Pipelines

`rolerag/pipelines/__init__.py` imports all concrete pipeline modules so their `register_pipeline()` calls populate the registry, then exports `create_pipeline`, `register_pipeline`, and `BasePipeline`.

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `rolerag/pipelines/pipelines_base.py` | Pipeline registry, `create_pipeline()`, abstract `create()`/`_run_item()`, logging buffer, retrieval-only mode, single/batch `run()`, `predict()`, and `evaluate()` wiring to `RAGEvaluator`. | Pipeline lifecycle, registry behavior, evaluation integration, retrieval-only semantics. |
| `rolerag/pipelines/naive_pipeline.py` | BM25 retriever + HyDE rewriter + reasoner + `BasicRetrievalStrategy` + `NaiveGenerationStrategy`. | Naive RAG composition. |
| `rolerag/pipelines/hybrid_pipeline.py` | BM25 and dense retrievers + reasoner + `HybridRetrievalStrategy`. | Hybrid retrieval composition. |
| `rolerag/pipelines/rerank_pipeline.py` | BM25 retriever + HF reranker + reasoner + basic retrieval with reranking. | Rerank composition. |
| `rolerag/pipelines/llmlingua_pipeline.py` | BM25 retriever + LLMLingua refiner + reasoner + `RefineGenerationStrategy`. | LLMLingua pipeline composition. |
| `rolerag/pipelines/llmlingua2_pipeline.py` | BM25 retriever + LLMLingua-2 refiner + reasoner + refine generation. | LLMLingua-2 pipeline composition. |
| `rolerag/pipelines/perqa_pipeline.py` | BM25 retriever + planner/executor/answer reasoners + PERQA retrieval/generation strategies; compacts supports for logs. | PERQA orchestration and log metadata. |
| `rolerag/pipelines/hipporag_pipeline.py` | HFEncoder retriever + entity-extraction reasoner + answer reasoner + HippoRAG retrieval + naive generation. | HippoRAG pipeline composition. |
| `rolerag/pipelines/graphrag_local_pipeline.py` | HFEncoder retriever + reasoner + local GraphRAG retrieval and generation. | Local GraphRAG composition. |
| `rolerag/pipelines/graphrag_global_pipeline.py` | Reasoner + global GraphRAG retrieval (map/reduce community reports) and generation. | Global GraphRAG composition. |
| `rolerag/pipelines/ucgr_pipeline.py` | HFDecoder reasoner + UCGR constrained retrieval; builds effective query from optional context and returns first retrieved document as answer. | UCGR query construction and answer extraction. |

### `rolerag/pipelines/generation_strategy/` — Generation Strategies

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `generation_strategy/__init__.py` | Exports naive, refine, and PERQA generation strategies. GraphRAG strategies are imported directly by their pipelines. | Generation-strategy package API. |
| `generation_strategy/naive_generation_strategy.py` | Formats retrieval documents, loads a reasoner prompt, combines system/user prompts, calls the reasoner, and extracts answer prefixes. | Standard RAG answer generation. |
| `generation_strategy/refine_generation_strategy.py` | Formats retrieval contexts, invokes a refiner before reasoner generation, and extracts prefixed answers. | Context-compression generation flows. |
| `generation_strategy/perqa_generation_strategy.py` | Aggregates PERQA candidate graphs using bridge/comparison aggregate prompts, parses JSON, and selects consensus final graphs. | PERQA final aggregation. |
| `generation_strategy/graphrag_generation_strategy.py` | Local/global GraphRAG generation strategies; format prompts with `context_data` or `report_data` and call the reasoner. | GraphRAG answer prompt variables and response-type behavior. |

### `rolerag/pipelines/retrieval_strategy/` — Retrieval Strategies

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `retrieval_strategy/__init__.py` | Exports all retrieval strategies. | Retrieval-strategy package API. |
| `retrieval_strategy/basic_retrieval_strategy.py` | Simple retrieval, optional rewrite, optional rerank, top-k resolution from method inference config. | Basic, rerank, and LLMLingua retrieval flow. |
| `retrieval_strategy/hybrid_retrieval_strategy.py` | Runs multiple retrievers and fuses results with reciprocal rank fusion. | Hybrid fusion behavior. |
| `retrieval_strategy/hipporag_retrieval_strategy.py` | Loads graph index, extracts query entities with NER, links entities through entity VDB, runs personalized PageRank, propagates scores through matrices to chunks, falls back to dense chunks. | HippoRAG retrieval algorithm and PPR parameters. |
| `retrieval_strategy/graphrag_retrieval_strategy.py` | Loads/caches graph indexes; local retrieval uses entity VDB, communities, text units, and relations; global retrieval selects communities, runs map prompts, aggregates support points. | GraphRAG local/global retrieval and context packing. |
| `retrieval_strategy/perqa_retrieval_strategy.py` | PERQA question typing, bridge/comparison planning, step execution, retrieval per subquery, candidate/incomplete graph construction. | PERQA planning/execution logic. |
| `retrieval_strategy/ucgr_retrieval_strategy.py` | Loads SEAL FMIndex, builds constrained decoding processor, generates candidate titles with HFDecoder, maps generated text to passage IDs. | UCGR constrained retrieval and FMIndex integration. |

### `rolerag/pipelines/prompts/` — Prompt Templates and Manager

`PromptManager` auto-imports prompt modules and loads each module's `PROMPTS` dictionary. Prompt modules should only expose `PROMPTS`; duplicate template names are rejected. `PromptText.format()` leaves unknown placeholders unchanged, which matters for safe formatting across strategy-specific variables.

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `prompts/prompt_manager.py` | `PromptTemplate`, safe `PromptText` formatting, auto-discovery of prompt modules, duplicate detection, and `get_template()`. | Prompt lookup behavior, adding prompt modules, template duplicate rules. |
| `prompts/graphrag_prompt.py` | GraphRAG entity extraction, continue/if-loop extraction, description summarization, community report, local QA, global map, and global reduce QA templates. | GraphRAG extraction, community report, and local/global QA wording. |
| `prompts/hipporag_prompt.py` | HippoRAG NER, OpenIE/triple extraction, and QA templates. | HippoRAG extraction and answer prompts. |
| `prompts/hyde_prompt.py` | HyDE templates for web search, scifact, arguana, TREC-COVID, FiQA, DBpedia, TREC News, and multilingual Tydi tasks. | Query-rewrite prompt selection. |
| `prompts/naive_prompt.py` | Naive RAG answer, reference-only, and no-reference templates. | Standard generation prompt wording. |
| `prompts/perqa_prompt.py` | PERQA question-type, bridge/comparison planning, step QA, bridge aggregate, and comparison aggregate prompts. | PERQA planning, execution, and aggregation wording. |
| `prompts/ucgr_prompt.py` | UCGR title/section retrieval prompt. | UCGR constrained-generation prompt. |
| `prompts/__init__.py` | Exports `PromptTemplate` and `PromptManager`. | Prompt package API. |

### `rolerag/evaluator/` — Evaluation Pipeline

| File | Responsibilities | Consult for changes involving |
|---|---|---|
| `evaluator/__init__.py` | Re-exports `RAGEvaluator`, `EvalItem`, metrics, and `load_eval_samples`. | Evaluation package public API. |
| `evaluator/datasets/__init__.py` | Loads JSONL evaluation samples, normalizes `golden_answers`, and applies sample count/random sampling. | Evaluation dataset format and sampling. |
| `evaluator/evaluators/rag_evaluator.py` | `RAGEvaluator` initializes metrics, runs pipeline predictions with begin/end logging, computes metrics, saves `metric_score.json` and `intermediate_logs.json`, prints generation/retrieval summaries. | Evaluation flow, output paths, metric aggregation, log persistence. |
| `evaluator/evaluators/__init__.py` | Exports `RAGEvaluator`. | Evaluator import surface. |
| `evaluator/metrics/__init__.py` | Defines `MetricCategory`, `BaseMetric`, normalization helpers, generation metrics (`em`, `f1`, `recall`, `precision`, `acc`), PERQA metrics (`plan_score`, `step_score`, `pse_g`, graph similarity/GED variants), retrieval metrics (`acc4r`, recall/precision/f1/map/mrr/ndcg at k), and `get_metric()` with dynamic top-k parsing. | Adding metrics, changing answer normalization, PERQA scoring, retrieval relevance rules, or metric registration. |

### Key Call Relationships

- **CLI and configuration:** `main.py` parses arguments, creates `Config(config_dir, pipeline_name=...)`, applies explicit CLI values into `config.final_config`, and then dispatches to `build_graph_index()`, `build_index()`, `run_pipeline()`, or `run_evaluation()`. Pipeline execution should be traced through `main.py` → `Config` → `create_pipeline()` → `BasePipeline.run()`/`evaluate()`.
- **Index building:** `main.py` chooses graph vs non-graph based on `index.graph_building.graph_index_method`. Non-graph builds call `load_corpus_records()` → `TokenSizeChunking` → `BM25Indexing`/`DenseIndexing`/`FMIndexing`; graph builds call the extraction registry (`hipporag` or `graphrag`) → `GraphBuilder` → `GraphIndexer` → persisted graph index.
- **Pipeline construction:** Each concrete pipeline reads `index.paths`, calls `get_method_role_config()` for role parameters, injects index or graph paths into role configs, calls `get_model()` for raw models, and then `assist_role()` to clone and bind adapters. Pipelines then choose retrieval and generation strategies.
- **Model vs role boundary:** `get_model()` reads `config/global.yaml` model defaults and constructs a raw model. `assist_role()` does not read global model defaults; it clones the raw model and builds a role adapter from method-YAML `roles` settings. This is the central extension contract for new models, roles, and pipelines.
- **Prompts:** Generation strategies and extraction/community classes call `PromptManager.get_template()` by name. Prompt modules supply `PROMPTS`; role adapters may preload named templates for PERQA. Changes to prompt names must be coordinated with method YAML and strategy code.
- **Retrieval:** `NaiveRAGPipeline`, `RerankPipeline`, `LLMLinguaPipeline`, and `LLMLingua2Pipeline` use `BasicRetrievalStrategy`; `HybridPipeline` uses reciprocal-rank fusion; `HipporagPipeline` uses PPR over graph matrices; `GraphRAGLocalPipeline`/`GraphRAGGlobalPipeline` use graph-index-backed local/global retrieval; `PERQAPipeline` uses planning/execution retrieval; `UCGRPipeline` uses constrained FMIndex generation.
- **Evaluation:** `BasePipeline.evaluate()` reads evaluation options and constructs `RAGEvaluator`. `RAGEvaluator.run()` loads samples through `load_eval_samples()`, calls `pipeline.predict()` for each `EvalItem`, collects `EvalRecord`s, calculates each configured metric through `BaseMetric.calculate_metric()`, and writes metric/log outputs under `global.evaluation.save_dir`.
- **Web/API components:** There is no collected web/API server layer. The API-related code is `models/api_decoder.py`, which wraps OpenAI-compatible external generation endpoints and is used through `get_model()` and `assist_role()` like any other model. Changes to API keys, base URLs, retry behavior, or request parameters should be coordinated with `config/global.yaml` and reasoner/rewriter role configs.

## Change Guide for External LLMs

After receiving a user request, an external LLM must not output code immediately. It should first classify the requested change, identify which files must change and why, and determine whether any required source files are still missing.

### Adding a New Model

The following files usually require attention:

- `models/`: Add the low-level model class. A generation model implements `generate()`, an embedding model implements `encode()`, and a reranking model implements `rerank()`.
- `models/utils.py`: Import the new model class and add it to `_MODEL_REGISTRY`.
- `models/__init__.py`: Add the import and update `__all__` if the model must be publicly exported.
- `config/global.yaml`: Add defaults under `global.models` using the same registered model name.
- `config/methods/<pipeline>.yaml`: If the model will serve a role, add the corresponding configuration under `roles`.
- First confirm that the model satisfies the target adapter's capability checks: a retriever requires `get_top_k()` or `encode()`, a reasoner or rewriter requires `generate()`, a reranker requires `rerank()`, and a refiner requires `compress()`. Use the corresponding adapter source as the authority for constructor arguments and invocation behavior.

### Adding a New Role

The following files usually require attention:

- `roles/role_base.py`: Add a role contract that defines the stable interface on which pipelines can depend.
- `roles/<new_role>.py`: Add an adapter that wraps a low-level model in the role interface.
- `roles/utils.py`: Add the role to `_ALLOWED_ROLES` and add a branch to `assist_role()`.
- `roles/__init__.py`: Add the import and update `__all__` if the role must be publicly exported.
- `config/methods/<pipeline>.yaml`: Add the role's configuration block.

### Adding a New Pipeline

The following files usually require attention:

- `rolerag/pipelines/<new_pipeline>_pipeline.py`: Add a pipeline that inherits from `BasePipeline` and implements `create()` and `_run_item()`.
- `rolerag/pipelines/pipelines_base.py`: Understand the interfaces of `BasePipeline`, `register_pipeline()`, and `create_pipeline()`; do not casually change the shared base class.
- `rolerag/pipelines/__init__.py`: Import the new pipeline module so that its registration code runs.
- `config/methods/<new_pipeline>.yaml`: Add method-level configuration, including `index.paths`, `inference`, `roles`, and `evaluation`.
- `config/global.yaml`: Change the top-level `pipeline_type` if the new pipeline should become the default.
- `README.md` / `COMMANDS.md`: Add usage commands if the pipeline should be exposed to users.

Pipeline registration must follow the actual signature of `register_pipeline()` in `pipelines_base.py`. When it is an ordinary two-argument registration function, call `register_pipeline("<name>", PipelineClass)` directly after the class definition. Do not use it as a decorator, and do not pass the class name as a string in the second argument.

If the user explicitly requests support for `@register_pipeline("<name>")`, first change `register_pipeline()` into a backward-compatible decorator factory: preserve direct registration when `pipeline_class` is provided; when only a name is supplied, return a decorator that accepts the class, registers it, and returns it unchanged. After the change, verify both existing direct calls and new decorator calls. Never write `@register_pipeline("<name>", "PipelineClass")`, because a string is not a pipeline class and the return value of an ordinary registration function cannot be used as a decorator.

### Configuring Role Parameters and Adding a Method YAML File

RoleRAG's runtime role parameters are primarily defined in `config/methods/<pipeline>.yaml`. Before changing configuration, an external LLM must determine whether it is modifying an existing pipeline or adding a new one:

- To change the model, prompt, or role parameters of an existing pipeline, edit the existing `config/methods/<pipeline>.yaml`.
- To add a pipeline, create `config/methods/<new_pipeline>.yaml`.
- `--pipeline <name>` loads `config/methods/<name>.yaml`.
- `<name>` must match the name registered by `register_pipeline("<name>", PipelineClass)`.
- A method YAML file typically contains `index`, `inference`, `roles`, and `evaluation`.
- `index.paths` is generally used to load existing index paths at runtime; index construction parameters primarily come from `config/index.yaml`.
- `inference` stores pipeline-specific inference parameters, such as top-k values, fusion parameters, graph retrieval parameters, and generation-stage parameters.
- The exact `roles` hierarchy is `roles.<role>.<registered_model_name>.<optional_instance_name>`. The registered model name must be a key in `_MODEL_REGISTRY` in `models/utils.py` and must have defaults in `global.models`; it is not the `model_name` field.
- Single-instance example:

  ```yaml
  roles:
    retriever:
      BM25Model:
        top_k: 5
    reasoner:
      OpenAIGenerator:
        prompt_name: "naive_rag"
        max_tokens: 256
  ```

- Multi-instance example: when one model serves several purposes within the same role, add an instance-name level. The pipeline must call `get_method_role_config()` with the matching instance name.

  ```yaml
  roles:
    reasoner:
      OpenAIGenerator:
        planner_reasoner:
          prompt_names:
            type_system: "perqa_type_system"
        executor_reasoner:
          prompt_name: "perqa_rag_qa"
  ```

- `get_model()` creates the underlying model from `global.models[model_name]`; `assist_role()` clones that model and uses only role parameters from the method YAML file to configure the adapter.
- Consult the corresponding adapter source to determine exactly which fields a role configuration supports, for example `roles/reasoner.py`, `roles/retriever.py`, `roles/reranker.py`, `roles/rewriter.py`, or `roles/refiner.py`. Do not infer fields from a responsibility summary.
- Adding YAML configuration alone does not activate it. The corresponding pipeline's `create()` method must read the role configuration and call `get_model()` and `assist_role()`, and `_run_item()` or a strategy must actually use the role.
- If a new role serves only an existing pipeline, edit that pipeline's method YAML file. If it belongs to a new pipeline, place it in the new method YAML file.

### Adding a Retrieval or Generation Strategy

The following files usually require attention:

- This project has no retrieval or generation strategy registry. A concrete pipeline imports a strategy directly and instantiates it in `__init__`; adding a strategy therefore requires changing the imports and instantiation code of every pipeline that uses it.
- `rolerag/pipelines/retrieval_strategy/`: Add or change a retrieval strategy. It should accept role objects supplied by the pipeline, return a `RetrieverResult`, and read pipeline-specific parameters from `config["inference"][<pipeline_name>]`.
- `rolerag/pipelines/generation_strategy/`: Add or change a generation strategy. It usually accepts `question`, `retrieval_result`, and `reasoner`, and returns an answer string. A complex method may define additional return values, but the pipeline must unpack them and construct `RAGOutput`.
- `rolerag/pipelines/prompts/`: If the strategy requires a prompt, add a `PROMPTS` template and ensure that `PromptManager` can load it.
- When adding a prompt file, imports such as `PromptTemplate` and `PromptManager` must exactly follow the real module paths demonstrated by the provided `prompt_manager.py`. Do not infer or abbreviate import paths from directory names.
- In each pipeline that uses the strategy, instantiate it in `__init__`, invoke it in `_run_item()`, and preserve the existing retrieval-only branch, logging behavior, and `RAGOutput` conventions.
- Update the corresponding strategy package's `__init__.py` if the new strategy must be imported through the package path. This export is not required when a pipeline imports the strategy directly from its module.

### Backward Compatibility and Fallback Rules

Unless the user explicitly requests otherwise, new or modified functionality must not:

- Switch to another strategy, role, model, or prompt when results are empty, configuration is missing, or an exception occurs.
- Return a substitute business result to conceal an error.
- Support aliases for legacy configuration fields or interfaces, or automatically migrate legacy configuration.
- Automatically select another existing implementation as a substitute.

A component may use its own explicit, stable parameter defaults, such as `max_tokens`, `temperature`, or `top_k`. Missing required configuration must raise a clear error. This rule does not require refactoring existing project behavior.

### Changing Configuration Behavior

The following files usually require attention:

- `config/config.py`: Configuration loading, merging, and dot-path access.
- `config/global.yaml`: Global defaults and default model parameters.
- `config/index.yaml`: Index construction defaults.
- `config/methods/<pipeline>.yaml`: Runtime configuration for one pipeline.
- `main.py`: CLI argument override logic.

### Requirements When Outputting Code

- Group output by file path.
- Explain why each file must change.
- If the full contents of a required file have not been provided, ask the user for it first; do not invent it.
- Prefer targeted replacements. Output a complete file only when it is short or when changes are widely dispersed.
- Provide at least one verification command after the change.

### Mandatory Self-Check Before Outputting Code

Before outputting any code, verify every item below:

- Do not output code, pseudocode, patches, or complete file contents until the user explicitly confirms in a subsequent turn.
- For every new YAML field, list its configuration path, the class or function that reads it, and where it takes effect.
- For every new prompt, list the template name, template file, the `PromptManager` lookup site, and where it is rendered before the model call.
- For any new module that depends on a template, adapter, low-level model, index, or external component, explicitly document the complete call chain from input to final invocation. Configuring a name or object does not mean it is executed. Before calling a dependency directly, confirm its method signature and parameter semantics.
- Do not call an instance method, function, or attribute whose existence has not been confirmed in the provided source.
- A new role's method name must be identical across its contract, adapter, strategy, and pipeline.
- Before adding a strategy, determine whether an existing strategy can be reused. If the user explicitly asks for a new strategy, explain why it must differ from existing strategies.
- Do not switch to another strategy, role, model, or prompt when results are empty, configuration is missing, or an exception occurs, and do not conceal errors with substitute business results.
- The list of files to add or modify must include every new module, prompt, and configuration file referenced by the code.
- Before output, verify the reference closure in this order: file -> import -> caller -> configuration field.
- For algorithms, data transformations, sorting, fusion, or scoring, confirm input structures, field semantics, and ordering conventions from the provided source. Explicitly list the invariants on which the algorithm depends and the actual fields it uses. Do not replace business-semantic fields with iteration positions, display fields, or guessed fields. Cover boundary inputs and result ordering in the verification plan.
- Verification commands may use only arguments defined by `argparse` in `main.py`; argument names and value forms must match exactly. Configuration without a CLI override must be changed through YAML or another provided configuration entry point. Do not invent CLI arguments.

### Implementation Constraint Checklist

The plan for every new or modified pipeline, role, retrieval strategy, or generation strategy must include the following checklists. If any checklist is missing, complete the plan before outputting code.

#### Required Configuration Checklist

| Configuration path | Required? | Reading class or function | Where it takes effect | Behavior when missing |
| --- | --- | --- | --- | --- |

When required configuration is missing, `Behavior when missing` must specify a clear error. Optional configuration may use explicit, stable component-level defaults, but the default value, its use site, and why it does not alter the business path must be documented. Do not use legacy field aliases, substitute configuration, silent skipping, or automatic fallback.

#### Prompt Rendering Responsibility Matrix

Every new or used prompt must be listed:

| Template name | Template file | Template variables | Rendering class or function | Final `generate()` call site |
| --- | --- | --- | --- | --- |

Runtime parameters in templates, including counts and top-k values, must be rendered from configuration variables rather than hard-coded. Do not assume that setting `prompt_name` alone activates or renders a template.

For a new module that calls another component, the responsibility matrix must also state whether the final call targets an adapter, a low-level implementation, or an external service. If the dependency is called directly, list the confirmed method signature, parameter-passing behavior, and return-value semantics.

#### No-Fallback and Compatibility Audit

Before outputting code, answer each question with `Yes` or `No`; every answer should be `No`:

- Does the implementation fall back to another strategy, role, model, or prompt, or return a substitute business result when results are empty, configuration is missing, or an exception occurs?
- Does it include aliases for legacy configuration fields or interfaces, or automatic compatibility with legacy configuration?
- Does it automatically switch prompts, models, roles, or strategies?
- Does it call any method, function, or attribute whose existence has not been confirmed in the provided source?


## Quick Reference for Common Changes

| Change | Files to inspect first |
| --- | --- |
| Add a CLI argument | `main.py` |
| Change configuration merging | `config/config.py` |
| Add a data field | `config/types.py` |
| Add a low-level model | `models/`, `models/utils.py`, `config/global.yaml`, and the target role adapter / pipeline |
| Add a role | `roles/role_base.py`, `roles/utils.py`, `roles/<role>.py` |
| Add a pipeline | `rolerag/pipelines/pipelines_base.py`, `rolerag/pipelines/__init__.py`, `rolerag/pipelines/<pipeline>_pipeline.py`, `config/methods/<pipeline>.yaml` |
| Change retrieval logic | `rolerag/pipelines/retrieval_strategy/` and the corresponding pipeline |
| Change generation logic | `rolerag/pipelines/generation_strategy/`, `rolerag/pipelines/prompts/`, and the corresponding pipeline |
| Change index construction | `main.py`, `indexes/indexing/`, `indexes/loading/`, `config/index.yaml` |
| Change evaluation metrics | `rolerag/evaluator/metrics/` |


## Authoritative CLI Argument Table

The table below is derived automatically from the current `argparse` definitions and direct configuration assignments in `main.py`. It is the only set of CLI arguments that may be assumed to exist.

| CLI argument | Configuration path written directly |
| --- | --- |
| `--build_index` | Does not write configuration directly (runtime action or dedicated workflow) |
| `--chunk` | `index.chunking.enable_chunking` |
| `--chunk_overlap` | `index.chunking.overlap_token_size` |
| `--chunk_size` | `index.chunking.chunk_size` |
| `--corpus` | `index.paths.corpus_path` |
| `--data_file` | `global.evaluation.data_file` |
| `--evaluate` | Does not write configuration directly (runtime action or dedicated workflow) |
| `--fm_forced_bos_token_id` | `index.fm.forced_bos_token_id` |
| `--fm_hf_model` | `index.fm.hf_model` |
| `--fm_index_name` | `index.fm.index_name` |
| `--fm_jobs` | `index.fm.jobs` |
| `--fm_keep_duplicates` | `index.fm.keep_duplicates` |
| `--fm_lowercase` | `index.fm.lowercase` |
| `--indexing_model` | `index.build_index.indexing_model` |
| `--metrics` | `global.evaluation.metrics` |
| `--pipeline` | `global.pipeline_type` |
| `--print_results` | `global.evaluation.print_results` |
| `--random_sample` | `global.evaluation.random_sample` |
| `--run_pipeline` | Does not write configuration directly (runtime action or dedicated workflow) |
| `--sample_num` | `global.evaluation.sample_num` |
| `--storage` | `index.paths.storage_dir` |
| `--verbose` | `global.evaluation.verbose` |

Do not assume that a configuration field absent from this table has a corresponding CLI argument, especially an `inference` field in `config/methods/<pipeline>.yaml`. Unless `main.py` explicitly defines an entry point, change it through YAML or another listed configuration entry point.

## Policy for Requesting Additional Files

To keep the context manageable, this document embeds the shared contract source, every role adapter including the refiner, one runnable simple flow (Naive), and one multi-instance role flow (PERQA). Concrete LLMLingua, graph-indexing, UCGR, HippoRAG, and GraphRAG pipeline or strategy implementations must still be supplied on demand.

An external LLM must follow these rules:

- If the current task requires the full contents of a file that is not in the "Key Source Appendix," explicitly request that file from the user first.
- Give exact paths when requesting files, for example `roles/reasoner.py`, `rolerag/pipelines/perqa_pipeline.py`, or `README.md`.
- Do not merely say "please provide the relevant files." List the required files and explain why each is needed.
- To understand project background only, you may request `README.md`. For code changes, prioritize the actual source files.
- Until the user supplies the required files, do not invent interfaces, function signatures, class names, configuration fields, registry formats, or complete code.
- Without the full target file, output only a project understanding, a change plan, and the list of additional files required. Do not output a patch.
- Do not infer callable methods, parameter names, YAML fields, or return structures solely from a file responsibility summary.
- Request only the smallest file set needed for the current decision. If an interface remains uncertain, then request the next specific file.

Examples of task-specific file requests:

- To add an OpenAI-compatible API generation model, request `models/api_decoder.py` to confirm the existing API model wrapper and parameter conventions.
- To add a Hugging Face embedding or reranking model, request `models/hf_encoder.py` to confirm the conventions for `encode()` and `rerank()`.
- To add a Hugging Face decoder model, request `models/hf_decoder.py` to confirm local generation model loading and generation parameters.
- To add a retrieval role or change retrieval role parameters, use the embedded `roles/retriever.py`; for graph retrieval, also request the closest graph retrieval strategy.
- To add a reranking role or change reranking parameters, use the embedded `roles/reranker.py`; also request the target pipeline and the closest reranking method YAML file.
- To add reasoner parameters or change a prompt call, use the embedded `roles/reasoner.py` and `prompt_manager.py`; if the template variables differ, request a similar prompt file.
- To add a rewriting role, use the embedded `roles/rewriter.py`; when changing an existing method, also request the pipeline and strategy that use it.
- To add or change a context-refinement role, use the embedded `roles/refiner.py`; for an LLMLingua or LLMLingua2 execution flow, also request the target pipeline and `rolerag/pipelines/generation_strategy/refine_generation_strategy.py` to confirm `refine()` inputs, question passing, and prompt-rendering order before generation.
- To add a role that reuses a generation model, read `roles/reasoner.py`, `rolerag/pipelines/prompts/prompt_manager.py`, and a similar prompt file in addition to the role adapter, target pipeline, and strategy. Confirm whether the adapter or strategy renders the prompt. Do not assume that `ReasonerRoleAdapter` automatically renders a template from `prompt_name`.
- To add a prompt, request `rolerag/pipelines/prompts/prompt_manager.py` and a similar prompt file to confirm the `PROMPTS` structure and template variables.
- To add a retrieval strategy, request the target pipeline, the closest retrieval strategy, and the target method YAML file. Confirm the `retrieve()` arguments, `RetrieverResult.metadata`, and `inference` configuration keys.
- To add a generation strategy, request the target pipeline, the closest generation strategy, a similar prompt file, and the target method YAML file. Confirm the `generate()` arguments, template variables, and return-value unpacking.
- To modify an existing pipeline, request the target `rolerag/pipelines/<pipeline>_pipeline.py` and every strategy file it imports to confirm `create()`, `_run_item()`, and role configuration lookup. Complete Naive and PERQA examples are already embedded.
- To add a standard RAG pipeline, the Naive implementation, `rolerag/pipelines/__init__.py`, `rolerag/pipelines/pipelines_base.py`, and `naive.yaml` are already embedded. If the index or role composition differs, request the closest pipeline and method YAML file.
- To follow a complex pipeline, request the corresponding pipeline and its retrieval and generation strategy files, such as the relevant PERQA, HippoRAG, GraphRAG, or UCGR files.
- To adjust an existing method YAML file, request the target `config/methods/<pipeline>.yaml` to confirm its current `index`, `inference`, `roles`, and `evaluation` structure.
- To add a method YAML file, request the closest `config/methods/*.yaml` and the target pipeline file to confirm the filename, registration name, role configuration, and pipeline lookup logic.


## Key Source Appendix

The files below contain shared contracts, every role adapter including the refiner, and runnable Naive and PERQA reference flows. For LLMLingua, graph indexing, UCGR, HippoRAG, GraphRAG, or another implementation not embedded here, request the complete target files first.

### File: `config/config.py`

```python
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    GLOBAL_FILE = "global.yaml"
    INDEX_FILE = "index.yaml"

    def __init__(self, config_dir_path: str, pipeline_name: str | None = None):
        self.final_config = self._load_all_configs(config_dir_path, pipeline_name=pipeline_name)
        self._setup_hf_endpoint()

    def _resolve_global_path(self, config_dir_path: str) -> Path:
        config_dir = Path(config_dir_path).expanduser().resolve()
        if not config_dir.is_dir():
            raise ValueError(f"Config directory not found: {config_dir}")
        return config_dir / self.GLOBAL_FILE

    def _resolve_index_path(self, config_dir_path: str) -> Path:
        config_dir = Path(config_dir_path).expanduser().resolve()
        if not config_dir.is_dir():
            raise ValueError(f"Config directory not found: {config_dir}")
        return config_dir / self.INDEX_FILE

    def _resolve_method_config_path(self, config_dir_path: str, pipeline_name: str) -> Path:
        config_dir = Path(config_dir_path).expanduser().resolve()
        return config_dir / "methods" / f"{pipeline_name}.yaml"

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise ValueError(f"Required config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f.read())
        if not isinstance(data, dict):
            raise TypeError(f"Config file must contain a YAML mapping: {path}")
        return data

    def _build_config_from_method_yaml(
        self,
        global_cfg: Dict[str, Any],
        index_cfg: Dict[str, Any],
        method_cfg: Dict[str, Any],
        selected_pipeline: str,
    ) -> Dict[str, Any]:
        
        evaluation_defaults = global_cfg["evaluation"]
        method_evaluation = method_cfg["evaluation"]

        global_block = dict(global_cfg.get("global", {}) or {})
        global_block.update({
            "pipeline_type": selected_pipeline,
            "evaluation": deep_merge_dicts(evaluation_defaults, method_evaluation),
        })

        merged_index = deep_merge_dicts(index_cfg, method_cfg["index"])

        return {
            "global": global_block,
            "index": merged_index,
            "inference": method_cfg["inference"],
            "roles": method_cfg.get("roles", {}),
        }

    def _load_all_configs(self, config_dir_path: str, pipeline_name: str | None = None) -> Dict[str, Any]:
        global_path = self._resolve_global_path(config_dir_path)
        index_path = self._resolve_index_path(config_dir_path)
        global_cfg = self._load_yaml_file(global_path)
        index_cfg = self._load_yaml_file(index_path)
        selected_pipeline = pipeline_name or global_cfg["pipeline_type"]

        return self._build_config_from_method_yaml(
            global_cfg=global_cfg,
            index_cfg=index_cfg,
            method_cfg=self._load_yaml_file(self._resolve_method_config_path(config_dir_path, selected_pipeline)),
            selected_pipeline=selected_pipeline,
        )

    def get(self, key: str) -> Any:
        if not key:
            return self.final_config

        value: Any = self.final_config
        traversed = []
        for part in key.split("."):
            if not isinstance(value, dict):
                path = ".".join(traversed) or "<root>"
                raise TypeError(f"Config path is not a dict: {path}")
            if part not in value:
                missing_path = ".".join(traversed + [part])
                raise KeyError(f"Config key not found: {missing_path}")
            value = value[part]
            traversed.append(part)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        if "." not in key:
            self.final_config[key] = value
            return

        keys = key.split(".")
        current = self.final_config
        traversed = []
        for part in keys[:-1]:
            if part not in current:
                missing_path = ".".join(traversed + [part])
                raise KeyError(f"Config key not found: {missing_path}")
            if not isinstance(current[part], dict):
                path = ".".join(traversed + [part])
                raise TypeError(f"Config path is not a dict: {path}")
            current = current[part]
            traversed.append(part)
        current[keys[-1]] = value

    def _setup_hf_endpoint(self) -> None:
        hf_mirror = self.get("global.hf_mirror")
        if hf_mirror:
            os.environ["HF_ENDPOINT"] = hf_mirror

```

### File: `config/global.yaml`

```yaml
global:
  hf_mirror: "YOUR_MIRROR_URL"
  seed: 2024
  models:
    BM25Model:
      k1: 1.5
      b: 0.75
    HFEncoder:
      model_name: "BAAI/bge-base-en-v1.5"
    OpenAIGenerator:
      model_name: "gpt-4o-mini-2024-07-18"
      api_key: ""
      base_url: "YOUR_BASE_URL_FOR_OPENAI"
    DeepSeekGenerator:
      model_name: "deepseek-v4-flash"
      api_key: ""
      base_url: "YOUR_BASE_URL_FOR_DEEPSEEK"
    HFDecoder:
      model_name: "UCGR"
      model_path: "modelList/UCGR"
      device_map: "cpu"
      use_fp16: false
      torch_dtype: "float32"
    HFReranker:
      model_name: "BAAI/bge-reranker-base"
    LLMLingua2Model:
      model_name: "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
      device_map: "cpu"
      use_llmlingua2: true
    LLMLinguaModel:
      model_name: "Qwen/Qwen2.5-0.5B-Instruct"
      # model_name: "NousResearch/Llama-2-7b-hf"
      device_map: "cpu"

pipeline_type: "naive"

evaluation:
  show_progress: true
  save_dir: "results"
  save_metric: true
  verbose: true
  print_results: true

```

### File: `config/index.yaml`

```yaml
paths:
  storage_dir: "indexes/storage"
  corpus_path: "data/hotpot_hippo.jsonl"


chunking:
  enable_chunking: true
  chunk_size: 1200
  overlap_token_size: 100


build_index:
  indexing_model: "HFEncoder"


dense:
  faiss_type: "FLAT"
  faiss_nlist: 1024
  faiss_nprobe: 10
  faiss_hnsw_m: 32
  faiss_ef_search: 64


bm25:
  k1: 1.5
  b: 0.75


fm:
  hf_model: "modelList/UCGR"
  jobs: 8
  forced_bos_token_id: 128000
  lowercase: false
  keep_duplicates: false
  index_name:


graph_building:
  graph_index_method: "hipporag"
  force_rebuild_graph: true

  extraction:
    graph_extraction_model: "OpenAIGenerator"
    graph_extraction_concurrency: 4
    graph_extraction_retries: 1

  methods:
    hipporag:
      index:
        use_entities_vdb: true
        use_relations_vdb: false
        use_e2r_r2c: true
        enable_graph_augmentation: true
        augment_top_k: 10
        augment_threshold: 0.8
        similarity_max: 1.0

    graphrag:
      extraction:
        graph_max_gleaning: 1

      index:
        use_entities_vdb: true
        use_relations_vdb: false
        use_e2r_r2c: false
        enable_graph_augmentation: false
        use_community: true
        entity_vdb_text_fields:
          - "entity_name"
          - "entity_type"
          - "description"
        graph_cluster_algorithm: "leiden"
        max_graph_cluster_size: 10
        graph_cluster_seed: 3735928559
        summary_max_tokens: 500
        enforce_sub_communities: false
        community_describe_max_tokens: 12000

```

### File: `config/methods/naive.yaml`

```yaml
index:
  paths:
    bm25_index_path: "indexes/storage/BM25Model_hotpot_hippo_c1200"

inference:
  naive:
    return_top_k: 5

roles:
  retriever:
    BM25Model:
      top_k: 5
  rewriter:
    OpenAIGenerator:
      method: "hyde"
      num_hypotheses: 4
      max_tokens: 512
      temperature: 0.7
      task: "web_search"
      language: "en"
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

### File: `config/types.py`

```python
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

```

### File: `main.py`

```python
#!/usr/bin/env python

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

for noisy_logger in ["httpx", "httpcore", "openai", "urllib3", "faiss", "faiss.loader", "sentence_transformers"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR.parent))
os.chdir(BASE_DIR)

_default_config_dir = BASE_DIR / "config"

from RoleRAG.config.config import Config
from RoleRAG.indexes import make_index_name

_EXTRACTION_REGISTRY = {}

def _register_extraction(method_name: str, extraction_cls):
    
    _EXTRACTION_REGISTRY[method_name] = extraction_cls

def _get_extraction(method_name: str):
    
    if method_name not in _EXTRACTION_REGISTRY:
        raise ValueError(f"Unknown graph method: {method_name}. Available: {list(_EXTRACTION_REGISTRY.keys())}")
    return _EXTRACTION_REGISTRY[method_name]


def _init_extraction_registry():
    
    if _EXTRACTION_REGISTRY:
        return
    from RoleRAG.indexes.indexing.graph_indexing.extraction import GraphRAGExtraction, HippoRAGExtraction
    _register_extraction("hipporag", HippoRAGExtraction)
    _register_extraction("graphrag", GraphRAGExtraction)


def _require_value(value, message: str):
    
    if value in (None, ""):
        raise ValueError(message)
    return value


def _require_config(config, key: str):
    
    return _require_value(config.get(key), f"{key} is required.")


def _get_storage_dir(config):
    
    return _require_config(config, "index.paths.storage_dir")


def _get_chunking_config(config):
    
    return {
        "chunk_size": _require_config(config, "index.chunking.chunk_size"),
        "overlap_token_size": _require_config(config, "index.chunking.overlap_token_size"),
        "enable_chunking": _require_config(config, "index.chunking.enable_chunking"),
    }


def _apply_build_target(config, build_target: str) -> None:
    
    build_target = build_target.lower()
    if build_target == "bm25":
        config["index.build_index.indexing_model"] = "BM25Model"
        config["index.graph_building.graph_index_method"] = None
    elif build_target == "dense":
        config["index.graph_building.graph_index_method"] = None
    elif build_target == "fmindex":
        config["index.build_index.indexing_model"] = "FMIndex"
        config["index.graph_building.graph_index_method"] = None
    elif build_target in {"hipporag", "graphrag"}:
        config["index.graph_building.graph_index_method"] = build_target
    else:
        raise ValueError(
            f"Unknown build target: {build_target}. "
            "Available: bm25, dense, fmindex, hipporag, graphrag"
        )


async def build_graph_index(config):
    
    _init_extraction_registry()
    from RoleRAG.models import get_model
    from RoleRAG.roles.utils import _configure_model, get_model_defaults
    from RoleRAG.indexes import load_corpus_records, TokenSizeChunking
    from RoleRAG.indexes.indexing.graph_indexing import GraphBuilder, GraphIndexer
    from RoleRAG.indexes.loading.graph_loading import get_graph_index_path
    from RoleRAG.config.types import Document

    graph_build_cfg = _require_config(config, "index.graph_building")
    graph_extract_cfg = _require_config(config, "index.graph_building.extraction")
    method = _require_config(config, "index.graph_building.graph_index_method")
    index_config = dict(_require_config(config, f"index.graph_building.methods.{method}.index"))
    corpus_path = _require_config(config, "index.paths.corpus_path")
    storage_dir = _get_storage_dir(config)
    chunking_cfg = _get_chunking_config(config)
    chunk_size = chunking_cfg["chunk_size"]
    overlap_token_size = chunking_cfg["overlap_token_size"]
    enable_chunking = chunking_cfg["enable_chunking"]
    force_rebuild = _require_value(
        graph_build_cfg.get("force_rebuild_graph"),
        "index.graph_building.force_rebuild_graph is required.",
    )

    dataset_name = Path(corpus_path).stem 

    extraction_cls = _get_extraction(method)

    save_path = get_graph_index_path(method, dataset_name, storage_dir)
    if force_rebuild and os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    print("=" * 60)
    print("Building Graph Index")
    print("=" * 60)
    print(f"Method: {method}")
    print(f"Corpus: {corpus_path} -> dataset: {dataset_name}")
    print(f"Save to: {save_path}")
    print(f"Encoder model: {_require_config(config, 'index.build_index.indexing_model')}")
    print(f"Chunking: {enable_chunking}, token_size={chunk_size}, overlap={overlap_token_size}")
    print("=" * 60)

    print("\n[1/5] Loading corpus...")
    corpus_records = load_corpus_records(corpus_path)
    if not corpus_records:
        raise ValueError(f"No corpus records loaded from {corpus_path}. Check the corpus file and loader format.")
    documents = [Document.from_dict(record) for record in corpus_records]
    print(f"Loaded {len(documents)} documents")

    if enable_chunking:
        print(f"\n[2/5] Chunking documents (token_size={chunk_size}, overlap={overlap_token_size})...")
        chunker = TokenSizeChunking(chunk_size=chunk_size, overlap_token_size=overlap_token_size)
        chunks = chunker.chunk(documents)
    else:
        chunks = documents
    print(f"Total chunks: {len(chunks)}")

    print(f"\n[3/5] Extracting entities and relations (method: {method})...")
    llm_model_name = _require_value(
        graph_extract_cfg.get("graph_extraction_model"),
        "index.graph_building.extraction.graph_extraction_model is required.",
    )
    llm_model = get_model(llm_model_name)
    llm_build_config = dict(get_model_defaults(llm_model_name))
    llm_build_config.update({"max_tokens": 4096, "temperature": 0.0})
    _configure_model(llm_model, llm_build_config)

    import inspect
    extraction_signature = inspect.signature(extraction_cls.__init__)
    extraction_kwargs = {
        "ask_fn": lambda prompt: llm_model.generate(prompt),
        "max_concurrency": _require_value(
            graph_extract_cfg.get("graph_extraction_concurrency"),
            "index.graph_building.extraction.graph_extraction_concurrency is required.",
        ),
        "max_retries": _require_value(
            graph_extract_cfg.get("graph_extraction_retries"),
            "index.graph_building.extraction.graph_extraction_retries is required.",
        ),
        "fail_on_empty": True,
        "max_gleaning": (
            _require_config(
                config,
                f"index.graph_building.methods.{method}.extraction.graph_max_gleaning",
            )
            if "max_gleaning" in extraction_signature.parameters
            else None
        ), 
    }
    extraction = extraction_cls(
        **{
            key: value
            for key, value in extraction_kwargs.items()
            if key in extraction_signature.parameters and value is not None
        }
    )
    extraction_result = await extraction.extract(chunks)
    print(f"Entities: {len(extraction_result.nodes)}, Relations: {len(extraction_result.edges)}")

    print("\n[4/5] Building graph...")
    builder = GraphBuilder()
    graph = builder.build(extraction_result)
    print(f"Graph nodes: {graph.node_num}, edges: {graph.edge_num}")

    print("\n[5/5] Building indices...")
    encoder_model_name = _require_config(config, "index.build_index.indexing_model")
    encoder_model = get_model(encoder_model_name)
    encoder_build_config = dict(get_model_defaults(encoder_model_name))
    _configure_model(encoder_model, encoder_build_config)

    indexer = GraphIndexer(config=index_config, encoder=encoder_model)
    indexer.build_indices(
        graph,
        chunks,
        llm_call=lambda prompt: llm_model.generate(prompt),
    )
    indexer.save(save_path)
    print(f"Graph index saved to {save_path}")
    print(f"Tip: Set index.paths.graph_index_path: \"{save_path}\" in method config to use it directly.")

    print("\nGraph index building completed!")


def build_index(config):
    corpus_path = _require_value(
        config.get("index.paths.corpus_path"),
        "index.paths.corpus_path is required.",
    )
    storage_dir = _get_storage_dir(config)
    indexing_model = _require_config(config, "index.build_index.indexing_model")
    chunking_cfg = _get_chunking_config(config)
    chunk_size = chunking_cfg["chunk_size"]
    overlap_token_size = chunking_cfg["overlap_token_size"] 
    enable_chunking = chunking_cfg["enable_chunking"]

    if indexing_model == "FMIndex":
        from RoleRAG.indexes.indexing.chunk_indexing import FMIndexing
        from RoleRAG.indexes.indexing.chunk_indexing.fm_indexing import make_fm_index_path

        hf_model = _require_config(config, "index.fm.hf_model") 
        jobs = _require_config(config, "index.fm.jobs")  
        forced_bos_token_id = _require_config(config, "index.fm.forced_bos_token_id")
        lowercase = _require_config(config, "index.fm.lowercase")
        keep_duplicates = _require_config(config, "index.fm.keep_duplicates")
        index_name = config.get("index.fm.index_name")
        index_path = make_fm_index_path(storage_dir, corpus_path, index_name=index_name)

        print("Building FMIndex for UCGR constrained generation...")
        print(f"Corpus: {corpus_path}")
        print(f"HF model/tokenizer: {hf_model}")
        print(f"Forced BOS token id: {forced_bos_token_id}")
        print(f"Jobs: {jobs}")
        print(f"Lowercase: {lowercase}")
        print(f"Keep duplicates: {keep_duplicates}")
        print(f"Saving FMIndex to {index_path}")

        fm_indexer = FMIndexing(
            hf_model=hf_model,
            forced_bos_token_id=forced_bos_token_id,
            jobs=jobs,
            lowercase=lowercase,
            keep_duplicates=keep_duplicates,
        )
        fm_indexer.build_index(corpus_path=corpus_path)
        fm_indexer.save(index_path)
        print(f"FMIndex built successfully with {len(fm_indexer.labels)} labels!")
        print(f"Use this trie path for UCGR-style inference: {index_path}")
        return

    from RoleRAG.indexes import load_corpus_records, TokenSizeChunking
    from RoleRAG.config.types import Document

    print(f"Loading corpus from {corpus_path}...")
    corpus_records = load_corpus_records(corpus_path)
    if not corpus_records:
        raise ValueError(f"No corpus records loaded from {corpus_path}. Check the corpus file and loader format.")
    documents = [Document.from_dict(record) for record in corpus_records]
    print(f"Loaded {len(documents)} documents")

    if enable_chunking:
        chunker = TokenSizeChunking(chunk_size=chunk_size, overlap_token_size=overlap_token_size)
        print(f"Chunking enabled, token_size={chunk_size}, overlap={overlap_token_size}")
        documents = chunker.chunk(documents)
        print(f"After chunking: {len(documents)} documents")
    else:
        chunk_size = None

    index_name = make_index_name(
        indexing_model,
        corpus_path,
        chunk_size,
        faiss_type=_require_config(config, "index.dense.faiss_type"), 
    )
    index_path = os.path.join(storage_dir, index_name)

    if indexing_model == "BM25Model":
        from RoleRAG.indexes.indexing.chunk_indexing import BM25Indexing

        print(f"Building BM25 index...")
        k1 = _require_config(config, "index.bm25.k1") 
        b = _require_config(config, "index.bm25.b") 
        bm25_indexer = BM25Indexing(k1=k1, b=b)
        bm25_indexer.documents = documents
        bm25_indexer.build_index(documents)
        print(f"Saving BM25 index to {index_path}...")
        bm25_indexer.save(index_path)
        print("BM25 index built successfully!")
    else:
        from RoleRAG.models import get_model
        from RoleRAG.roles.utils import _configure_model, get_model_defaults
        from RoleRAG.indexes.indexing.chunk_indexing import DenseIndexing

        print(f"Building Dense index with {indexing_model}...")
        model_config = get_model_defaults(indexing_model)
        dense_faiss_type = _require_config(config, "index.dense.faiss_type")
        dense_faiss_nlist = _require_config(config, "index.dense.faiss_nlist")
        dense_faiss_nprobe = _require_config(config, "index.dense.faiss_nprobe")
        dense_faiss_hnsw_m = _require_config(config, "index.dense.faiss_hnsw_m")
        dense_faiss_ef_search = _require_config(config, "index.dense.faiss_ef_search")
        encoder_model = get_model(indexing_model)
        _configure_model(encoder_model, model_config)

        dense_indexer = DenseIndexing(
            encoder=encoder_model,
            batch_size=getattr(encoder_model, "batch_size"),
            faiss_type=dense_faiss_type, 
            faiss_nlist=dense_faiss_nlist,  
            faiss_nprobe=dense_faiss_nprobe,  
            faiss_hnsw_m=dense_faiss_hnsw_m, 
            faiss_ef_search=dense_faiss_ef_search, 
        )
        dense_indexer.documents = documents
        dense_indexer.build_index(documents)
        print(f"Saving Dense index to {index_path}...")
        dense_indexer.save(index_path)
        print("Dense index built successfully!")


def _is_graph_build(config) -> bool:
    return bool(
        config.get("index.graph_building")
        and config.get("index.graph_building.graph_index_method")
    )


def run_pipeline(config, query):
    from RoleRAG.rolerag.pipelines import create_pipeline

    pipeline = create_pipeline(config.final_config)
    result = pipeline.run(query)

    print(f"\nQuery: {result.query}")
    print(f"Answer: {result.answer}")

    if result.retrieval_result and result.retrieval_result.documents:
        print(f"\nRetrieved {len(result.retrieval_result.documents)} documents:")
        for i, doc in enumerate(result.retrieval_result.documents):
            score = getattr(doc, 'score', None)
            if score is None and result.retrieval_result.scores and i < len(result.retrieval_result.scores):
                score = result.retrieval_result.scores[i]
            if score is not None and isinstance(score, (int, float)):
                print(f"  Doc {i+1} (score: {score:.4f}): {doc.content[:80]}...")
            else:
                print(f"  Doc {i+1}: {doc.content[:80]}...")

    return result


def run_evaluation(config):
    from RoleRAG.rolerag.evaluator.datasets import load_eval_samples
    from RoleRAG.rolerag.pipelines import create_pipeline

    data_file = _require_config(config, "global.evaluation.data_file")  
    sample_num = _require_config(config, "global.evaluation.sample_num")  
    random_sample = _require_config(config, "global.evaluation.random_sample")  
    pipeline_type = _require_config(config, "global.pipeline_type")
    index_paths = config.get("index.paths")
    inference_cfg = config.get("inference")

    print("\nLoading evaluation samples...")
    eval_samples = load_eval_samples(
        data_file=data_file,
        sample_num=sample_num,
        random_sample=random_sample,
    )
    print(f"Loaded {len(eval_samples)} samples")

    print("\nInitializing pipeline...")
    pipeline = create_pipeline(config.final_config)

    print("\nRunning evaluation...")
    results = pipeline.evaluate(eval_samples)

    print("\nDone!")
    return results


def main():
    parser = argparse.ArgumentParser(description="RoleRAG Framework CLI")

    parser.add_argument("--build_index", type=str, choices=["bm25", "dense", "fmindex", "hipporag", "graphrag"], help="Build the specified index type")
    parser.add_argument("--run_pipeline", type=str, help="Run one pipeline inference query with the current configuration")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the configured pipeline on the evaluation dataset")
    parser.add_argument("--corpus", type=str, help="Override index.paths.corpus_path, the corpus file used for index or graph construction")
    parser.add_argument("--data_file", type=str, help="Override global.evaluation.data_file, the evaluation dataset path")
    parser.add_argument("--indexing_model", type=str, help="Override index.build_index.indexing_model, such as BM25Model, HFEncoder, or FMIndex")
    parser.add_argument("--storage", type=str, help="Override index.paths.storage_dir, the root directory used to save built indices")
    parser.add_argument("--chunk", action="store_true", help="Force-enable index.chunking.enable_chunking for this run")
    parser.add_argument("--chunk_size", type=int, help="Override index.chunking.chunk_size, the max token count per chunk")
    parser.add_argument("--chunk_overlap", type=int, help="Override index.chunking.overlap_token_size, the token overlap between adjacent chunks")
    parser.add_argument("--pipeline", type=str, help="Override global.pipeline_type, such as naive, hipporag, or graphrag_local")
    parser.add_argument("--sample_num", type=int, help="Override global.evaluation.sample_num, the number of evaluation samples to load")
    parser.add_argument("--random_sample", action="store_true", help="Override global.evaluation.random_sample to sample evaluation data randomly")
    parser.add_argument("--metrics", type=str, nargs="+", help="Override global.evaluation.metrics with the metrics to compute during evaluation")
    parser.add_argument("--verbose", action="store_true", help="Override global.evaluation.verbose to print per-sample evaluation details")
    parser.add_argument("--print_results", action="store_true", help="Override global.evaluation.print_results to print the final evaluation summary table")
    parser.add_argument("--fm_hf_model", type=str, help="Override index.fm.hf_model, the tokenizer/model used to build FMIndex")
    parser.add_argument("--fm_jobs", type=int, help="Override index.fm.jobs, the number of workers used for FMIndex construction")
    parser.add_argument("--fm_forced_bos_token_id", type=int, help="Override index.fm.forced_bos_token_id, the constrained-generation BOS token id for FMIndex")
    parser.add_argument("--fm_index_name", type=str, help="Override index.fm.index_name, the optional FMIndex output file name under storage_dir")
    parser.add_argument("--fm_lowercase", action="store_true", help="Override index.fm.lowercase to lowercase FMIndex titles before indexing")
    parser.add_argument("--fm_keep_duplicates", action="store_true", help="Override index.fm.keep_duplicates to keep duplicate Target_psg ids in FMIndex")

    args = parser.parse_args()

    config = Config(str(_default_config_dir), pipeline_name=args.pipeline)

    if args.corpus:
        config["index.paths.corpus_path"] = args.corpus
    if args.data_file:
        config["global.evaluation.data_file"] = args.data_file
    if args.pipeline:
        config["global.pipeline_type"] = args.pipeline
    if args.build_index:
        _apply_build_target(config, args.build_index)
    if args.indexing_model:
        config["index.build_index.indexing_model"] = args.indexing_model
    if args.storage:
        config["index.paths.storage_dir"] = args.storage
    if args.chunk:
        config["index.chunking.enable_chunking"] = True
    if args.chunk_size:
        config["index.chunking.chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        config["index.chunking.overlap_token_size"] = args.chunk_overlap
    if args.sample_num is not None:
        config["global.evaluation.sample_num"] = args.sample_num
    if args.random_sample:
        config["global.evaluation.random_sample"] = True
    if args.metrics:
        config["global.evaluation.metrics"] = args.metrics
    if args.verbose:
        config["global.evaluation.verbose"] = True
    if args.print_results:
        config["global.evaluation.print_results"] = True
    if args.fm_hf_model:
        config["index.fm.hf_model"] = args.fm_hf_model
    if args.fm_jobs is not None:
        config["index.fm.jobs"] = args.fm_jobs
    if args.fm_forced_bos_token_id is not None:
        config["index.fm.forced_bos_token_id"] = args.fm_forced_bos_token_id
    if args.fm_index_name:
        config["index.fm.index_name"] = args.fm_index_name
    if args.fm_lowercase:
        config["index.fm.lowercase"] = True
    if args.fm_keep_duplicates:
        config["index.fm.keep_duplicates"] = True

    if args.build_index:
        corpus_path = _require_config(config, "index.paths.corpus_path")

        print("=" * 60)
        print("Building Index")
        print("=" * 60)
        print(f"Build target: {args.build_index}")
        print(f"Corpus: {corpus_path}")
        print(f"Storage: {_get_storage_dir(config)}")
        if _is_graph_build(config):
            print(f"Graph method: {_require_config(config, 'index.graph_building.graph_index_method')}")
        else:
            print(f"Indexing Model: {_require_config(config, 'index.build_index.indexing_model')}")
        print("=" * 60)

        if _is_graph_build(config):
            asyncio.run(build_graph_index(config))
        else:
            build_index(config)
            print("\nIndex building completed!")

    if args.run_pipeline:
        print("=" * 60)
        print("Running Pipeline")
        print("=" * 60)
        pipeline_type = _require_config(config, "global.pipeline_type")
        print(f"Pipeline: {pipeline_type}")
        print(f"Query: {args.run_pipeline}")
        if pipeline_type in {"hipporag", "graphrag_local", "graphrag_global"}:
            print(f"Graph index: {_require_config(config, 'index.paths.graph_index_path')}")
        print("=" * 60)

        run_pipeline(config, args.run_pipeline)

    if args.evaluate:
        print("=" * 60)
        print("RoleRAG Evaluation")
        print("=" * 60)
        print(f"Pipeline: {_require_config(config, 'global.pipeline_type')}")
        print(f"Data file: {_require_config(config, 'global.evaluation.data_file')}")
        if _require_config(config, "global.evaluation.sample_num"):
            print(
                f"Sample num: {_require_config(config, 'global.evaluation.sample_num')} "
                f"({'random' if _require_config(config, 'global.evaluation.random_sample') else 'first N'})"
            )
        print(
            f"Metrics: {', '.join(_require_config(config, 'global.evaluation.metrics'))}"
        )
        print(f"Verbose: {_require_config(config, 'global.evaluation.verbose')}")
        print(f"Print results: {_require_config(config, 'global.evaluation.print_results')}")
        print("=" * 60)

        run_evaluation(config)


if __name__ == "__main__":
    main()

```

### File: `models/__init__.py`

```python
from RoleRAG.models.models_base import BaseModel, BaseEncoder, BaseCrossEncoder, BaseSeq2Seq
from RoleRAG.models.BM25_model import BM25Model
from RoleRAG.models.api_decoder import OpenAIGenerator, DeepSeekGenerator
from RoleRAG.models.hf_decoder import HFDecoder
from RoleRAG.models.llmlingua2_model import LLMLingua2Model
from RoleRAG.models.llmlingua_model import LLMLinguaModel
from RoleRAG.models.hf_encoder import (
    HFEncoder,
    HFReranker,
)
from RoleRAG.models.hf_encoder_decoder import HFEncoderDecoder
from RoleRAG.models.utils import get_model

__all__ = [
    "BaseModel",
    "BaseEncoder",
    "BaseCrossEncoder",
    "BaseSeq2Seq",
    "BM25Model",
    "OpenAIGenerator",
    "DeepSeekGenerator",
    "HFDecoder",
    "LLMLingua2Model",
    "LLMLinguaModel",
    "HFEncoder",
    "HFEncoderDecoder",
    "HFReranker",
    "get_model",
]

```

### File: `models/models_base.py`

```python
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

```

### File: `models/utils.py`

```python
import os
from typing import Any, Dict, Optional

from RoleRAG.models.BM25_model import BM25Model
from RoleRAG.models.api_decoder import OpenAIGenerator, DeepSeekGenerator
from RoleRAG.models.hf_decoder import HFDecoder
from RoleRAG.models.llmlingua2_model import LLMLingua2Model
from RoleRAG.models.llmlingua_model import LLMLinguaModel
from RoleRAG.models.hf_encoder import (
    HFEncoder,
    HFReranker,
)


_MODEL_REGISTRY = {
    "BM25Model": BM25Model,
    "HFEncoder": HFEncoder,
    "OpenAIGenerator": OpenAIGenerator,
    "HFDecoder": HFDecoder,
    "HFReranker": HFReranker,
    "DeepSeekGenerator": DeepSeekGenerator,
    "LLMLingua2Model": LLMLingua2Model,
    "LLMLinguaModel": LLMLinguaModel,
}

_GLOBAL_MODEL_CONFIG_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_global_model_defaults() -> Dict[str, Dict[str, Any]]:
    global _GLOBAL_MODEL_CONFIG_CACHE
    if _GLOBAL_MODEL_CONFIG_CACHE is None:
        import yaml

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "global.yaml")
        if not os.path.exists(config_path):
            _GLOBAL_MODEL_CONFIG_CACHE = {}
        else:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            model_config = ((config.get("global", {}) or {}).get("models", {}) or {})
            if not isinstance(model_config, dict):
                raise TypeError("global.models must be a mapping")
            _GLOBAL_MODEL_CONFIG_CACHE = {
                name: values.copy()
                for name, values in model_config.items()
                if isinstance(values, dict)
            }
    return _GLOBAL_MODEL_CONFIG_CACHE


def get_model(model_type: str, **kwargs) -> Any:
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(_MODEL_REGISTRY.keys())}")

    model_config = _load_global_model_defaults().get(model_type, {}).copy()
    model_config.update(kwargs)
    return _MODEL_REGISTRY[model_type](**model_config)

```

### File: `rolerag/pipelines/__init__.py`

```python
from RoleRAG.rolerag.pipelines.pipelines_base import create_pipeline, register_pipeline, BasePipeline
from RoleRAG.rolerag.pipelines.naive_pipeline import NaiveRAGPipeline
from RoleRAG.rolerag.pipelines.llmlingua2_pipeline import LLMLingua2Pipeline
from RoleRAG.rolerag.pipelines.llmlingua_pipeline import LLMLinguaPipeline
from RoleRAG.rolerag.pipelines.rerank_pipeline import RerankPipeline
from RoleRAG.rolerag.pipelines.hybrid_pipeline import HybridPipeline
from RoleRAG.rolerag.pipelines.perqa_pipeline import PERQAPipeline
from RoleRAG.rolerag.pipelines.hipporag_pipeline import HipporagPipeline
from RoleRAG.rolerag.pipelines.ucgr_pipeline import UCGRPipeline
from RoleRAG.rolerag.pipelines.graphrag_local_pipeline import GraphRAGLocalPipeline
from RoleRAG.rolerag.pipelines.graphrag_global_pipeline import GraphRAGGlobalPipeline

__all__ = [
    "BasePipeline",
    "NaiveRAGPipeline",
    "LLMLingua2Pipeline",
    "LLMLinguaPipeline",
    "RerankPipeline",
    "HybridPipeline",
    "PERQAPipeline",
    "HipporagPipeline",
    "UCGRPipeline",
    "GraphRAGLocalPipeline",
    "GraphRAGGlobalPipeline",
    "create_pipeline",
    "register_pipeline",
]

```

### File: `rolerag/pipelines/generation_strategy/__init__.py`

```python
from RoleRAG.rolerag.pipelines.generation_strategy.naive_generation_strategy import NaiveGenerationStrategy, generate_answer
from RoleRAG.rolerag.pipelines.generation_strategy.refine_generation_strategy import RefineGenerationStrategy
from RoleRAG.rolerag.pipelines.generation_strategy.perqa_generation_strategy import PERQAGenerationStrategy, generate_perqa_answer

__all__ = [
    "NaiveGenerationStrategy",
    "RefineGenerationStrategy",
    "PERQAGenerationStrategy",
    "generate_answer",
    "generate_perqa_answer",
]

```

### File: `rolerag/pipelines/generation_strategy/naive_generation_strategy.py`

```python
import logging
import re
from typing import Any

from RoleRAG.config.types import RetrieverResult
from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)


class NaiveGenerationStrategy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @staticmethod
    def format_retrieval_result(retrieval_result) -> str:
        if retrieval_result is None:
            return ""
        if hasattr(retrieval_result, "documents"):
            docs = retrieval_result.documents
        elif isinstance(retrieval_result, list):
            docs = retrieval_result
        else:
            docs = [retrieval_result]

        formatted = []
        for idx, doc in enumerate(docs):
            if hasattr(doc, "content"):
                content = doc.content
            elif isinstance(doc, dict):
                content = doc.get("content", doc.get("contents", str(doc)))
            else:
                content = str(doc)
            formatted.append(f"Doc {idx + 1}: {content}")

        return "\n\n".join(formatted)

    @staticmethod
    def get_first_document_content(retrieval_result) -> str:
        if retrieval_result is None or not hasattr(retrieval_result, "documents"):
            return ""
        if not retrieval_result.documents:
            return ""
        doc = retrieval_result.documents[0]
        if hasattr(doc, "content"):
            return doc.content
        if isinstance(doc, dict):
            return doc.get("content", doc.get("contents", str(doc)))
        return str(doc)

    def generate(
        self,
        question: str,
        retrieval_result: RetrieverResult = None,
        reasoner: Any = None,
    ) -> str:
        if reasoner is None:
            raise ValueError("reasoner is required")

        max_tokens = getattr(reasoner, 'max_tokens', 256)
        temperature = getattr(reasoner, 'temperature', 0.0)
        prompt_name = getattr(reasoner, "prompt_name", "naive_rag")
        prompt_template_obj = PromptManager.get_template(prompt_name)

        reference = self.format_retrieval_result(retrieval_result)
        prompt_variables = {
            "question": question,
            "reference": reference,
        }

        try:
            system_prompt = prompt_template_obj.system_prompt.format(**prompt_variables)
            user_prompt = prompt_template_obj.user_prompt.format(**prompt_variables)
        except Exception as exc:
            raise ValueError(f"Failed to format prompt template '{prompt_name}': {exc}") from exc

        if system_prompt:
            prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            prompt = user_prompt

        answer = reasoner.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        answer_prefix = getattr(reasoner, "answer_prefix", None)
        if answer_prefix:
            return _extract_after_prefix(answer, answer_prefix)
        return answer


def format_retrieval_result(retrieval_result) -> str:
    return NaiveGenerationStrategy.format_retrieval_result(retrieval_result)


def generate_answer(
    question: str,
    retrieval_result: RetrieverResult = None,
    reasoner: Any = None,
) -> str:
    return NaiveGenerationStrategy().generate(
        question=question,
        retrieval_result=retrieval_result,
        reasoner=reasoner,
    )


def _extract_after_prefix(text: str, prefix: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    pattern = rf"(?:^|\n)\s*{re.escape(prefix)}\s*(.+)$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return text

```

### File: `rolerag/pipelines/naive_pipeline.py`

```python
from typing import Any, Dict

from RoleRAG.roles.utils import assist_role, get_method_role_config
from RoleRAG.config.types import RAGOutput
from RoleRAG.rolerag.pipelines.pipelines_base import BasePipeline, register_pipeline
from RoleRAG.models import get_model
from RoleRAG.rolerag.pipelines.generation_strategy.naive_generation_strategy import NaiveGenerationStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy import BasicRetrievalStrategy


class NaiveRAGPipeline(BasePipeline):

    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.index_cfg = config["index"]
        self.paths_cfg = self.index_cfg["paths"]
        self.bm25_index_path = self.paths_cfg["bm25_index_path"]

        self.retriever_cfg = get_method_role_config(config, "retriever", "BM25Model")
        self.retriever_cfg["index_path"] = self.bm25_index_path
        self.rewriter_cfg = get_method_role_config(config, "rewriter", "OpenAIGenerator")
        self.reasoner_cfg = get_method_role_config(config, "reasoner", "OpenAIGenerator")

        self.retriever = None
        self.rewriter = None
        self.reasoner = None
        self.retrieval_strategy = BasicRetrievalStrategy(config)
        self.generation_strategy = NaiveGenerationStrategy(config)
        self.create()

    def create(self) -> None:
        bm25_model = get_model("BM25Model")
        self.retriever = assist_role(bm25_model, "retriever", self.retriever_cfg)
        rewriter_model = get_model("OpenAIGenerator")
        self.rewriter = assist_role(
            rewriter_model,
            "rewriter",
            self.rewriter_cfg,
        )
        
        reasoner_model = get_model("OpenAIGenerator")
        self.reasoner = assist_role(
            reasoner_model,
            "reasoner",
            self.reasoner_cfg,
        )


    def _run_item(self, query: str, **kwargs) -> RAGOutput:
        retrieval_result = self.retrieval_strategy.retrieve(
            query=query,
            retriever=self.retriever,
            rewriter=self.rewriter,
        )
        self.logset("retrieval_result", retrieval_result)
        if self._is_retrieval_only():
            return self._retrieval_only_output(query, retrieval_result)

        answer = self.generation_strategy.generate(
            question=query,
            retrieval_result=retrieval_result,
            reasoner=self.reasoner,
        )
        return RAGOutput(
            query=query,
            answer=answer,
            retrieval_result=retrieval_result,
        )


register_pipeline("naive", NaiveRAGPipeline)

```

### File: `rolerag/pipelines/pipelines_base.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type

from RoleRAG.config.types import EvalItem, RAGOutput


_PIPELINE_REGISTRY: Dict[str, Type["BasePipeline"]] = {}


def register_pipeline(name: str, pipeline_class: Type["BasePipeline"]) -> None:
    _PIPELINE_REGISTRY[name] = pipeline_class


def create_pipeline(config: Dict[str, Any]) -> "BasePipeline":
    global_cfg = config["global"]
    pipeline_type = global_cfg["pipeline_type"]

    if pipeline_type not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline type: {pipeline_type}. "
            f"Available types: {list(_PIPELINE_REGISTRY.keys())}"
        )

    return _PIPELINE_REGISTRY[pipeline_type](config)


class BasePipeline(ABC):

    config: Dict[str, Any]
    retriever: Any = None
    reasoner: Any = None

    def __init__(self, config: Dict[str, Any], **kwargs):
        self.config = config
        self._log_buffer: Optional[Dict[str, Any]] = None

    def log_begin(self, sample: Optional[EvalItem] = None) -> None:
        self._log_buffer = {}

    def logset(self, key: str, value: Any) -> None:
        if self._log_buffer is not None:
            self._log_buffer[key] = value

    def logappend(self, key: str, value: Any) -> None:
        if self._log_buffer is not None:
            self._log_buffer.setdefault(key, []).append(value)

    def log_end(self) -> Dict[str, Any]:
        log_data = self._log_buffer or {}
        self._log_buffer = None
        return log_data

    def _get_eval_option(self, key: str, evaluation: Optional[Dict[str, Any]]) -> Any:
        if isinstance(evaluation, dict) and key in evaluation and evaluation[key] is not None:
            return evaluation[key]
        global_cfg = self.config["global"]
        evaluation_cfg = global_cfg["evaluation"]
        if key not in evaluation_cfg or evaluation_cfg[key] is None:
            raise KeyError(f"Evaluation config key not found: global.evaluation.{key}")
        return evaluation_cfg[key]

    def _get_method_inference_config(self) -> Dict[str, Any]:
        pipeline_type = self.config["global"]["pipeline_type"]
        inference_cfg = self.config.get("inference", {}) or {}
        method_cfg = inference_cfg.get(pipeline_type)
        if method_cfg is None and pipeline_type.startswith("graphrag_"):
            method_cfg = inference_cfg.get("graphrag")
        return method_cfg or {}

    def _is_retrieval_only(self) -> bool:
        return bool(self._get_method_inference_config().get("retrieval_only", False))

    def _retrieval_only_output(self, query: str, retrieval_result: Any, metadata: Optional[Dict[str, Any]] = None) -> RAGOutput:
        output_metadata = dict(metadata or {})
        output_metadata["retrieval_only"] = True
        return RAGOutput(
            query=query,
            answer="",
            retrieval_result=retrieval_result,
            metadata=output_metadata,
        )

    def run(self, query: Union[str, List[str]], **kwargs) -> Union[RAGOutput, List[RAGOutput]]:
        if isinstance(query, str):
            return self._run_item(query, **kwargs)
        return [self._run_item(q, **kwargs) for q in query]

    def predict(self, sample: EvalItem, **kwargs) -> RAGOutput:
        return self._run_item(sample.query, eval_item=sample, **kwargs)

    @abstractmethod
    def _run_item(self, query: str, **kwargs) -> RAGOutput:

        pass


    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError("Subclasses must implement create()")

    def evaluate(self, dataset, evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from RoleRAG.rolerag.evaluator.evaluators import RAGEvaluator

        metrics = self._get_eval_option("metrics", evaluation)
        show_progress = self._get_eval_option("show_progress", evaluation)
        save_dir = self._get_eval_option("save_dir", evaluation)
        save_metric = self._get_eval_option("save_metric", evaluation)
        verbose = self._get_eval_option("verbose", evaluation)
        print_results = self._get_eval_option("print_results", evaluation)

        evaluator = RAGEvaluator(
            pipeline=self,
            metrics=metrics,
            show_progress=show_progress,
            config=self.config,
            save_dir=save_dir,
            save_metric=save_metric,
            verbose=verbose,
        )
        results = evaluator.run(dataset)
        if print_results:
            evaluator.print_results(results)
        return results

    def evalute(self, *args, **kwargs) -> Dict[str, Any]:
        return self.evaluate(*args, **kwargs)

```

### File: `rolerag/pipelines/prompts/naive_prompt.py`

```python
from RoleRAG.rolerag.pipelines.prompts.prompt_manager import PromptTemplate


PROMPTS = {
    "naive_rag": PromptTemplate(
        system_prompt="",
        user_prompt="Answer the question based on the given documents.Output only the answer, no explanations.\n\nDocuments:\n{reference}\n\nQuestion: {question}",
    ),
    "reference_only": PromptTemplate(
        system_prompt="",
        user_prompt="Here are the relevant documents:\n\n{reference}",
    ),
    "no_reference": PromptTemplate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Question: {question}",
    ),
}

```

### File: `rolerag/pipelines/prompts/prompt_manager.py`

```python
from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from typing import Any, Dict


class PromptText(str):
    _pattern = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

    def format(self, **kwargs: Any) -> str:  # type: ignore[override]
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in kwargs:
                return str(kwargs[key])
            return match.group(0)

        return self._pattern.sub(replace, str(self))


@dataclass(frozen=True)
class PromptTemplate:

    system_prompt: PromptText = PromptText("Answer the question based on the given document.")
    user_prompt: PromptText = PromptText("Question: {question}\n\nThe following are given documents.\n\n{reference}")

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_prompt", PromptText(self.system_prompt))
        object.__setattr__(self, "user_prompt", PromptText(self.user_prompt))


class PromptManager:

    _templates: Dict[str, PromptTemplate] = {}
    _loaded = False

    @classmethod
    def _load_templates(cls) -> None:
        if cls._loaded:
            return

        package = importlib.import_module("RoleRAG.rolerag.pipelines.prompts")
        for module_info in pkgutil.iter_modules(package.__path__):
            if module_info.name == "prompt_manager":
                continue
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            prompt_map = getattr(module, "PROMPTS", None)
            if not prompt_map:
                continue
            if not isinstance(prompt_map, dict):
                raise TypeError(f"PROMPTS in {module.__name__} must be a dict")
            for name, template in prompt_map.items():
                if not isinstance(template, PromptTemplate):
                    raise TypeError(f"Prompt '{name}' in {module.__name__} must be a PromptTemplate")
                if name in cls._templates:
                    raise KeyError(f"Duplicate prompt template name: {name}")
                cls._templates[name] = template

        cls._loaded = True

    @classmethod
    def get_template(cls, name: str) -> PromptTemplate:
        
        cls._load_templates()
        if name not in cls._templates:
            raise KeyError(f"Prompt template '{name}' not found in prompts package")
        return cls._templates[name]

```

### File: `rolerag/pipelines/retrieval_strategy/__init__.py`

```python
from RoleRAG.rolerag.pipelines.retrieval_strategy.basic_retrieval_strategy import BasicRetrievalStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy.graphrag_retrieval_strategy import GraphRAGRetrievalStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy.hipporag_retrieval_strategy import HippoRAGRetrievalStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy.hybrid_retrieval_strategy import HybridRetrievalStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy.perqa_retrieval_strategy import PERQARetrievalStrategy
from RoleRAG.rolerag.pipelines.retrieval_strategy.ucgr_retrieval_strategy import UCGRRetrievalStrategy

__all__ = [
    "BasicRetrievalStrategy",
    "GraphRAGRetrievalStrategy",
    "HybridRetrievalStrategy",
    "HippoRAGRetrievalStrategy",
    "PERQARetrievalStrategy",
    "UCGRRetrievalStrategy",
]

```

### File: `rolerag/pipelines/retrieval_strategy/basic_retrieval_strategy.py`

```python
from typing import Any, List, Optional, Tuple

from RoleRAG.roles.role_base import BaseRetrieverContract
from RoleRAG.config.types import Document, RetrieverResult


class BasicRetrievalStrategy:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        inference_cfg = (self.config.get("inference", {}) or {})
        self.method_name = self.config.get("global", {}).get("pipeline_type")
        method_cfg = inference_cfg.get(self.method_name, {}) if self.method_name else {}
        self.default_return_top_k = self._to_int_or_none(method_cfg.get("return_top_k"))
        self.default_rerank_top_k = self._to_int_or_none(method_cfg.get("rerank_top_k"))

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    def _resolve_return_top_k(self, return_top_k: Optional[int], fallback: int = 20) -> int:
        if return_top_k is not None:
            return int(return_top_k)
        if self.default_return_top_k is not None:
            return self.default_return_top_k
        return fallback

    def _resolve_rerank_top_k(self, rerank_top_k: Optional[int], fallback: int) -> int:
        if rerank_top_k is not None:
            return int(rerank_top_k)
        if self.default_rerank_top_k is not None:
            return self.default_rerank_top_k
        return fallback

    def retrieve(
        self,
        query: str,
        retriever: BaseRetrieverContract,
        reranker: Any = None,
        rewriter: Any = None,
        return_top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverResult:
        if retriever is None:
            raise ValueError("BasicRetrievalStrategy.retrieve() requires a retriever")

        effective_query = query
        if rewriter is not None:
            effective_query = rewriter.rewrite(query, **kwargs)

        if reranker is None:
            result = self.simple(
                query=effective_query,
                retriever=retriever,
                return_top_k=return_top_k,
                **kwargs,
            )
        else:
            result = self.retrieve_then_rerank(
                query=effective_query,
                retriever=retriever,
                reranker=reranker,
                return_top_k=return_top_k,
                **kwargs,
            )

        if result.metadata is None:
            result.metadata = {}
        if rewriter is not None:
            result.metadata["original_query"] = query
            result.metadata["rewritten_query"] = effective_query
        return result

    def simple(
        self,
        query: str,
        retriever: BaseRetrieverContract,
        return_top_k: int = None,
        **kwargs,
    ) -> RetrieverResult:
        return_top_k = self._resolve_return_top_k(return_top_k)

        result = retriever.search(query, top_k=return_top_k, **kwargs)

        return RetrieverResult(
            documents=result.documents[:return_top_k],
            scores=result.scores[:return_top_k],
            metadata={
                "strategy": "simple",
                "query": query,
            },
        )

    def retrieve_then_rerank(
        self,
        query: str,
        retriever: BaseRetrieverContract,
        reranker: Any,
        return_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        **kwargs,
    ) -> RetrieverResult:
        return_top_k = self._resolve_return_top_k(return_top_k)
        rerank_top_k = self._resolve_rerank_top_k(rerank_top_k, fallback=return_top_k)

        result = self.simple(
            query=query,
            retriever=retriever,
            return_top_k=return_top_k,
            **kwargs,
        )

        reranked_result = self._rerank_retrieval_result(
            retrieval_result=result,
            reranker=reranker,
            query=query,
            top_k=rerank_top_k,
        )
        reranked_result.metadata["strategy"] = "retrieve_then_rerank"
        return reranked_result

    @staticmethod
    def _rerank_documents(
        documents: List[Document],
        reranker: Any,
        query: str,
        top_k: int = 10,
    ) -> Tuple[List[Document], List[float]]:
        if not documents:
            return [], []

        doc_texts = [doc.content for doc in documents]
        reranked = reranker.rerank(query, doc_texts, top_k=top_k)

        if not reranked:
            return documents, [doc.score for doc in documents if doc.score is not None]

        text_to_doc = {doc.content: doc for doc in documents}
        reranked_docs = []
        reranked_scores = []

        for doc_text, score in reranked:
            doc = text_to_doc.get(doc_text)
            if doc:
                doc.score = score
                reranked_docs.append(doc)
                reranked_scores.append(score)

        return reranked_docs, reranked_scores

    def _rerank_retrieval_result(
        self,
        retrieval_result: RetrieverResult,
        reranker: Any,
        query: str,
        top_k: int = 5,
    ) -> RetrieverResult:
        reranked_docs, reranked_scores = self._rerank_documents(
            documents=retrieval_result.documents,
            reranker=reranker,
            query=query,
            top_k=top_k,
        )

        return RetrieverResult(
            documents=reranked_docs,
            scores=reranked_scores,
            metadata=retrieval_result.metadata.copy() if retrieval_result.metadata else {},
        )

```

### File: `roles/__init__.py`

```python
from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.reranker import RerankerRoleAdapter
from RoleRAG.roles.retriever import RetrieverRoleAdapter
from RoleRAG.roles.rewriter import HyDERewriter, RewriterRoleAdapter
from RoleRAG.roles.refiner import RefinerRoleAdapter
from RoleRAG.roles.role_base import (
    BaseGeneratorContract,
    BaseRerankerContract,
    BaseRetrieverContract,
    BaseRewriterContract,
    BaseRefinerContract,
)
from RoleRAG.roles.utils import (
    _configure_model,
    assist_role,
    get_method_role_config,
    get_model_defaults,
)

__all__ = [
    "BaseGeneratorContract",
    "BaseRerankerContract",
    "BaseRetrieverContract",
    "BaseRewriterContract",
    "BaseRefinerContract",
    "ReasonerRoleAdapter",
    "HyDERewriter",
    "RerankerRoleAdapter",
    "RetrieverRoleAdapter",
    "RewriterRoleAdapter",
    "RefinerRoleAdapter",
    "_configure_model",
    "assist_role",
    "get_method_role_config",
    "get_model_defaults",
]

```

### File: `roles/reasoner.py`

```python
from typing import Any, Dict, Optional

from RoleRAG.roles.role_base import BaseGeneratorContract


class ReasonerRoleAdapter(BaseGeneratorContract):

    def __init__(
        self,
        model,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        prompt_name: str = "naive_rag",
        prompt_names: Optional[Dict[str, str]] = None,
        answer_prefix: Optional[str] = None,
        graph_index_path: Optional[str] = None,
        fm_index_path: Optional[str] = None,
        **kwargs
    ):
        self._model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.prompt_name = prompt_name
        self.prompt_names = prompt_names or {}
        self.prompt_templates = self._load_prompt_templates(self.prompt_names)
        self.answer_prefix = answer_prefix
        self.graph_index_path = graph_index_path
        self.fm_index_path = fm_index_path

    @staticmethod
    def _load_prompt_templates(prompt_names: Dict[str, str]) -> Dict[str, Any]:
        if not prompt_names:
            return {}

        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        return {
            key: PromptManager.get_template(prompt_name)
            for key, prompt_name in prompt_names.items()
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        system = system_prompt or self.system_prompt
        if hasattr(self._model, "generate"):
            if system:
                prompt = f"{system}\n\n{prompt}"
            return self._model.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        raise TypeError(f"Model {type(self._model).__name__} does not support generate")

```

### File: `roles/refiner.py`

```python
from typing import List, Optional

from RoleRAG.roles.role_base import BaseRefinerContract


class RefinerRoleAdapter(BaseRefinerContract):

    def __init__(
        self,
        model,
        rate: float = 0.33,
        method: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        self._model = model
        self.rate = rate
        self.method = method
        self.instruction = instruction
        self.model_name = getattr(model, "model_name", type(model).__name__)

    def refine(
        self,
        context: List[str],
        **kwargs,
    ) -> str:
        options = {
            "rate": kwargs.pop("rate", self.rate),
        }
        method = kwargs.pop("method", self.method)
        if method is not None:
            options["method"] = method

        question = kwargs.pop("question", None)
        if getattr(self._model, "requires_question", False):
            if not question:
                raise ValueError(f"{self.model_name} requires a non-empty question")
            options["question"] = question

        instruction = kwargs.pop("instruction", self.instruction)
        if instruction is not None:
            options["instruction"] = instruction
        options.update(kwargs)

        result = self._model.compress(context, **options)
        
        if isinstance(result, dict) and "compressed_prompt" in result:
            return result["compressed_prompt"]
        if isinstance(result, str):
            return result

```

### File: `roles/reranker.py`

```python
from typing import List, Tuple, Union

from RoleRAG.config.types import Document
from RoleRAG.roles.role_base import BaseRerankerContract


class RerankerRoleAdapter(BaseRerankerContract):

    def __init__(self, model, top_k: int = 10, **kwargs):
        self._model = model
        self.top_k = top_k
        self.model_name = self._get_model_name()

    def _get_model_name(self):
        return getattr(self._model, "model_name", type(self._model).__name__)

    def rerank(
        self,
        query: str,
        documents: Union[List[Document], List[str]],
        top_k: int = 10,
        **kwargs
    ) -> List[Tuple[Union[Document, str], float]]:
        top_k = kwargs.pop("top_k", self.top_k)

        if not documents:
            return []

        if isinstance(documents[0], Document):
            doc_strings = [doc.content for doc in documents]
            remaining_docs = list(documents)
        else:
            doc_strings = documents
            remaining_docs = None

        results = self._model.rerank(query, doc_strings, top_k=top_k)

        if remaining_docs is not None:
            reranked_results = []
            for text, score in results:
                for idx, doc in enumerate(remaining_docs):
                    if doc.content == text:
                        reranked_results.append((doc, score))
                        remaining_docs.pop(idx)
                        break
            return reranked_results
        return results

```

### File: `roles/retriever.py`

```python
from typing import Optional

from RoleRAG.config.types import RetrieverResult
from RoleRAG.roles.role_base import BaseRetrieverContract


class RetrieverRoleAdapter(BaseRetrieverContract):


    def __init__(
        self,
        model,
        top_k: int = 5,
        index_path: Optional[str] = None,
        graph_index_path: Optional[str] = None,
        **kwargs
    ):
        self._model = model
        self.top_k = top_k
        self.index_path = index_path
        self.graph_index_path = graph_index_path
        self.model_name = self._get_model_name()
        self._backend = self._detect_backend()
        if not self.index_path and not self.graph_index_path:
            raise ValueError("RetrieverRoleAdapter requires index_path or graph_index_path")

        self._indexer = None
        self._search_fn = None
        if self.index_path:
            if self._backend == "bm25":
                self._setup_fn = self._setup_bm25_retriever
                self._search_fn = self._search_bm25
            else:
                self._setup_fn = self._setup_dense_retriever
                self._search_fn = self._search_dense

            self._setup_fn()

    def _get_model_name(self):
        return getattr(self._model, "model_name", type(self._model).__name__)

    def _detect_backend(self) -> str:
        if hasattr(self._model, "get_top_k"):
            return "bm25"
        if hasattr(self._model, "encode"):
            return "dense"
        raise TypeError(f"Model {type(self._model).__name__} does not support search")

    def _setup_bm25_retriever(self):
        from RoleRAG.indexes.indexing.chunk_indexing import BM25Indexing

        self._indexer = BM25Indexing()
        self._indexer.load(self.index_path)

    def _setup_dense_retriever(self):
        from RoleRAG.indexes.indexing.chunk_indexing import DenseIndexing

        self._indexer = DenseIndexing(encoder=self._model)
        self._indexer.load(self.index_path)

    def encode(self, texts, is_query: bool = False, **kwargs):
        if not hasattr(self._model, "encode"):
            raise TypeError(
                f"Retriever backend {type(self._model).__name__} does not support encode"
            )
        return self._model.encode(texts, is_query=is_query, **kwargs)

    def search(self, query: str, **kwargs) -> RetrieverResult:
        if self._search_fn is None:
            raise ValueError("RetrieverRoleAdapter.search() requires index_path")
        top_k = kwargs.get("top_k", self.top_k)
        return self._search_fn(query, top_k)

    def _search_bm25(self, query: str, top_k: int) -> RetrieverResult:
        documents, scores = self._indexer.search(query, top_k=top_k)

        return RetrieverResult(
            documents=documents,
            scores=scores,
            metadata={"model_type": "bm25", "query": query}
        )

    def _search_dense(self, query: str, top_k: int) -> RetrieverResult:
        query_embedding = self._model.encode(query, is_query=True)
        documents, scores = self._indexer.search(query_embedding, top_k=top_k)

        return RetrieverResult(
            documents=documents,
            scores=scores,
            metadata={"model_type": "dense", "query": query}
        )
    

```

### File: `roles/rewriter.py`

```python
from typing import Any, List, Optional

from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.role_base import BaseRewriterContract


class HyDERewriter:

    def __init__(
        self,
        reasoner: Any,
        task: str = "web_search",
        language: str = "en",
        num_hypotheses: int = 8,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
    ):
        self.reasoner = reasoner
        self.task = task
        self.language = language
        self.num_hypotheses = num_hypotheses
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop if stop is not None else ["\n\n\n"]

    def build_prompt(self, query: str) -> str:
        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        template_name = f"hyde_{self.task}"
        template = PromptManager.get_template(template_name)
        if self.task == "mr_tydi":
            return template.user_prompt.format(language=self.language, query=query)
        return template.user_prompt.format(query=query)

    def generate(self, query: str, **kwargs) -> List[str]:
        from RoleRAG.rolerag.pipelines.prompts import PromptManager

        template_name = f"hyde_{self.task}"
        template = PromptManager.get_template(template_name)
        prompt = self.build_prompt(query)
        hypothesis_documents = []
        for _ in range(kwargs.pop("num_hypotheses", self.num_hypotheses)):
            text = self.reasoner.generate(
                prompt,
                system_prompt=template.system_prompt or None,
                max_tokens=kwargs.pop("max_tokens", self.max_tokens),
                temperature=kwargs.pop("temperature", self.temperature),
                stop=kwargs.pop("stop", self.stop),
                **kwargs,
            )
            if text:
                hypothesis_documents.append(str(text).strip())
        return hypothesis_documents

    def rewrite(self, query: str, **kwargs) -> str:
        hypothesis_documents = self.generate(query, **kwargs)
        if not hypothesis_documents:
            return query
        expanded_context = " ".join(doc for doc in hypothesis_documents if doc)
        if not expanded_context:
            return query
        return f"{query} {expanded_context}".strip()


class RewriterRoleAdapter(BaseRewriterContract):

    def __init__(
        self,
        model,
        method: str = "hyde",
        num_hypotheses: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.7,
        task: str = "web_search",
        language: str = "en",
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        self.method = method
        self.num_hypotheses = num_hypotheses
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.task = task
        self.language = language
        self.stop = stop

        if self.method == "hyde":
            self._model = HyDERewriter(
                reasoner=ReasonerRoleAdapter(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                task=task,
                language=language,
                num_hypotheses=num_hypotheses,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        else:
            raise ValueError(f"Unknown rewriter method: {self.method}")

    def rewrite(self, query: str, **kwargs) -> str:
        if hasattr(self._model, "rewrite"):
            return self._model.rewrite(query, **kwargs)
        raise TypeError(f"Model {type(self._model).__name__} does not support rewrite")

```

### File: `roles/role_base.py`

```python
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
        top_k: int = 10,
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

```

### File: `roles/utils.py`

```python
import os
from typing import Any, Dict, Optional

from RoleRAG.roles.reasoner import ReasonerRoleAdapter
from RoleRAG.roles.reranker import RerankerRoleAdapter
from RoleRAG.roles.retriever import RetrieverRoleAdapter
from RoleRAG.roles.rewriter import RewriterRoleAdapter
from RoleRAG.roles.refiner import RefinerRoleAdapter


_MODEL_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_ALLOWED_ROLES = {"retriever", "reranker", "reasoner", "rewriter", "refiner"}


def _load_yaml_config(filename: str) -> Dict[str, Any]:
    import yaml

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", filename)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_model_config() -> Dict[str, Any]:
    global _MODEL_CONFIG_CACHE
    if _MODEL_CONFIG_CACHE is None:
        global_cfg = _load_yaml_config("global.yaml")
        global_block = global_cfg.get("global", {}) or {}
        model_cfg = global_block.get("models", {}) or {}
        if not isinstance(model_cfg, dict):
            raise TypeError("Model config must be a mapping: global.models")
        _MODEL_CONFIG_CACHE = model_cfg
    return _MODEL_CONFIG_CACHE


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    config = _load_model_config()
    if model_name not in config:
        raise KeyError(f"Model config not found in global.yaml global.models: {model_name}")
    model_cfg = config[model_name]
    if not isinstance(model_cfg, dict):
        raise TypeError(f"Model config must be a mapping: {model_name}")
    return model_cfg.copy()


def get_method_role_config(
    config: Dict[str, Any],
    role: str,
    model_name: str,
    instance_name: str = None,
) -> Dict[str, Any]:
    role_cfg = ((config.get("roles", {}) or {}).get(role, {}) or {}).get(model_name, {}) or {}
    if not isinstance(role_cfg, dict):
        raise TypeError(f"Role config must be a mapping: role={role}, model={model_name}")
    if instance_name:
        instance_cfg = role_cfg.get(instance_name, {}) or {}
        if not isinstance(instance_cfg, dict):
            raise TypeError(
                f"Role instance config must be a mapping: role={role}, model={model_name}, instance={instance_name}"
            )
        return instance_cfg.copy()
    return role_cfg.copy()


def _configure_model(model, config: Dict[str, Any]):
    for key, value in config.items():
        if hasattr(model, key):
            setattr(model, key, value)


def assist_role(
    model,
    role: str,
    role_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    role = role.lower()
    if role not in _ALLOWED_ROLES:
        raise ValueError(
            f"Unknown role: {role}. "
            f"Available roles: retriever, reranker, reasoner, rewriter, refiner"
        )

    if hasattr(model, "clone") and callable(model.clone):
        model = model.clone()

    merged_config = role_config.copy() if role_config else {}

    if role == "retriever":
        return RetrieverRoleAdapter(
            model=model,
            top_k=merged_config.get("top_k", 5),
            index_path=merged_config.get("index_path"),
            graph_index_path=merged_config.get("graph_index_path"),
            **kwargs
        )

    if role == "reranker":
        return RerankerRoleAdapter(
            model=model,
            top_k=merged_config.get("top_k", 10),
            **kwargs
        )

    if role == "reasoner":
        return ReasonerRoleAdapter(
            model=model,
            max_tokens=merged_config.get("max_tokens", 256),
            temperature=merged_config.get("temperature", 0.0),
            system_prompt=merged_config.get("system_prompt"),
            prompt_name=merged_config.get("prompt_name", "naive_rag"),
            prompt_names=merged_config.get("prompt_names"),
            answer_prefix=merged_config.get("answer_prefix"),
            graph_index_path=merged_config.get("graph_index_path"),
            fm_index_path=merged_config.get("fm_index_path"),
            **kwargs
        )

    if role == "rewriter":
        return RewriterRoleAdapter(
            model=model,
            method=merged_config.get("method", "hyde"),
            num_hypotheses=merged_config.get("num_hypotheses", 4),
            max_tokens=merged_config.get("max_tokens", 512),
            temperature=merged_config.get("temperature", 0.7),
            task=merged_config.get("task", "web_search"),
            language=merged_config.get("language", "en"),
            stop=merged_config.get("stop"),
            **kwargs
        )

    if role == "refiner":
        return RefinerRoleAdapter(
            model=model,
            rate=merged_config.get("rate", 0.33),
            method=merged_config.get("method"),
            instruction=merged_config.get("instruction"),
            **kwargs
        )

    raise ValueError(
        f"Unknown role: {role}. "
        f"Available roles: retriever, reranker, reasoner, rewriter, refiner"
    )

```

## Recommended Collaboration Workflow

I will append a specific request after this context. Respond in the following format when you receive it:

### 1. Project Understanding Summary

- Modules involved in the current request
- Relevant call chains or data flows
- Files that must be read or supplied; if additional files are required, list their exact paths and explain why

### 2. Change Plan

- Target functionality
- Files to modify
- Files to add
- Configuration changes
- Risks
- Verification approach

### 3. Await Confirmation

End the response here. Do not output code, pseudocode, patches, or complete file contents until I explicitly confirm the plan and request code in a subsequent turn.

### 4. Output Code After Confirmation

Use the following format for code:

```text
### File: path/to/file.py

```python
Complete code or a precise targeted replacement
```
```

For a targeted replacement, specify:

- Which section to replace
- The function, class, or location where it should be inserted
- The surrounding code to use as an anchor

## My Specific Request

Describe the functionality to change, the error, and the expected behavior here. Include the full contents of additional relevant files when necessary.
