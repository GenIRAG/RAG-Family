#!/usr/bin/env python

import argparse
import asyncio
import logging
import os
import shutil
import sys
import traceback
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
from RoleRAG.errors import format_cli_error
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


def main(argv=None):
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
    parser.add_argument("--debug", action="store_true", help="Show the full exception traceback when a command fails")

    args = parser.parse_args(argv)

    try:
        _run_commands(args)
    except KeyboardInterrupt:
        print("RoleRAG interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"RoleRAG failed.\n{format_cli_error(exc)}", file=sys.stderr)
            print("Use --debug to show the full traceback.", file=sys.stderr)
        return 1
    return 0


def _run_commands(args):
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
    sys.exit(main())
