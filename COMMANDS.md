## Index

Build a dense index:

```bash
python main.py --build_index dense
python main.py --build_index dense --indexing_model HFEncoder
python main.py --build_index dense --corpus data/hotpot_hippo.json --indexing_model HFEncoder
```

Build a BM25 index:

```bash
python main.py --build_index bm25
python main.py --build_index bm25 --corpus data/2wikimultihopqa_t1.jsonl
```

Build a Graph index:

```bash
python main.py --build_index hipporag
python main.py --build_index hipporag --indexing_model HFEncoder
python main.py --build_index graphrag
python main.py --build_index graphrag --corpus data/hotpot_hippo.jsonl --indexing_model HFEncoder
```

Build a FM index:

```bash
python main.py --build_index fmindex
python main.py --build_index fmindex --corpus data/topiocqa_grpo_test_index.jsonl
```

## Pipelines

Run Naive RAG:

```bash
python main.py --pipeline naive --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
```

Run LLMLingua-2 RAG:

```bash
python main.py --pipeline llmlingua2 --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
```

Run LLMLingua RAG:

```bash
python main.py --pipeline llmlingua --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
```

To use LongLLMLingua, set `roles.refiner.LLMLinguaModel.method` to
`"longllmlingua"` in `config/methods/llmlingua.yaml`, then run the same
`llmlingua` pipeline command.

Run PERQA:

```bash
python main.py --pipeline perqa --run_pipeline "What is the cause of death of director of film Vrijdag?"
```

Run HippoRAG:

```bash
python main.py --pipeline hipporag --run_pipeline "Were Scott Derrickson and Ed Wood of the same nationality?"
```

## Evaluate

```bash
python main.py --evaluate
python main.py --pipeline hybrid --evaluate
python main.py --pipeline llmlingua2 --evaluate
python main.py --pipeline llmlingua --evaluate
python main.py --pipeline rerank --evaluate
python main.py --pipeline perqa --evaluate
python main.py --pipeline hipporag --evaluate
python main.py --pipeline graphrag_local --evaluate
python main.py --pipeline graphrag_global --evaluate
python main.py --pipeline ucgr --evaluate
```
