import os


def make_index_name(
    model_name: str,
    corpus_path: str,
    chunk_size: int = None,
    faiss_type: str = None,
) -> str:
    model_short = model_name.split("/")[-1]
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
    name = f"{model_short}_{corpus_name}"
    if chunk_size:
        name += f"_c{chunk_size}"
    if faiss_type and model_short != "BM25Model":
        name += f"_{faiss_type.lower()}"
    return name
