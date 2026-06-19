import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, VECTORSTORE_DIR

# Module-level singleton so the model loads once per session
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns the embedding model, loading it once and reusing it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_or_load_vector_store(chunks: list, cache_key: str) -> FAISS:
    """
    Loads a saved FAISS index for this set of PDFs if it exists,
    otherwise builds one from scratch and saves it.
    This means re-uploading the same paper(s) is instant.
    """
    index_path = os.path.join(VECTORSTORE_DIR, cache_key)
    embeddings = get_embeddings()

    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    return vector_store
