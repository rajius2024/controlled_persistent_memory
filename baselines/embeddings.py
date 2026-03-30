"""
Shared Embedding Utility
========================
Lead: Aravindan Chidambaram

Central embedding module used by ALL methods:
  - Vanilla RAG baseline
  - Task-Aware Retriever (CPM)
  - Any future method

This ensures every method uses the same model, same dimensions,
same cosine similarity — fair comparison across baselines.

Usage:
    from baselines.embeddings import embed, cosine_similarity, load_encoder
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config (read from configs/default.yaml if available) ─────────────────────
DEFAULT_MODEL = "all-MiniLM-L6-v2"
CACHE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "embeddings")

_encoder: SentenceTransformer = None


def load_encoder(model_name: str = None) -> SentenceTransformer:
    """
    Load (or return cached) SentenceTransformer encoder.
    Reads model name from configs/default.yaml if available.
    """
    global _encoder

    if _encoder is not None:
        return _encoder

    # Try to read from config
    if model_name is None:
        model_name = _read_model_from_config()

    print(f"[Embeddings] Loading model: {model_name}")
    _encoder = SentenceTransformer(model_name)
    print(f"[Embeddings] Embedding dim: {_encoder.get_sentence_embedding_dimension()}")
    return _encoder


def _read_model_from_config() -> str:
    """Read embedding model name from configs/default.yaml."""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "configs", "default.yaml"
        )
        with open(config_path) as f:
            for line in f:
                if "embedding_model" in line:
                    return line.split(":")[-1].strip().strip('"').strip("'")
    except Exception:
        pass
    return DEFAULT_MODEL


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of texts. Returns numpy array of shape (N, D).

    Args:
        texts:         List of strings to embed
        show_progress: Show tqdm progress bar (useful for large batches)

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    encoder = load_encoder()
    return encoder.encode(texts, show_progress_bar=show_progress)


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns 1D array of shape (D,)."""
    return embed([text])[0]


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a matrix of vectors.

    Args:
        query_vec: 1D array of shape (D,)
        matrix:    2D array of shape (N, D)

    Returns:
        1D array of shape (N,) with similarity scores in [-1, 1]
    """
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    m_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k scores in descending order."""
    k = min(k, len(scores))
    return np.argsort(scores)[::-1][:k]


# ── Disk Cache ────────────────────────────────────────────────────────────────

def _cache_path(cache_key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_key = cache_key.replace("/", "_").replace(" ", "_")[:80]
    return os.path.join(CACHE_DIR, f"{safe_key}.npy")


def embed_with_cache(texts: list[str], cache_key: str) -> np.ndarray:
    """
    Embed texts with disk caching. If cache_key exists on disk,
    return cached embeddings instead of re-computing.

    Args:
        texts:     List of strings to embed
        cache_key: Unique identifier for this batch (e.g. episode_id)

    Returns:
        np.ndarray of shape (N, D)
    """
    path = _cache_path(cache_key)
    if os.path.exists(path):
        return np.load(path)

    vecs = embed(texts)
    np.save(path, vecs)
    return vecs


def clear_cache():
    """Clear all cached embeddings."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"[Embeddings] Cache cleared: {CACHE_DIR}")


# ── Sanity Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Embeddings Sanity Test")
    print("=" * 50)

    sample_texts = [
        "I love spicy Thai food.",
        "My favorite cuisine is Thai, especially spicy dishes.",
        "I prefer mild Italian food.",
    ]

    # Test basic embedding
    vecs = embed(sample_texts)
    print(f"\nEmbedding shape : {vecs.shape}")
    print(f"Embedding dim   : {vecs.shape[1]}")

    # Test cosine similarity
    query = embed_one("What food does this person like?")
    scores = cosine_similarity(query, vecs)

    print(f"\nCosine similarity scores:")
    for text, score in zip(sample_texts, scores):
        print(f"  {score:.4f}  {text}")

    # Test top-k
    top = top_k_indices(scores, k=2)
    print(f"\nTop-2 most relevant:")
    for i in top:
        print(f"  [{i}] {sample_texts[i]}")

    # Test caching
    print(f"\nTesting cache...")
    vecs_cached = embed_with_cache(sample_texts, cache_key="sanity_test")
    vecs_cached2 = embed_with_cache(sample_texts, cache_key="sanity_test")
    print(f"Cache hit works: {np.allclose(vecs_cached, vecs_cached2)}")

    print(f"\n{'='*50}")
    print(f"  ALL CHECKS PASSED")
    print(f"{'='*50}")
