"""Semantic talent matching using sentence-transformer embeddings.

Where the TF-IDF matcher relies on lexical overlap, this engine embeds job
summaries into a dense vector space with a sentence-transformer model and ranks
them against a candidate's skills by cosine similarity. Semantically related
terms therefore match even without sharing the same words — e.g. "ML" sits
close to "machine learning", "k8s" close to "kubernetes".

The corpus is embedded once and the result is persisted to disk, so the model
only runs on the (small) query at request time.

The heavy ``sentence-transformers`` import is deferred into ``get_model`` so the
pure ranking logic (``rank_matches``) stays importable and unit-testable without
downloading a model.
"""

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Cache loaded models per process so Streamlit reruns don't reload from disk.
_MODEL_CACHE = {}


def get_model(model_name=DEFAULT_MODEL):
    """Load (and cache) a SentenceTransformer model by name."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts, model_name=DEFAULT_MODEL, batch_size=64):
    """Embed an iterable of texts into L2-normalized float32 vectors.

    Normalizing means a later dot product is exactly cosine similarity.
    """
    model = get_model(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def load_or_build_embeddings(texts, cache_path, model_name=DEFAULT_MODEL):
    """Return corpus embeddings, building and persisting them if needed.

    The cache is reused only when its row count matches ``texts`` (a cheap guard
    against a stale cache after the dataset changes); otherwise it is rebuilt.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    texts = list(texts)

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached

    embeddings = embed_texts(texts, model_name=model_name)
    np.save(cache_path, embeddings)
    return embeddings


def rank_matches(query_embedding, corpus_embeddings, top_n=5):
    """Rank corpus rows by cosine similarity to a query embedding.

    Both inputs are assumed L2-normalized (see ``embed_texts``), so the dot
    product is the cosine similarity. Returns ``(indices, scores)`` for the
    ``top_n`` most similar rows, sorted from best to worst.
    """
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    corpus = np.asarray(corpus_embeddings, dtype=np.float32)
    if corpus.ndim != 2 or corpus.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    similarities = corpus @ query
    n = min(top_n, similarities.shape[0])
    # argpartition for the top-n, then sort just those n by score (descending).
    candidate_idx = np.argpartition(-similarities, n - 1)[:n]
    ordered = candidate_idx[np.argsort(-similarities[candidate_idx])]
    return ordered, similarities[ordered]
