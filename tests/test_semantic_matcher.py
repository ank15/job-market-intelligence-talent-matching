"""Unit tests for the semantic matcher's ranking and caching logic.

These tests exercise the pure-numpy parts only; they never load a
sentence-transformer model, so they run fast and offline.
"""

import numpy as np
import pytest

from src.matching import semantic_matcher as sm


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class TestRankMatches:
    def test_orders_by_similarity(self):
        corpus = np.stack([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
        query = _unit([1, 0])
        idx, scores = sm.rank_matches(query, corpus, top_n=3)
        # Closest first: row 0 (identical), then row 1, then the orthogonal row 2.
        assert list(idx) == [0, 1, 2]
        assert scores[0] >= scores[1] >= scores[2]
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_respects_top_n(self):
        corpus = np.stack([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0, 1])])
        idx, scores = sm.rank_matches(_unit([1, 0]), corpus, top_n=2)
        assert len(idx) == 2
        assert len(scores) == 2

    def test_accepts_2d_query(self):
        corpus = np.stack([_unit([1, 0]), _unit([0, 1])])
        idx, _ = sm.rank_matches(_unit([1, 0]).reshape(1, -1), corpus, top_n=1)
        assert idx[0] == 0

    def test_empty_corpus(self):
        idx, scores = sm.rank_matches(_unit([1, 0]), np.empty((0, 2)), top_n=5)
        assert len(idx) == 0
        assert len(scores) == 0


class TestLoadOrBuildEmbeddings:
    def test_uses_cache_without_loading_model(self, tmp_path, monkeypatch):
        # A matching cache must be reused without ever touching the model.
        cache = tmp_path / "emb.npy"
        precomputed = np.ones((3, 4), dtype=np.float32)
        np.save(cache, precomputed)

        def _boom(*args, **kwargs):
            raise AssertionError("model should not be loaded on a cache hit")

        monkeypatch.setattr(sm, "get_model", _boom)

        out = sm.load_or_build_embeddings(["a", "b", "c"], cache)
        assert np.array_equal(out, precomputed)

    def test_rebuilds_when_row_count_mismatches(self, tmp_path, monkeypatch):
        cache = tmp_path / "emb.npy"
        np.save(cache, np.ones((2, 4), dtype=np.float32))  # stale: only 2 rows

        def _fake_embed(texts, **kwargs):
            return np.zeros((len(list(texts)), 4), dtype=np.float32)

        monkeypatch.setattr(sm, "embed_texts", _fake_embed)

        out = sm.load_or_build_embeddings(["a", "b", "c"], cache)
        assert out.shape == (3, 4)
        # And the rebuilt embeddings are persisted back to the cache.
        assert np.load(cache).shape == (3, 4)
