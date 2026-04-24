"""
4 memory backends for the Multi-Memory Agent.

  1. ShortTermMemory   — sliding-window conversation buffer
  2. LongTermProfile   — JSON-backed KV store (Redis-compatible interface)
  3. EpisodicMemory    — timestamped JSON episode log
  4. SemanticMemory    — keyword/TF-IDF search store (Chroma/FAISS interface)
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Short-term memory  (ConversationBufferMemory / sliding window)
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """Keeps the last *max_messages* turns in an in-memory ring buffer."""

    def __init__(self, max_messages: int = 8) -> None:
        self.max_messages = max_messages
        self._buffer: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._buffer.append({"role": role, "content": content})
        if len(self._buffer) > self.max_messages:
            self._buffer = self._buffer[-self.max_messages:]

    def get(self) -> List[Dict[str, str]]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def token_estimate(self) -> int:
        """Rough word-count proxy for token usage."""
        return sum(len(m["content"].split()) for m in self._buffer)

    def trim_to_budget(self, budget_words: int) -> None:
        """Drop oldest messages until total word count fits budget."""
        while self._buffer and self.token_estimate() > budget_words:
            self._buffer.pop(0)


# ---------------------------------------------------------------------------
# 2. Long-term profile  (dict / JSON — Redis-compatible interface)
# ---------------------------------------------------------------------------

class LongTermProfile:
    """
    Persistent key-value store for user profile facts.

    Interface mirrors a Redis HSET/HGET pattern so it can be swapped for
    an actual Redis client without changing the agent code.
    """

    def __init__(self, storage_path: str = "profile_store.json") -> None:
        self._path = storage_path
        self._store: Dict[str, Dict[str, Any]] = {}
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                self._store = json.load(f)

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)

    # -- public API -----------------------------------------------------------

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        return dict(self._store.get(user_id, {}))

    def update(self, user_id: str, facts: Dict[str, Any]) -> None:
        """
        Merge *facts* into the user profile.
        New values ALWAYS override old ones (conflict resolution: last-write-wins).
        Each key carries a _ts_<key> timestamp so history is auditable.
        """
        if user_id not in self._store:
            self._store[user_id] = {}
        now = datetime.now().isoformat()
        for key, value in facts.items():
            self._store[user_id][key] = value
            self._store[user_id][f"_ts_{key}"] = now
        self._save()

    def delete_key(self, user_id: str, key: str) -> bool:
        """Delete a single fact — supports right-to-erasure / TTL workflows."""
        profile = self._store.get(user_id, {})
        removed = False
        for k in [key, f"_ts_{key}"]:
            if k in profile:
                del profile[k]
                removed = True
        if removed:
            self._save()
        return removed

    def delete_user(self, user_id: str) -> None:
        """Hard-delete all data for a user (GDPR right-to-erasure)."""
        if user_id in self._store:
            del self._store[user_id]
            self._save()


# ---------------------------------------------------------------------------
# 3. Episodic memory  (JSON episode log)
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Append-only log of significant conversation episodes.

    Each episode stores: who, what happened, outcome, and tags for retrieval.
    Supports keyword search to surface relevant past experiences.
    """

    def __init__(self, storage_path: str = "episodic_log.json") -> None:
        self._path = storage_path
        self._episodes: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                self._episodes = json.load(f)

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._episodes, f, indent=2, ensure_ascii=False)

    def add_episode(
        self,
        user_id: str,
        summary: str,
        outcome: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        episode = {
            "id": len(self._episodes) + 1,
            "user_id": user_id,
            "summary": summary,
            "outcome": outcome,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
        }
        self._episodes.append(episode)
        self._save()
        return episode

    def get_recent(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        user_eps = [e for e in self._episodes if e["user_id"] == user_id]
        return user_eps[-limit:]

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Keyword overlap search over episode summaries + tags."""
        candidates = self._episodes
        if user_id:
            candidates = [e for e in candidates if e["user_id"] == user_id]

        query_tokens = set(_tokenize(query))
        scored: List[tuple] = []
        for ep in candidates:
            text = ep["summary"] + " " + " ".join(ep.get("tags", []))
            ep_tokens = set(_tokenize(text))
            overlap = len(query_tokens & ep_tokens)
            if overlap:
                scored.append((overlap, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def delete_user(self, user_id: str) -> None:
        self._episodes = [e for e in self._episodes if e["user_id"] != user_id]
        self._save()


# ---------------------------------------------------------------------------
# 4. Semantic memory  (keyword TF-IDF store — Chroma/FAISS fallback)
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Document store with TF-IDF keyword retrieval.

    Interface mirrors a Chroma collection (add / query) so it can be replaced
    by a real vector database without changing the agent code.
    """

    def __init__(self, storage_path: str = "semantic_store.json") -> None:
        self._path = storage_path
        self._docs: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
                self._docs = data.get("documents", [])

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"documents": self._docs}, f, indent=2, ensure_ascii=False)

    # -- Chroma-compatible API ------------------------------------------------

    def add(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        doc = {
            "id": len(self._docs) + 1,
            "content": content,
            "metadata": metadata or {},
            "added_at": datetime.now().isoformat(),
        }
        self._docs.append(doc)
        self._save()
        return doc

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top-k documents ranked by TF-IDF cosine similarity."""
        if not self._docs:
            return []

        corpus = [d["content"] for d in self._docs]
        scores = _tfidf_scores(query_text, corpus)
        ranked = sorted(
            zip(scores, self._docs), key=lambda x: x[0], reverse=True
        )
        return [doc for score, doc in ranked[:top_k] if score > 0]

    def delete_by_metadata(self, key: str, value: Any) -> int:
        before = len(self._docs)
        self._docs = [
            d for d in self._docs if d.get("metadata", {}).get(key) != value
        ]
        if len(self._docs) < before:
            self._save()
        return before - len(self._docs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "tôi", "bạn", "là", "có", "và", "của", "một", "này", "không", "được",
    "the", "a", "an", "is", "are", "was", "were", "i", "you", "it", "in",
    "on", "at", "to", "for", "of", "and", "or", "my", "your",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    buf = ""
    for ch in text:
        if ch.isalnum() or ch in "àáảãạăắặằẳẵâấầẩẫậèéẹẻẽêếềệểễìíỉĩịòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ":
            buf += ch
        else:
            if buf and buf not in _STOPWORDS:
                tokens.append(buf)
            buf = ""
    if buf and buf not in _STOPWORDS:
        tokens.append(buf)
    return tokens


def _tfidf_scores(query: str, corpus: List[str]) -> List[float]:
    """Compute TF-IDF cosine similarity between query and each corpus doc."""
    n = len(corpus)
    query_tokens = _tokenize(query)

    # IDF: log( (N+1) / (df+1) ) + 1
    df: Counter = Counter()
    tokenized_corpus = [_tokenize(doc) for doc in corpus]
    for tokens in tokenized_corpus:
        for tok in set(tokens):
            df[tok] += 1

    def tfidf_vec(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec: Dict[str, float] = {}
        for tok, cnt in tf.items():
            idf = math.log((n + 1) / (df.get(tok, 0) + 1)) + 1
            vec[tok] = (cnt / total) * idf
        return vec

    q_vec = tfidf_vec(query_tokens)
    scores: List[float] = []
    for tokens in tokenized_corpus:
        d_vec = tfidf_vec(tokens)
        # cosine similarity
        dot = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in q_vec)
        q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1e-9
        d_norm = math.sqrt(sum(v ** 2 for v in d_vec.values())) or 1e-9
        scores.append(dot / (q_norm * d_norm))
    return scores
