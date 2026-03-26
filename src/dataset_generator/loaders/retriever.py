"""TF-IDF based retriever for semantic document chunk selection."""

from __future__ import annotations

import math
import re
from collections import Counter


class TFIDFRetriever:
    """Retrieve relevant document chunks using TF-IDF similarity.

    Zero external dependencies — uses pure Python TF-IDF with cosine similarity.
    """

    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self._doc_vectors: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self) -> None:
        """Compute TF vectors and IDF scores for all documents."""
        n = len(self.documents)
        if n == 0:
            return

        # Document frequency
        df: Counter[str] = Counter()
        for doc in self.documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1

        # IDF: log(N / df)
        self._idf = {token: math.log(n / count) for token, count in df.items() if count < n}

        # TF-IDF vectors per document
        for doc in self.documents:
            tf = Counter(self._tokenize(doc))
            vec = Counter({t: freq * self._idf.get(t, 0) for t, freq in tf.items()})
            self._doc_vectors.append(vec)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve the top-k most relevant documents for the query.

        Args:
            query: Search query text.
            k: Number of documents to return.

        Returns:
            List of document texts, sorted by relevance (most relevant first).
        """
        if not self.documents:
            return []

        # Compute query vector
        qtf = Counter(self._tokenize(query))
        qvec = Counter({t: freq * self._idf.get(t, 0) for t, freq in qtf.items()})

        # Cosine similarity with each document
        scores = []
        for i, dvec in enumerate(self._doc_vectors):
            score = _cosine_similarity(qvec, dvec)
            scores.append((score, i))

        # Sort by score descending, return top-k
        scores.sort(reverse=True)
        return [self.documents[idx] for _, idx in scores[:k]]


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Compute cosine similarity between two sparse vectors."""
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
