"""FAISS-backed RAG over the FAQ/KB documents in knowledge/."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger("voice.rag")

DEFAULT_MIN_SCORE: dict[str, float] = {
    "fastembed": 0.42,
    "hashing": 0.34,
}

# how much of the blended retrieval score comes from the embedding cosine
EMBEDDER_WEIGHT: dict[str, float] = {
    "fastembed": 0.6,
    "hashing": 0.4,
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "for", "and", "or", "with", "your", "you", "your", "my", "me", "i", "it", "that",
    "this", "those", "these", "at", "as", "by", "from", "what", "when", "where", "which",
    "how", "do", "does", "did", "can", "could", "will", "would", "should", "if", "then",
    "there", "their", "about", "into", "over", "after", "before", "so", "than", "too",
    "very", "just", "any", "some", "out", "up", "down", "off", "again", "please",
    "have", "has", "had", "get", "got", "let", "tell", "say", "go", "going", "am",
    "being", "need", "want", "make", "made", "come", "take", "taken", "us",
}

_SUFFIXES = ("ss", "us", "is", "oes", "ches", "shes", "xes", "ies")

# below this chunk count every document is scored during retrieval
_CANDIDATE_CAP = 512

# a single query term must appear in at least this many chunks to ground an answer
_SINGLE_TOKEN_MIN_DF = 3


def _build_idf(
    body_tokens: list[set[str]], title_tokens: list[set[str]], doc_count: int
) -> tuple[dict[str, float], dict[str, int]]:
    df: Counter[str] = Counter()
    for body, title in zip(body_tokens, title_tokens):
        for token in body | title:
            df[token] += 1
    idf = {
        token: math.log((doc_count + 1) / (count + 1)) + 1.0
        for token, count in df.items()
    }
    return idf, dict(df)


def normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("s") and not token.endswith(_SUFFIXES):
        token = token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9']+", text.lower()):
        if raw in STOPWORDS:
            continue
        token = normalize_token(raw)
        if token in STOPWORDS:
            continue
        out.add(token)
    return out


@dataclass
class Chunk:
    doc_id: str
    title: str
    text: str
    source: str
    order: int

    def preview(self, limit: int = 160) -> str:
        flat = " ".join(self.text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "..."


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "score": round(self.score, 4),
            "source": self.chunk.source,
        }


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Dependency-free lexical embedder: hashed word/char n-gram features.

    Used as an offline fallback so retrieval works without model downloads.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def _features(self, text: str) -> Counter[str]:
        tokens = self._tokenize(text)
        feats: Counter[str] = Counter(f"w:{t}" for t in tokens)
        feats.update(f"b:{a} {b}" for a, b in zip(tokens, tokens[1:]))
        flat = " ".join(tokens)
        feats.update(f"c:{flat[i : i + 3]}" for i in range(max(0, len(flat) - 2)))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text).items():
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[row, idx] += sign * (1.0 + math.log1p(weight))
        return _l2_normalize(out)


class FastEmbedEmbedder:
    """Sentence embedding via fastembed (ONNX, local)."""

    name = "fastembed"

    def __init__(self, model_name: str, cache_dir: str) -> None:
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = int(np.asarray(probe).shape[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = [np.asarray(v, dtype=np.float32) for v in self._model.embed(texts)]
        return _l2_normalize(np.vstack(vectors))


class VectorStore:
    """Cosine-similarity index: FAISS IndexFlatIP when available, numpy otherwise."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.backend = "numpy"
        self._vectors: np.ndarray | None = None
        self._index = None
        try:
            import faiss  # noqa: PLC0415

            self._index = faiss.IndexFlatIP(dim)
            self.backend = "faiss"
        except Exception as exc:  # noqa: BLE001
            logger.warning("faiss unavailable (%s); using numpy cosine index", exc)

    @property
    def size(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._index is not None:
            self._index.add(vectors)
            return
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        k = max(1, min(k, max(1, self.size)))
        if self._index is not None:
            scores, idx = self._index.search(queries, k)
            return scores, idx
        assert self._vectors is not None
        sims = queries @ self._vectors.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx


def load_chunks(kb_path: str | Path, chunk_chars: int = 700, overlap: int = 120) -> list[Chunk]:
    root = Path(kb_path)
    chunks: list[Chunk] = []
    if not root.exists():
        logger.warning("kb path %s does not exist", root)
        return chunks
    for path in sorted(root.glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        source = path.name
        doc_title = source
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line.startswith("# "):
            doc_title = first_line[2:].strip()
            text = text[len(first_line) :]
        sections = _split_sections(text)
        if not sections:
            sections = [(doc_title, text.strip())]
        for idx, (heading, body) in enumerate(sections):
            title = heading or doc_title
            for piece in _window(body, chunk_chars, overlap):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        doc_id=f"{source}#{idx}",
                        title=title,
                        text=piece.strip(),
                        source=source,
                        order=len(chunks),
                    )
                )
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if buf or heading:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def _window(text: str, chunk_chars: int, overlap: int) -> list[str]:
    if len(text) <= chunk_chars:
        return [text]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_chars:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            pieces.append(current)
        if len(para) <= chunk_chars:
            current = para
        else:
            step = max(1, chunk_chars - overlap)
            for start in range(0, len(para), step):
                pieces.append(para[start : start + chunk_chars])
            current = ""
    if current:
        pieces.append(current)
    return pieces


@dataclass
class RAGEngine:
    chunks: list[Chunk]
    embedder: Embedder | None
    store: VectorStore | None
    min_score: float
    top_k: int = 4
    provider: str = "none"
    embedder_weight: float = 0.5
    index_state: dict[str, Any] = field(default_factory=dict)
    _body_tokens: list[set[str]] = field(default_factory=list, repr=False)
    _title_tokens: list[set[str]] = field(default_factory=list, repr=False)
    _corpus_vocab: set[str] = field(default_factory=set, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _df: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, settings: Any) -> RAGEngine:
        chunks = load_chunks(settings.kb_path, settings.rag_chunk_chars, settings.rag_chunk_overlap)
        embedder, provider = _build_embedder(settings)
        min_score = (
            settings.rag_min_score
            if settings.rag_min_score is not None
            else DEFAULT_MIN_SCORE.get(provider, 0.35)
        )
        weight = EMBEDDER_WEIGHT.get(provider, 0.5)
        if embedder is None or not chunks:
            logger.warning(
                "RAG disabled (provider=%s, chunks=%d)", provider, len(chunks)
            )
            return cls(
                chunks=chunks,
                embedder=None,
                store=None,
                min_score=min_score,
                top_k=settings.rag_top_k,
                provider=provider,
                embedder_weight=weight,
            )
        vectors = embedder.embed([c.text for c in chunks])
        store = VectorStore(vectors.shape[1])
        store.add(vectors)
        engine = cls(
            chunks=chunks,
            embedder=embedder,
            store=store,
            min_score=min_score,
            top_k=settings.rag_top_k,
            provider=provider,
            embedder_weight=weight,
        )
        engine._body_tokens = [content_tokens(c.text) for c in chunks]
        engine._title_tokens = [content_tokens(c.title) for c in chunks]
        engine._corpus_vocab = (
            set().union(*engine._body_tokens).union(*engine._title_tokens)
            if chunks
            else set()
        )
        engine._idf, engine._df = _build_idf(engine._body_tokens, engine._title_tokens, len(chunks))
        engine.index_state = {
            "chunks": len(chunks),
            "dim": int(vectors.shape[1]),
            "backend": store.backend,
            "scoring": "idf-coverage+cosine",
            "sources": sorted({c.source for c in chunks}),
        }
        logger.info(
            "rag index ready: %s chunks=%d dim=%d backend=%s min_score=%.2f",
            provider,
            len(chunks),
            vectors.shape[1],
            store.backend,
            min_score,
        )
        return engine

    @property
    def ready(self) -> bool:
        return self.embedder is not None and self.store is not None and bool(self.chunks)

    def lexical_score(self, query_tokens: set[str], index: int) -> float:
        """IDF-weighted fraction of the query terms covered by one document."""
        if not query_tokens:
            return 0.0
        total = sum(self._idf.get(t, 0.0) for t in query_tokens)
        if total <= 0:
            return 0.0
        body = self._body_tokens[index] if index < len(self._body_tokens) else set()
        title = self._title_tokens[index] if index < len(self._title_tokens) else set()
        numerator = sum(self._idf.get(t, 0.0) for t in query_tokens if t in body or t in title)
        title_bonus = 0.5 * sum(self._idf.get(t, 0.0) for t in query_tokens if t in title)
        return min((numerator + title_bonus) / total, 1.0)

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        if not self.ready or not query.strip():
            return []
        assert self.embedder is not None and self.store is not None
        k = k or self.top_k
        # for small corpora (and for the hashing embedder) score every chunk:
        # raw cosine alone is too weak to pick a safe candidate subset.
        candidates = self.store.size if self.store.size <= _CANDIDATE_CAP else max(k, k * 8)
        vector = self.embedder.embed([query])
        scores, idx = self.store.search(vector, candidates)
        query_tokens = content_tokens(query)
        known = query_tokens & self._corpus_vocab if self._corpus_vocab else query_tokens
        # a lone shared word is too weak to ground an answer ("weekend" appears in
        # one shipping passage, but a weather question is still out of domain);
        # allow it only when the word is common across the knowledge base.
        if len(known) == 1:
            only = next(iter(known))
            if self._df.get(only, 0) < _SINGLE_TOKEN_MIN_DF:
                known = set()
        blended: list[tuple[float, int, float]] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            cosine = float(score)
            lex = self.lexical_score(known, int(i))
            final = self.embedder_weight * cosine + (1.0 - self.embedder_weight) * lex
            blended.append((final, int(i), cosine))
        blended.sort(key=lambda item: item[0], reverse=True)
        return [
            Hit(chunk=self.chunks[i], score=score)
            for score, i, _ in blended[:k]
        ]

    def grounded_hits(self, hits: list[Hit]) -> list[Hit]:
        return [h for h in hits if h.score >= self.min_score]

    def build_context(self, hits: list[Hit], max_chars: int = 2400) -> str:
        parts: list[str] = []
        used = 0
        for n, hit in enumerate(self.grounded_hits(hits), start=1):
            block = f"[{n}] ({hit.chunk.title}) {hit.chunk.text}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "ready": self.ready,
            "min_score": self.min_score,
            "top_k": self.top_k,
            "index": self.index_state,
            "chunks": len(self.chunks),
        }


def _build_embedder(settings: Any) -> tuple[Embedder | None, str]:
    provider = (settings.embeddings_provider or "auto").lower()
    if provider == "none":
        return None, "none"
    if provider in ("fastembed", "auto"):
        try:
            return FastEmbedEmbedder(settings.embeddings_model, settings.embedding_cache_dir), (
                "fastembed"
            )
        except Exception as exc:  # noqa: BLE001
            if provider == "fastembed":
                raise
            logger.warning("fastembed unavailable (%s); falling back to hashing embedder", exc)
    if provider in ("hashing", "auto"):
        return HashingEmbedder(), "hashing"
    raise ValueError(f"unknown embeddings provider: {provider}")
