"""Tiny dependency-free BM25 retriever over bundled markdown docs. Works offline."""
from __future__ import annotations

import math
import re
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self._doc_tokens = [_tokenize(d["title"] + " " + d["text"]) for d in docs]
        self._df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = sum(len(t) for t in self._doc_tokens) / max(len(self._doc_tokens), 1)
        self._n = len(docs)

    def _score(self, query_terms: list[str], idx: int) -> float:
        toks = self._doc_tokens[idx]
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks)
        score = 0.0
        for term in set(query_terms):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
            f = tf.get(term, 0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * (f * (self.k1 + 1) / denom)
        return score

    def search(self, query: str, k: int = 3) -> list[dict]:
        terms = _tokenize(query)
        ranked = sorted(
            range(self._n), key=lambda i: self._score(terms, i), reverse=True
        )
        out = []
        for i in ranked[:k]:
            d = dict(self.docs[i])
            d["score"] = round(self._score(terms, i), 4)
            if d["score"] > 0:
                out.append(d)
        return out


def load_docs(data_dir: str | Path) -> list[dict]:
    docs = []
    for path in sorted(Path(data_dir, "docs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        docs.append({"id": path.stem, "title": title, "text": text})
    return docs
