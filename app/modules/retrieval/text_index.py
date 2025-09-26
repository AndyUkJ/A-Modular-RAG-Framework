# app/modules/retrieval/text_index.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
import re


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", (text or "").lower()) if t]


class BM25LiteIndex:
    """
    轻量 BM25（句子粒度），无第三方依赖：
      - 输入：docs.jsonl，每行: {"doc_id","title","sent_id","text"}
      - 检索：search(queries, top_k)
    """
    def __init__(self, path: str, k1: float = 1.5, b: float = 0.75):
        self.path = Path(path)
        self.k1 = k1
        self.b = b

        self.docs: List[Dict[str, Any]] = []
        self.N = 0
        self.avgdl = 0.0
        self.tf: Dict[str, Dict[int, int]] = {}   # term -> {doc_idx: tf}
        self.df: Dict[str, int] = {}              # term -> df
        self.doc_lens: List[int] = []             # tokens count per doc

        if self.path.exists():
            self._build()

    def _build(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                toks = _tokenize(text)
                doc_idx = len(self.docs)
                self.docs.append(obj)
                self.doc_lens.append(len(toks))
                for t in toks:
                    bucket = self.tf.setdefault(t, {})
                    bucket[doc_idx] = bucket.get(doc_idx, 0) + 1
        self.N = len(self.docs)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0
        for term, postings in self.tf.items():
            self.df[term] = len(postings)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) if self.N > 0 else 0.0

    def _score_doc(self, q_terms: List[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.doc_lens[doc_idx] if doc_idx < len(self.doc_lens) else 0
        for t in q_terms:
            f = self.tf.get(t, {}).get(doc_idx, 0)
            if f == 0:
                continue
            idf = self._idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * (f * (self.k1 + 1)) / (denom or 1.0)
        return score

    def search(self, queries: List[str], top_k: int = 20, alpha_merge: str = "max") -> List[Tuple[int, float]]:
        """
        返回 [(doc_idx, score)]；多查询合并：
          - alpha_merge="max" 取每个文档在多查询下的最大得分
          - alpha_merge="sum" 求和
        """
        if not self.N:
            return []
        candidates = set()
        q_terms_list = []
        for q in queries:
            q_terms = _tokenize(q)
            q_terms_list.append(q_terms)
            for t in set(q_terms):
                postings = self.tf.get(t)
                if postings:
                    candidates.update(postings.keys())

        scores: Dict[int, float] = {}
        for doc_idx in candidates:
            s_list = [self._score_doc(q_terms, doc_idx) for q_terms in q_terms_list]
            s = sum(s_list) if alpha_merge == "sum" else (max(s_list) if s_list else 0.0)
            if s > 0:
                scores[doc_idx] = s

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return ranked

    def doc_meta(self, doc_idx: int) -> Dict[str, Any]:
        return dict(self.docs[doc_idx]) if 0 <= doc_idx < len(self.docs) else {}
