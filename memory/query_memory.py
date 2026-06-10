"""题目级记忆 - 证据缓存与复用"""
from collections import OrderedDict

import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import FACT_TABLE_MAX_SIZE
from utils.logger import logger


class QueryMemory:
    """题目级记忆管理器"""

    def __init__(self):
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.fact_table: dict[str, str] = {}  # 跨题目事实表
        self.max_cache_size = 200
        self._bm25_index: BM25Okapi | None = None
        self._bm25_dirty: bool = True  # 标记索引是否需要重建

    def lookup(self, question: str, keywords: list[str]) -> dict | None:
        """查找相似问题的缓存（BM25 + Jaccard 混合评分 + LRU）"""
        if not self.cache or not keywords:
            return None

        self._ensure_bm25_index()

        cache_keys = list(self.cache.keys())

        # BM25 评分
        if self._bm25_index is not None:
            bm25_scores = self._bm25_index.get_scores(keywords)
            max_bm25 = bm25_scores.max()
            bm25_norm = bm25_scores / max_bm25 if max_bm25 > 0 else bm25_scores
        else:
            bm25_norm = np.zeros(len(cache_keys))

        # Jaccard 评分
        jaccard_scores = np.array([
            self._jaccard(keywords, self.cache[qid].get("keywords", []))
            for qid in cache_keys
        ])

        # 综合评分: 0.6 * BM25 + 0.4 * Jaccard
        combined = 0.6 * bm25_norm + 0.4 * jaccard_scores
        best_idx = int(np.argmax(combined))
        best_score = float(combined[best_idx])

        # 动态阈值：0.6
        if best_score >= 0.6:
            hit_qid = cache_keys[best_idx]
            self.cache.move_to_end(hit_qid)  # LRU
            logger.debug(f"  缓存命中: {hit_qid} (score={best_score:.2f})")
            return self.cache[hit_qid]

        return None

    def _ensure_bm25_index(self):
        """确保 BM25 索引是最新的"""
        if self._bm25_dirty:
            corpus = [entry.get("keywords", []) for entry in self.cache.values()]
            # BM25Okapi 需要非空语料
            if corpus and any(len(kws) > 0 for kws in corpus):
                try:
                    self._bm25_index = BM25Okapi(corpus)
                except Exception:
                    self._bm25_index = None
            else:
                self._bm25_index = None
            self._bm25_dirty = False

    def store(
        self,
        qid: str,
        question: str,
        keywords: list[str],
        evidence: str,
        answer: str,
        token_cost: int,
    ):
        """存储题目记忆"""
        self.cache[qid] = {
            "question": question,
            "keywords": keywords,
            "evidence": evidence,
            "answer": answer,
            "token_cost": token_cost,
        }

        # LRU 淘汰
        if len(self.cache) > self.max_cache_size:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.debug(f"  缓存淘汰: {evicted_key}")

        self._bm25_dirty = True  # 缓存变更，需要重建 BM25 索引

    def add_fact(self, key: str, value: str):
        """添加跨题目事实"""
        if len(self.fact_table) >= FACT_TABLE_MAX_SIZE:
            # 简单 FIFO 淘汰
            oldest_key = next(iter(self.fact_table))
            del self.fact_table[oldest_key]
        self.fact_table[key] = value

    def lookup_fact(self, key: str) -> str | None:
        """查找事实"""
        return self.fact_table.get(key)

    def _jaccard(self, list1: list[str], list2: list[str]) -> float:
        set1 = set(list1)
        set2 = set(list2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
