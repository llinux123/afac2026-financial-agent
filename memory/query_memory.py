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
        self._keyword_index: dict[str, set[str]] = {}  # 关键词 -> qid 集合 (倒排索引)
        self._hit_window: list[bool] = []  # 最近查询命中记录
        self._dynamic_threshold: float = 0.6  # 动态阈值（初始=0.6）

    def lookup(self, question: str, keywords: list[str]) -> dict | None:
        """查找相似问题的缓存（倒排索引加速 + BM25+Jaccard + 动态阈值 + LRU）"""
        if not self.cache or not keywords:
            return None

        # 第一步：通过倒排索引获取候选集
        candidates = self._get_candidates_from_index(keywords)

        # 如果倒排索引无候选，降级为全量查询
        if not candidates:
            candidates = set(self.cache.keys())

        # 第二步：仅在候选集内做 BM25+Jaccard 评分
        if len(candidates) == len(self.cache):
            # 全量模式：用原有 BM25 索引
            self._ensure_bm25_index()
            cache_keys = list(self.cache.keys())

            jaccard_scores = np.array([
                self._jaccard(keywords, self.cache[qid].get("keywords", []))
                for qid in cache_keys
            ])

            if self._bm25_index is not None:
                bm25_scores = self._bm25_index.get_scores(keywords)
                max_bm25 = bm25_scores.max()
                if max_bm25 > 0:
                    bm25_norm = bm25_scores / max_bm25
                    combined = 0.6 * bm25_norm + 0.4 * jaccard_scores
                else:
                    combined = jaccard_scores
            else:
                combined = jaccard_scores
            best_idx = int(np.argmax(combined))
            best_score = float(combined[best_idx])
            best_qid = cache_keys[best_idx]
        else:
            # 候选模式：仅在候选集内计算 Jaccard（避免重建BM25）
            best_score = 0.0
            best_qid = None
            for qid in candidates:
                entry = self.cache.get(qid)
                if not entry:
                    continue
                score = self._jaccard(keywords, entry.get("keywords", []))
                if score > best_score:
                    best_score = score
                    best_qid = qid

        # 第三步：动态阈值判断
        if best_qid and best_score >= self._dynamic_threshold:
            self.cache.move_to_end(best_qid)
            self._record_hit(True)
            logger.debug(f"  缓存命中: {best_qid} (score={best_score:.2f}, threshold={self._dynamic_threshold:.2f})")
            return self.cache[best_qid]

        self._record_hit(False)
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

        # 更新倒排索引
        for kw in keywords:
            if kw not in self._keyword_index:
                self._keyword_index[kw] = set()
            self._keyword_index[kw].add(qid)

        # LRU 淘汰
        if len(self.cache) > self.max_cache_size:
            evicted_key, evicted_entry = self.cache.popitem(last=False)
            # 清理倒排索引
            for kw in evicted_entry.get("keywords", []):
                if kw in self._keyword_index:
                    self._keyword_index[kw].discard(evicted_key)
                    if not self._keyword_index[kw]:
                        del self._keyword_index[kw]
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

    def _get_candidates_from_index(self, keywords: list[str]) -> set[str]:
        """通过倒排索引获取候选 qid 集合"""
        candidates = set()
        # 取前5个关键词查找
        for kw in keywords[:5]:
            if kw in self._keyword_index:
                candidates |= self._keyword_index[kw]
        # 过滤已淘汰的 qid
        candidates &= set(self.cache.keys())
        return candidates

    def _record_hit(self, hit: bool):
        """记录命中情况并动态调整阈值"""
        self._hit_window.append(hit)
        # 保持窗口大小 <= 100
        if len(self._hit_window) > 100:
            self._hit_window = self._hit_window[-100:]
        # 每20次查询调整一次阈值
        if len(self._hit_window) % 20 == 0:
            self._update_threshold()

    def _update_threshold(self):
        """根据命中率动态调整阈值"""
        if len(self._hit_window) < 20:
            return
        recent = self._hit_window[-50:]  # 看最近50次
        hit_rate = sum(recent) / len(recent)

        if hit_rate < 0.2:
            # 命中率太低，降低阈值
            self._dynamic_threshold = max(0.4, self._dynamic_threshold - 0.05)
        elif hit_rate > 0.6:
            # 命中率很高，提高阈值保证质量
            self._dynamic_threshold = min(0.8, self._dynamic_threshold + 0.05)

        logger.debug(f"  动态阈值调整: {self._dynamic_threshold:.2f} (hit_rate={hit_rate:.2f})")

    def _jaccard(self, list1: list[str], list2: list[str]) -> float:
        set1 = set(list1)
        set2 = set(list2)
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
