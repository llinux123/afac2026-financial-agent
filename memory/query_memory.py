"""题目级记忆 - 证据缓存与复用"""
from collections import OrderedDict
from config.settings import QUERY_CACHE_SIMILARITY, FACT_TABLE_MAX_SIZE
from utils.logger import logger


class QueryMemory:
    """题目级记忆管理器"""

    def __init__(self):
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.fact_table: dict[str, str] = {}  # 跨题目事实表
        self.max_cache_size = 200

    def lookup(self, question: str, keywords: list[str]) -> dict | None:
        """查找相似问题的缓存（真正的 LRU）"""
        hit_qid = None
        hit_entry = None
        for qid, entry in self.cache.items():
            cached_keywords = entry.get("keywords", [])
            similarity = self._jaccard(keywords, cached_keywords)
            if similarity >= QUERY_CACHE_SIMILARITY:
                hit_qid = qid
                hit_entry = entry
                break

        if hit_qid:
            # LRU: 命中后移到末尾，淘汰时优先淘汰最久未使用的
            self.cache.move_to_end(hit_qid)
            logger.debug(f"  缓存命中: {hit_qid} (similarity={similarity:.2f})")
            return hit_entry
        return None

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
