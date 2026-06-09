"""BM25 + 倒排索引构建"""
import json
import pickle
import re
from collections import defaultdict

import jieba
from rank_bm25 import BM25Okapi

from config.settings import INDEX_DIR, PARSED_DIR
from offline.chunker import Chunk
from utils.logger import logger

# 停用词表
STOP_WORDS = set("的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 "
                 "说 要 去 你 会 着 没有 看 好 自己 这 那 她 他 它 们 "
                 "为 与 及 或 而 但 之 以 于 从 对 等 将 被 把 向 比 "
                 "所 如 果 能 可 并 且 该 此 其 更 最 已 经 还 又 "
                 "中 时 后 前 年 月 日 第 条 章 节 款 项 "
                 "应 当 须 需 得 则 即 若 如".split())


def build_financial_dict(parsed_docs: list[dict], top_n: int = 200) -> list[str]:
    """从文档中自动抽取高频金融术语，构建自定义词典"""
    # 候选术语：2-8字的中文词
    term_freq: dict[str, int] = defaultdict(int)
    pattern = re.compile(r"[\u4e00-\u9fa5]{2,8}")

    for doc in parsed_docs:
        text = doc.get("raw_text", "")[:20000]  # 取前2万字
        # 用 jieba 粗分
        words = jieba.lcut(text)
        for w in words:
            if len(w) >= 2 and w not in STOP_WORDS:
                term_freq[w] += 1

    # 按频率排序取 top_n
    sorted_terms = sorted(term_freq.items(), key=lambda x: -x[1])[:top_n]
    custom_words = [term for term, freq in sorted_terms if freq >= 3]

    # 添加到 jieba 词典
    for word in custom_words:
        jieba.add_word(word)

    logger.info(f"金融自定义词典: {len(custom_words)} 个词")
    return custom_words


def tokenize(text: str) -> list[str]:
    """分词：jieba + 去停用词"""
    words = jieba.lcut(text)
    return [w.strip() for w in words if w.strip() and w.strip() not in STOP_WORDS and len(w.strip()) >= 1]


class SearchIndex:
    """搜索索引：BM25 + 倒排索引"""

    def __init__(self):
        # 按文档类型分组的 BM25 索引
        self.bm25_l3: dict[str, BM25Okapi] = {}  # L3 chunk 的 BM25
        self.bm25_l2: dict[str, BM25Okapi] = {}  # L2 chunk 的 BM25
        self.bm25_global: BM25Okapi | None = None  # 全局 BM25 (B榜用)

        # chunk 存储
        self.chunks_by_id: dict[str, Chunk] = {}
        self.chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        self.chunks_l3_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        self.chunks_l2_by_doc: dict[str, list[Chunk]] = defaultdict(list)

        # 全局 L3 chunks (B榜用)
        self.all_l3_chunks: list[Chunk] = []
        self.all_l3_tokenized: list[list[str]] = []

        # 倒排索引
        self.inverted_index: dict[str, set[str]] = defaultdict(set)

        # 文档元数据
        self.doc_meta: dict[str, dict] = {}

    def build(self, all_chunks: dict[str, list[Chunk]], doc_meta: dict[str, dict]):
        """构建索引"""
        self.doc_meta = doc_meta

        # 收集所有 chunks
        for doc_id, chunks in all_chunks.items():
            for chunk in chunks:
                self.chunks_by_id[chunk.chunk_id] = chunk
                self.chunks_by_doc[doc_id].append(chunk)

                if chunk.level == "L3":
                    self.chunks_l3_by_doc[doc_id].append(chunk)
                elif chunk.level == "L2":
                    self.chunks_l2_by_doc[doc_id].append(chunk)

        # 按文档类型构建 BM25
        doc_types = set(c.doc_type for chunks in all_chunks.values() for c in chunks)
        for doc_type in doc_types:
            # L3 BM25
            l3_chunks = [c for chunks in all_chunks.values()
                         for c in chunks if c.doc_type == doc_type and c.level == "L3"]
            if l3_chunks:
                tokenized = [tokenize(c.text) for c in l3_chunks]
                self.bm25_l3[doc_type] = BM25Okapi(tokenized)
                # 存储引用以便检索时映射
                self.bm25_l3[f"{doc_type}_chunks"] = l3_chunks  # type: ignore

        # 全局 L3 BM25 (B榜用)
        self.all_l3_chunks = [c for chunks in all_chunks.values()
                               for c in chunks if c.level == "L3"]
        self.all_l3_tokenized = [tokenize(c.text) for c in self.all_l3_chunks]
        if self.all_l3_tokenized:
            self.bm25_global = BM25Okapi(self.all_l3_tokenized)

        # 构建倒排索引
        for chunk_id, chunk in self.chunks_by_id.items():
            if chunk.level in ("L3", "L4"):
                words = tokenize(chunk.text)
                for word in set(words):
                    self.inverted_index[word].add(chunk_id)

        logger.info(
            f"索引构建完成: {len(self.chunks_by_id)} chunks, "
            f"{len(self.all_l3_chunks)} L3 chunks, "
            f"{len(self.inverted_index)} 个词项"
        )

    def search_bm25(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int = 20,
        level: str = "L3",
    ) -> list[tuple[Chunk, float]]:
        """BM25 检索

        Args:
            query: 查询文本
            doc_ids: 限定文档范围 (None 表示全局搜索)
            top_k: 返回数量
            level: 检索层级 L2/L3
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        if doc_ids is not None:
            # 限定文档范围检索
            return self._search_in_docs(query_tokens, doc_ids, top_k, level)
        else:
            # 全局检索
            return self._search_global(query_tokens, top_k)

    def _search_in_docs(
        self,
        query_tokens: list[str],
        doc_ids: list[str],
        top_k: int,
        level: str,
    ) -> list[tuple[Chunk, float]]:
        """在指定文档中检索"""
        candidates = []
        for doc_id in doc_ids:
            if level == "L3":
                chunks = self.chunks_l3_by_doc.get(doc_id, [])
            elif level == "L2":
                chunks = self.chunks_l2_by_doc.get(doc_id, [])
            else:
                chunks = self.chunks_by_doc.get(doc_id, [])

            if not chunks:
                continue

            tokenized = [tokenize(c.text) for c in chunks]
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query_tokens)

            for chunk, score in zip(chunks, scores):
                if score > 0:
                    candidates.append((chunk, float(score)))

        # 按分数排序
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def _search_global(
        self,
        query_tokens: list[str],
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """全局检索 (B榜用)"""
        if not self.bm25_global or not self.all_l3_chunks:
            return []

        scores = self.bm25_global.get_scores(query_tokens)
        indexed = [(self.all_l3_chunks[i], float(scores[i]))
                   for i in range(len(scores)) if scores[i] > 0]
        indexed.sort(key=lambda x: -x[1])
        return indexed[:top_k]

    def keyword_search(
        self,
        keywords: list[str],
        doc_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[tuple[Chunk, float]]:
        """关键词倒排索引检索"""
        matched_chunks: dict[str, int] = defaultdict(int)

        for kw in keywords:
            # 精确匹配
            if kw in self.inverted_index:
                for chunk_id in self.inverted_index[kw]:
                    if doc_ids is None or self.chunks_by_id[chunk_id].doc_id in doc_ids:
                        matched_chunks[chunk_id] += 1

            # 子串匹配 (处理分词差异)
            for term, chunk_ids in self.inverted_index.items():
                if kw in term or term in kw:
                    for chunk_id in chunk_ids:
                        if doc_ids is None or self.chunks_by_id[chunk_id].doc_id in doc_ids:
                            matched_chunks[chunk_id] += 0.5

        results = [
            (self.chunks_by_id[cid], float(score))
            for cid, score in matched_chunks.items()
            if score > 0
        ]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def locate_documents(
        self,
        query: str,
        doc_summaries: dict[str, dict],
        top_n: int = 5,
    ) -> list[str]:
        """B榜文档定位：根据查询找到最相关的 top-N 文档"""
        query_tokens = tokenize(query)
        doc_scores: dict[str, float] = defaultdict(float)

        # 方法1: 关键词匹配文档摘要
        for doc_id, summary_data in doc_summaries.items():
            keywords = summary_data.get("keywords", [])
            l1_summary = summary_data.get("l1_summary", "")

            # 关键词交集
            overlap = sum(1 for qt in query_tokens if qt in keywords)
            doc_scores[doc_id] += overlap * 2

            # L1 摘要 BM25 分数
            summary_tokens = tokenize(l1_summary)
            overlap2 = sum(1 for qt in query_tokens if qt in set(summary_tokens))
            doc_scores[doc_id] += overlap2

        # 方法2: 元数据匹配
        meta = self.doc_meta
        for doc_id, meta_data in meta.items():
            companies = meta_data.get("companies", [])
            for company in companies:
                if company in query:
                    doc_scores[doc_id] += 5

        # 排序返回 top-N
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        return [doc_id for doc_id, score in sorted_docs[:top_n] if score > 0]

    def save(self, path: str | None = None):
        """持久化索引"""
        path = path or str(INDEX_DIR / "search_index.pkl")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "chunks_by_id": {k: v.to_dict() for k, v in self.chunks_by_id.items()},
            "inverted_index": {k: list(v) for k, v in self.inverted_index.items()},
            "doc_meta": self.doc_meta,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"索引已保存: {path}")

    def load(self, path: str | None = None):
        """加载索引"""
        path = path or str(INDEX_DIR / "search_index.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)

        # 重建 chunks
        for cid, cdata in data["chunks_by_id"].items():
            chunk = Chunk(
                chunk_id=cdata["chunk_id"],
                doc_id=cdata["doc_id"],
                doc_type=cdata["doc_type"],
                level=cdata["level"],
                section_path=cdata["section_path"],
                text=cdata["text"],
                metadata=cdata.get("metadata", {}),
            )
            self.chunks_by_id[cid] = chunk
            self.chunks_by_doc[chunk.doc_id].append(chunk)
            if chunk.level == "L3":
                self.chunks_l3_by_doc[chunk.doc_id].append(chunk)
                self.all_l3_chunks.append(chunk)
            elif chunk.level == "L2":
                self.chunks_l2_by_doc[chunk.doc_id].append(chunk)

        # 重建倒排索引
        for term, cids in data["inverted_index"].items():
            self.inverted_index[term] = set(cids)

        self.doc_meta = data["doc_meta"]

        # 重建 BM25
        self.all_l3_tokenized = [tokenize(c.text) for c in self.all_l3_chunks]
        if self.all_l3_tokenized:
            self.bm25_global = BM25Okapi(self.all_l3_tokenized)

        logger.info(f"索引加载完成: {len(self.chunks_by_id)} chunks")
