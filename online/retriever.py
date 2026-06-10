"""混合检索引擎 - BM25 + 关键词 + 选项增强检索"""
from dataclasses import dataclass

from config.settings import BM25_TOP_K, DOC_LOCATE_TOP_N
from offline.chunker import Chunk
from offline.index_builder import SearchIndex
from online.question_analyzer import AnalyzedQuestion
from utils.logger import logger


@dataclass
class RetrievalResult:
    """检索结果"""
    chunks: list[tuple[Chunk, float]]  # (chunk, score) 列表
    located_doc_ids: list[str]          # 定位到的文档 (B榜)
    search_method: str                  # 使用的检索方法描述


class Retriever:
    """混合检索引擎"""

    def __init__(self, index: SearchIndex, doc_summaries: dict[str, dict] | None = None):
        self.index = index
        self.doc_summaries = doc_summaries or {}

    def retrieve(
        self,
        question: AnalyzedQuestion,
        top_k: int = BM25_TOP_K,
    ) -> RetrievalResult:
        """检索与题目相关的文档块"""
        if question.doc_ids:
            # A榜: 有明确的 doc_ids
            return self._retrieve_a_board(question, top_k)
        else:
            # B榜: 需要先定位文档
            return self._retrieve_b_board(question, top_k)

    def _retrieve_a_board(
        self,
        question: AnalyzedQuestion,
        top_k: int,
    ) -> RetrievalResult:
        """A榜检索策略: 在已知文档中精细检索"""
        doc_ids = question.doc_ids
        all_results: dict[str, tuple[Chunk, float]] = {}

        # 1. 用完整问题做 BM25 检索
        for query in question.search_queries[:3]:
            results = self.index.search_bm25(
                query=query,
                doc_ids=doc_ids,
                top_k=top_k,
                level="L3",
            )
            for chunk, score in results:
                cid = chunk.chunk_id
                if cid not in all_results or all_results[cid][1] < score:
                    all_results[cid] = (chunk, score)

        # 2. 选项增强检索: 每个选项单独检索
        for opt_key, opt_keywords in question.option_keywords.items():
            if opt_keywords:
                opt_query = " ".join(opt_keywords[:5])
                results = self.index.search_bm25(
                    query=opt_query,
                    doc_ids=doc_ids,
                    top_k=5,
                    level="L3",
                )
                for chunk, score in results:
                    cid = chunk.chunk_id
                    adjusted_score = score * 0.8  # 选项检索权重稍低
                    if cid not in all_results or all_results[cid][1] < adjusted_score:
                        all_results[cid] = (chunk, adjusted_score)

        # 3. 关键词倒排索引补充
        kw_results = self.index.keyword_search(
            keywords=question.keywords[:8],
            doc_ids=doc_ids,
            top_k=10,
        )
        for chunk, score in kw_results:
            cid = chunk.chunk_id
            adjusted_score = score * 1.5
            if cid not in all_results or all_results[cid][1] < adjusted_score:
                all_results[cid] = (chunk, adjusted_score)

        # 排序并取 top-K
        sorted_results = sorted(all_results.values(), key=lambda x: -x[1])
        return RetrievalResult(
            chunks=sorted_results[:top_k],
            located_doc_ids=doc_ids,
            search_method="A榜-BM25+选项增强+关键词",
        )

    def _retrieve_b_board(
        self,
        question: AnalyzedQuestion,
        top_k: int,
    ) -> RetrievalResult:
        """B榜检索策略: 先定位文档，再精细检索"""
        # 第一轮: 文档定位
        located_docs = self.index.locate_documents(
            query=question.question,
            doc_summaries=self.doc_summaries,
            top_n=DOC_LOCATE_TOP_N,
        )

        if not located_docs:
            # 降级: 全局 BM25 检索
            logger.warning(f"  B榜文档定位失败 [{question.qid}]，使用全局检索")
            return self._global_fallback(question, top_k)

        logger.debug(f"  B榜定位文档: {located_docs}")

        # 第二轮: 在定位文档中精细检索 (复用 A榜策略)
        # 临时设置 doc_ids 以复用 A榜逻辑
        original_doc_ids = question.doc_ids
        question.doc_ids = located_docs
        result = self._retrieve_a_board(question, top_k)
        question.doc_ids = original_doc_ids

        result.located_doc_ids = located_docs
        result.search_method = f"B榜-文档定位({len(located_docs)}篇)+BM25+选项增强"

        return result

    def _global_fallback(
        self,
        question: AnalyzedQuestion,
        top_k: int,
    ) -> RetrievalResult:
        """全局降级检索"""
        all_results: dict[str, tuple[Chunk, float]] = {}

        for query in question.search_queries[:3]:
            results = self.index.search_bm25(
                query=query,
                doc_ids=None,
                top_k=top_k,
                level="L3",
            )
            for chunk, score in results:
                cid = chunk.chunk_id
                if cid not in all_results or all_results[cid][1] < score:
                    all_results[cid] = (chunk, score)

        sorted_results = sorted(all_results.values(), key=lambda x: -x[1])

        # 收集命中的文档 ID
        hit_docs = list(set(c.doc_id for c, _ in sorted_results[:top_k]))

        return RetrievalResult(
            chunks=sorted_results[:top_k],
            located_doc_ids=hit_docs,
            search_method="全局降级BM25",
        )

    def retrieve_for_option(
        self,
        question: AnalyzedQuestion,
        option_key: str,
        option_text: str,
        doc_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """针对单个选项的定向检索 (用于补充检索)"""
        query = f"{question.question} {option_text}"
        return self.index.search_bm25(
            query=query,
            doc_ids=doc_ids or question.doc_ids or None,
            top_k=top_k,
            level="L3",
        )
