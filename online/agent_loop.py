"""主控 Agent 循环 - 串行逐题处理"""
from typing import Callable
from online.question_analyzer import AnalyzedQuestion, analyze_question
from online.retriever import Retriever, RetrievalResult
from online.reranker import Reranker
from online.evidence_compressor import EvidenceCompressor
from online.reasoner import Reasoner
from online.answer_formatter import extract_answer
from offline.index_builder import SearchIndex
from memory.doc_memory import DocMemory
from memory.query_memory import QueryMemory
from memory.memory_compressor import MemoryCompressor
from utils.api_client import QwenClient
from utils.token_counter import TokenBudget, BudgetConfig
from utils.logger import logger


class AgentLoop:
    """主控 Agent"""

    def __init__(
        self,
        client: QwenClient,
        index: SearchIndex,
        doc_memory: DocMemory,
        token_budget: TokenBudget,
    ):
        self.client = client
        self.retriever = Retriever(index, doc_memory.get_all_summaries())
        self.reranker = Reranker(client)
        self.compressor = EvidenceCompressor(client)
        self.reasoner = Reasoner(client)
        self.query_memory = QueryMemory()
        self.mem_compressor = MemoryCompressor()
        self.token_budget = token_budget

    def process_question(
        self,
        question_data: dict,
        question_idx: int,
        total_questions: int,
    ) -> dict:
        """处理单道题目，返回结果"""
        # 1. 题目分析 (0 Token)
        analyzed = analyze_question(question_data)
        qid = analyzed.qid
        logger.info(f"[{question_idx + 1}/{total_questions}] {qid} "
                    f"type={analyzed.question_type}, domain={analyzed.domain}")

        # 2. 查缓存
        cached = self.query_memory.lookup(analyzed.question, analyzed.keywords)
        if cached:
            logger.info(f"  缓存命中，跳过检索和推理")
            return {
                "qid": qid,
                "answer": cached["answer"],
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "method": "cache",
            }

        # 3. 获取预算
        budget = self.token_budget.get_budget(question_idx, total_questions)

        # 4. 检索 (0 Token)
        retrieval = self.retriever.retrieve(analyzed)
        chunks = retrieval.chunks
        logger.info(f"  [Agent] 检索完成: {len(chunks)} chunks, 方法={retrieval.search_method}, 定位文档={retrieval.located_doc_ids}")

        if not chunks:
            logger.warning(f"  检索无结果 [{qid}]")
            return {
                "qid": qid,
                "answer": "A",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "method": "no_evidence",
            }

        # 5. 重排序 (条件触发)
        total_api_tokens = 0
        if self.reranker.should_rerank(chunks):
            logger.info(f"  [Agent] 触发重排序: top1与top5分数接近")
            chunks, rerank_tokens = self.reranker.rerank(
                analyzed.question, chunks,
            )
            total_api_tokens += rerank_tokens
        else:
            logger.info(f"  [Agent] 跳过重排序: 分数差距足够大")

        # 6. 证据压缩
        evidence, compress_tokens = self.compressor.compress(
            chunks=chunks,
            question=analyzed.question,
            keywords=analyzed.keywords,
            max_tokens=budget.max_evidence_tokens,
        )
        total_api_tokens += compress_tokens

        # 7. 推理
        enable_thinking = (
            analyzed.question_type == "multi" and budget.mode == "normal"
        )
        model_output, response = self.reasoner.reason(
            analyzed, evidence, enable_thinking=enable_thinking,
        )
        total_api_tokens += response.total_tokens

        # 8. 答案提取
        answer = extract_answer(model_output, analyzed)
        logger.info(f"  答案: {answer} (tokens={total_api_tokens:,})")

        # 9. 记忆更新
        self.query_memory.store(
            qid=qid,
            question=analyzed.question,
            keywords=analyzed.keywords,
            evidence=self.mem_compressor.compress_evidence(evidence),
            answer=answer,
            token_cost=total_api_tokens,
        )

        # 提取事实
        facts = self.mem_compressor.extract_facts(
            analyzed.question, answer, evidence,
        )
        for key, value in facts:
            self.query_memory.add_fact(key, value)

        # 10. Token 预算记录
        self.token_budget.record(
            qid, response.prompt_tokens, response.completion_tokens,
        )

        return {
            "qid": qid,
            "answer": answer,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "method": retrieval.search_method,
            "model_output": model_output,
        }

    def process_all(
        self,
        questions: list[dict],
        on_question_done: Callable | None = None,
    ) -> list[dict]:
        """处理所有题目

        Args:
            questions: 题目列表。
            on_question_done: 每处理完一道题目后的回调函数，接收 result dict。
        """
        results = []
        total = len(questions)

        for i, q in enumerate(questions):
            try:
                result = self.process_question(q, i, total)
                results.append(result)
            except Exception as e:
                qid = q.get("qid", f"q_{i}")
                logger.error(f"处理失败 [{qid}]: {e}")
                result = {
                    "qid": qid,
                    "answer": "A",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "method": "error",
                    "error": str(e),
                }
                results.append(result)

            # 逐题回调：持久化
            if on_question_done is not None:
                try:
                    on_question_done(result)
                except Exception as cb_err:
                    logger.warning(f"逐题回调失败 [{result.get('qid', '?')}]: {cb_err}")

        return results
