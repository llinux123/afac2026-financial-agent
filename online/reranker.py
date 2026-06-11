"""Qwen 语义重排序 - 条件触发"""
from config.settings import RERANK_TOP_K, RERANK_TRIGGER_THRESHOLD
from config.prompts import RERANK_PROMPT
from offline.chunker import Chunk
from utils.api_client import QwenClient
from utils.logger import logger


class Reranker:
    """语义重排序器"""

    def __init__(self, client: QwenClient):
        self.client = client

    def should_rerank(self, scored_chunks: list[tuple[Chunk, float]]) -> bool:
        """判断是否需要重排序

        当 BM25 结果分数差距小时触发（top-1 与 top-5 分差 < 阈值）
        """
        if len(scored_chunks) < 5:
            return False

        top1_score = scored_chunks[0][1]
        top5_score = scored_chunks[4][1]

        if top5_score == 0:
            return False

        score_ratio = (top1_score - top5_score) / top1_score if top1_score > 0 else 0
        return score_ratio < RERANK_TRIGGER_THRESHOLD

    def rerank(
        self,
        question: str,
        chunks: list[tuple[Chunk, float]],
        top_k: int = RERANK_TOP_K,
    ) -> tuple[list[tuple[Chunk, float]], int]:
        """执行语义重排序

        Returns:
            (重排序后的 chunks, 消耗的 token 数)
        """
        # 只取 BM25 top-10 进行重排序
        candidates = chunks[:10]
        if len(candidates) <= 2:
            return chunks, 0

        # 记录重排前顺序
        before_detail = ", ".join(
            f"{i+1}.{chunk.chunk_id}(score={score:.3f})"
            for i, (chunk, score) in enumerate(candidates[:5])
        )
        logger.info(f"  [重排序] 重排前top-{len(candidates)}: {before_detail}{' ...' if len(candidates) > 5 else ''}")

        # 构建 passages 文本 (截断到 300 字以节省 Token)
        passages_text = ""
        for i, (chunk, score) in enumerate(candidates):
            truncated = chunk.text[:300]
            passages_text += f"{i + 1}. [{chunk.section_path}] {truncated}\n"

        prompt = RERANK_PROMPT.format(
            question=question,
            passages=passages_text,
        )

        response = self.client.simple_chat(
            prompt=prompt,
            temperature=0.0,
            max_tokens=200,
        )

        # 解析排序结果
        reranked = self._parse_rerank_result(response.content, candidates)

        if reranked:
            after_detail = ", ".join(
                f"{i+1}.{chunk.chunk_id}(score={score:.3f})"
                for i, (chunk, score) in enumerate(reranked[:5])
            )
            logger.info(
                f"  [重排序] 重排后top-{len(reranked)}: {after_detail}{' ...' if len(reranked) > 5 else ''}, "
                f"token={response.total_tokens}"
            )
            return reranked[:top_k], response.total_tokens
        else:
            logger.warning(
                f"  [重排序] 解析失败，保持原序, raw={response.content.strip()!r}, "
                f"token={response.total_tokens}"
            )
            return chunks[:top_k], response.total_tokens

    def _parse_rerank_result(
        self,
        result_text: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """解析重排序结果"""
        import re
        # 提取数字序号
        numbers = re.findall(r"\d+", result_text)
        if not numbers:
            return []

        indices = []
        for n in numbers:
            idx = int(n) - 1  # 转为 0-based
            if 0 <= idx < len(candidates) and idx not in indices:
                indices.append(idx)

        if not indices:
            return []

        # 按重排序结果排列，分数改为排名分
        result = []
        for rank, idx in enumerate(indices):
            chunk, _ = candidates[idx]
            # 用排名分数替代原始 BM25 分数
            new_score = len(candidates) - rank
            result.append((chunk, float(new_score)))

        return result
