"""证据动态压缩模块"""
import re

from config.settings import (
    DEDUP_SIMILARITY_THRESHOLD,
    SENTENCE_RELEVANCE_THRESHOLD,
    LONG_CHUNK_THRESHOLD,
)
from config.prompts import SUMMARIZE_EVIDENCE
from offline.chunker import Chunk
from utils.api_client import QwenClient
from utils.token_counter import estimate_tokens, truncate_to_tokens
from utils.logger import logger


class EvidenceCompressor:
    """证据动态压缩器"""

    def __init__(self, client: QwenClient):
        self.client = client

    def compress(
        self,
        chunks: list[tuple[Chunk, float]],
        question: str,
        keywords: list[str],
        max_tokens: int = 15000,
    ) -> tuple[str, int]:
        """压缩证据到指定 Token 预算内

        Returns:
            (压缩后的证据文本, 消耗的 API token 数)
        """
        total_api_tokens = 0

        # 阶段 1: 去重
        deduped = self._deduplicate(chunks)
        logger.debug(f"  压缩: 去重 {len(chunks)}→{len(deduped)}")

        # 阶段 2: 关键句子提取
        filtered = self._extract_relevant_sentences(deduped, question, keywords)
        logger.debug(f"  压缩: 句子提取 {len(deduped)}→{len(filtered)}")

        # 阶段 3: 检查是否超长，需要动态摘要
        evidence_text = self._format_evidence(filtered)
        current_tokens = estimate_tokens(evidence_text)

        if current_tokens > max_tokens:
            # 对长 chunk 做摘要压缩
            evidence_text, api_tokens = self._summarize_long_chunks(
                filtered, question, max_tokens,
            )
            total_api_tokens += api_tokens

        # 最终截断保护
        final_tokens = estimate_tokens(evidence_text)
        if final_tokens > max_tokens:
            evidence_text = truncate_to_tokens(evidence_text, max_tokens)

        logger.debug(
            f"  压缩完成: {estimate_tokens(evidence_text):,} tokens "
            f"(预算={max_tokens:,})"
        )
        return evidence_text, total_api_tokens

    def _deduplicate(
        self, chunks: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        """Jaccard 去重"""
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for chunk, score in chunks[1:]:
            is_dup = False
            for existing_chunk, _ in result:
                if self._jaccard_similarity(chunk.text, existing_chunk.text) > DEDUP_SIMILARITY_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                result.append((chunk, score))

        return result

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """字符 n-gram Jaccard 相似度"""
        n = 3
        if len(text1) < n or len(text2) < n:
            return 0.0

        def ngrams(text: str) -> set[str]:
            return {text[i:i + n] for i in range(len(text) - n + 1)}

        set1 = ngrams(text1)
        set2 = ngrams(text2)
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    def _extract_relevant_sentences(
        self,
        chunks: list[tuple[Chunk, float]],
        question: str,
        keywords: list[str],
    ) -> list[tuple[Chunk, str, float]]:
        """提取与问题相关的句子

        Returns:
            [(chunk, filtered_text, score), ...]
        """
        result = []
        keyword_set = set(keywords)

        for chunk, chunk_score in chunks:
            sentences = self._split_sentences(chunk.text)
            if not sentences:
                continue

            # 计算每个句子的相关性分数
            scored_sentences = []
            for i, sent in enumerate(sentences):
                score = self._sentence_relevance(sent, keyword_set, question)

                # 包含数字/日期/金额的句子加分
                if re.search(r"\d+\.?\d*\s*(%|万|亿|元|年|月|日|天)", sent):
                    score += 0.2

                scored_sentences.append((i, sent, score))

            # 保留高相关句子 + 前后各 1 句上下文
            selected = set()
            for i, sent, score in scored_sentences:
                if score >= SENTENCE_RELEVANCE_THRESHOLD:
                    selected.add(i)
                    if i > 0:
                        selected.add(i - 1)
                    if i < len(sentences) - 1:
                        selected.add(i + 1)

            # 如果没有高相关句子，保留前 3 个得分最高的
            if not selected:
                scored_sentences.sort(key=lambda x: -x[2])
                for i, _, _ in scored_sentences[:3]:
                    selected.add(i)

            # 按原文顺序拼接
            filtered_parts = [
                sentences[i] for i in sorted(selected)
            ]
            filtered_text = "\n".join(filtered_parts)

            if filtered_text.strip():
                result.append((chunk, filtered_text, chunk_score))

        return result

    def _sentence_relevance(
        self, sentence: str, keywords: set[str], question: str,
    ) -> float:
        """计算句子相关性分数"""
        if not sentence.strip():
            return 0.0

        score = 0.0
        sent_len = max(len(sentence), 1)

        # 关键词命中率
        for kw in keywords:
            if kw in sentence:
                score += 1.0

        # 问题中的词命中率
        question_words = set(re.findall(r"[\u4e00-\u9fa5]{2,}", question))
        for word in question_words:
            if word in sentence and word not in {"正确", "错误", "以下", "哪项"}:
                score += 0.3

        # 归一化
        return score / max(len(keywords), 1)

    def _summarize_long_chunks(
        self,
        chunks: list[tuple[Chunk, str, float]],
        question: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """对过长的 chunk 调用 Qwen 生成摘要"""
        total_api_tokens = 0
        summarized = []

        for chunk, filtered_text, score in chunks:
            text_tokens = estimate_tokens(filtered_text)
            if text_tokens <= LONG_CHUNK_THRESHOLD:
                # 不长，直接用
                summarized.append((chunk, filtered_text, score))
                continue

            # 调用 Qwen 生成摘要
            max_chars = int(max_tokens / len(chunks) * 1.5)  # 粗略分配
            prompt = SUMMARIZE_EVIDENCE.format(
                question=question,
                text=filtered_text[:4000],  # 截断输入
                max_chars=max(200, max_chars),
            )

            response = self.client.simple_chat(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1000,
            )
            total_api_tokens += response.total_tokens

            if response.content and not response.content.startswith("[API_ERROR"):
                summarized.append((chunk, response.content, score))
            else:
                # 摘要失败，截断原文
                summarized.append((chunk, filtered_text[:max_chars], score))

        return self._format_evidence(summarized), total_api_tokens

    def _format_evidence(
        self,
        chunks: list[tuple],
    ) -> str:
        """将 chunks 格式化为紧凑的证据文本"""
        parts = []
        current_doc = ""

        for item in chunks:
            if len(item) == 3:
                chunk, text, score = item
            else:
                chunk, score = item
                text = chunk.text

            # 文档分隔
            if chunk.doc_id != current_doc:
                current_doc = chunk.doc_id
                doc_type = chunk.doc_type
                parts.append(f"\n[文档: {chunk.doc_id} ({doc_type})]")

            # 段落内容
            if chunk.section_path:
                parts.append(f"## {chunk.section_path}")
            parts.append(text.strip())

        return "\n".join(parts)

    def _split_sentences(self, text: str) -> list[str]:
        """中文句子切分"""
        pattern = r"(?<=[。！？；])\s*"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
