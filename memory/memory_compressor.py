"""记忆压缩策略 - TF-IDF 句子排序 + 规则加分"""
import re
import math

import jieba
import numpy as np

from utils.logger import logger


class MemoryCompressor:
    """记忆压缩器 - 基于 TF-IDF 句子重要性排序"""

    def compress_evidence(
        self, evidence: str, question: str = "", max_chars: int = 2000
    ) -> str:
        """压缩证据为简短摘要 (本地计算，不调用 LLM)

        Args:
            evidence: 原始证据文本
            question: 相关问题（可选，用于计算相关性）
            max_chars: 最大保留字符数
        """
        if len(evidence) <= max_chars:
            return evidence

        # 中文分句
        sentences = self._split_sentences(evidence)
        if not sentences:
            return evidence[:max_chars]

        # 如果没有 question，回退到规则压缩
        if not question:
            return self._rule_based_compress(evidence, max_chars)

        # TF-IDF 句子排序
        scores = self._tfidf_sentence_scores(sentences, question)

        # 选择高分句子 + 上下文
        selected_indices = self._select_sentences(sentences, scores, max_chars)

        # 按原文顺序拼接
        result = "\n".join(sentences[i] for i in sorted(selected_indices))

        # 最终截断保护
        if len(result) > max_chars:
            result = result[:max_chars]

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """中文分句"""
        # 按句号、问号、感叹号、分号分割
        pattern = r"(?<=[。！？；\n])"
        parts = re.split(pattern, text)
        sentences = []
        for part in parts:
            part = part.strip()
            if part and len(part) > 5:  # 过滤太短的片段
                sentences.append(part)
        return sentences

    def _tfidf_sentence_scores(
        self, sentences: list[str], question: str
    ) -> np.ndarray:
        """计算每个句子与问题的 TF-IDF 相关性分数"""
        n_sentences = len(sentences)
        scores = np.zeros(n_sentences)

        # 问题分词
        question_tokens = set(jieba.lcut(question))
        question_tokens = {t for t in question_tokens if len(t) >= 2}

        if not question_tokens:
            return scores

        # 对每个句子分词
        sentence_token_sets = []
        for sent in sentences:
            tokens = set(jieba.lcut(sent))
            tokens = {t for t in tokens if len(t) >= 2}
            sentence_token_sets.append(tokens)

        # 计算 IDF（基于句子集合作为文档）
        all_tokens = set()
        for tokens in sentence_token_sets:
            all_tokens.update(tokens)

        idf = {}
        for token in all_tokens:
            doc_freq = sum(1 for tokens in sentence_token_sets if token in tokens)
            idf[token] = math.log((n_sentences + 1) / (doc_freq + 1)) + 1

        # 计算每句与问题的相关性
        for i, sent_tokens in enumerate(sentence_token_sets):
            # 与问题的交集，加权 IDF
            overlap = question_tokens & sent_tokens
            if overlap:
                scores[i] = sum(idf.get(t, 1.0) for t in overlap)

            # 加分：包含数字/金额/比例
            if re.search(r"\d+\.?\d*\s*(%|万|亿|元|年|月|日|天)", sentences[i]):
                scores[i] += 2.0

            # 加分：包含条件/约束关键词
            condition_keywords = ["应", "须", "不得", "可以", "例外", "条件", "但"]
            for kw in condition_keywords:
                if kw in sentences[i]:
                    scores[i] += 0.5
                    break

            # 加分：标题行
            if sentences[i].startswith("[") or sentences[i].startswith("##"):
                scores[i] += 3.0

        return scores

    def _select_sentences(
        self, sentences: list[str], scores: np.ndarray, max_chars: int
    ) -> set[int]:
        """选择高分句子 + 上下文，不超过字符预算"""
        n = len(sentences)
        # 按分数降序排列
        ranked_indices = np.argsort(-scores)

        selected = set()
        total_chars = 0
        budget = int(max_chars * 0.9)  # 预留10%余量

        for idx in ranked_indices:
            idx = int(idx)
            if scores[idx] <= 0:
                break
            if total_chars >= budget:
                break

            # 添加当前句子
            if idx not in selected:
                selected.add(idx)
                total_chars += len(sentences[idx])

            # 添加上下文（前后各1句）
            if idx > 0 and idx - 1 not in selected:
                if total_chars + len(sentences[idx - 1]) <= budget:
                    selected.add(idx - 1)
                    total_chars += len(sentences[idx - 1])
            if idx < n - 1 and idx + 1 not in selected:
                if total_chars + len(sentences[idx + 1]) <= budget:
                    selected.add(idx + 1)
                    total_chars += len(sentences[idx + 1])

        # 如果什么都没选到，至少选前几句
        if not selected:
            for i in range(min(3, n)):
                selected.add(i)

        return selected

    def _rule_based_compress(self, evidence: str, max_chars: int) -> str:
        """兜底：基于规则的压缩（无 question 时使用）"""
        lines = evidence.split("\n")
        important_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 保留标题行
            if line.startswith("[") or line.startswith("##"):
                important_lines.append(line)
                continue
            # 保留含数字的行
            if re.search(r"\d+\.?\d*\s*(%|万|亿|元|年|月|日|天)", line):
                important_lines.append(line)
                continue
            # 保留含条件关键词的行
            if any(kw in line for kw in ["应", "须", "不得", "可以", "例外", "条件"]):
                important_lines.append(line)

        result = "\n".join(important_lines)
        if len(result) > max_chars:
            result = result[:max_chars]

        return result if result else evidence[:max_chars]

    def extract_facts(
        self, question: str, answer: str, evidence: str,
    ) -> list[tuple[str, str]]:
        """从问答过程中提取事实三元组"""
        facts = []

        # 提取 "X = Y" 模式的事实
        patterns = [
            (r"(.{2,15})为(\d+\.?\d*\s*%?)", lambda m: (m.group(1), m.group(2))),
            (r"(.{2,15})是(\d+\.?\d*\s*(?:万|亿)?元)", lambda m: (m.group(1), m.group(2))),
            (r"(.{2,15})(?:为|是)(.{2,20}(?:天|日|月|年))", lambda m: (m.group(1), m.group(2))),
        ]

        for pattern, extractor in patterns:
            for match in re.finditer(pattern, evidence):
                try:
                    key, value = extractor(match)
                    facts.append((key.strip(), value.strip()))
                except Exception:
                    pass

        return facts[:10]
