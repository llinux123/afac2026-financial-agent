"""记忆压缩策略"""
from utils.logger import logger


class MemoryCompressor:
    """记忆压缩器"""

    def compress_evidence(self, evidence: str, max_chars: int = 2000) -> str:
        """压缩证据为简短摘要 (本地规则，不调用 LLM)"""
        if len(evidence) <= max_chars:
            return evidence

        # 策略: 保留包含数字和关键词的句子
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
            import re
            if re.search(r"\d+\.?\d*\s*(%|万|亿|元|年|月|日|天)", line):
                important_lines.append(line)
                continue
            # 保留含条件关键词的行
            if any(kw in line for kw in ["应", "须", "不得", "可以", "例外", "条件"]):
                important_lines.append(line)

        result = "\n".join(important_lines)
        if len(result) > max_chars:
            result = result[:max_chars]

        return result

    def extract_facts(
        self, question: str, answer: str, evidence: str,
    ) -> list[tuple[str, str]]:
        """从问答过程中提取事实三元组"""
        import re
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
