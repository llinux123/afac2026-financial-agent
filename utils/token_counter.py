"""Token 估算与预算监控"""
from dataclasses import dataclass

from config.settings import TOTAL_TOKEN_BUDGET, BUDGET_MODES
from utils.logger import logger


@dataclass
class BudgetConfig:
    """单题预算配置"""
    mode: str
    max_evidence_tokens: int
    max_reasoning_tokens: int
    avg_per_question: int


class TokenBudget:
    """Token 预算管理器"""

    def __init__(self, total_budget: int = TOTAL_TOKEN_BUDGET):
        self.total_budget = total_budget
        self.used = 0
        self.per_question: list[dict] = []

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.used)

    @property
    def usage_ratio(self) -> float:
        return self.used / self.total_budget if self.total_budget > 0 else 1.0

    def record(self, qid: str, prompt_tokens: int, completion_tokens: int):
        """记录单题 Token 消耗"""
        total = prompt_tokens + completion_tokens
        self.used += total
        self.per_question.append({
            "qid": qid,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        })
        logger.debug(
            f"  [{qid}] tokens={total:,}, "
            f"累计={self.used:,}/{self.total_budget:,} "
            f"({self.usage_ratio:.1%})"
        )

    def get_budget(
        self, question_idx: int, total_questions: int,
    ) -> BudgetConfig:
        """根据当前进度动态分配单题预算"""
        remaining_questions = total_questions - question_idx
        if remaining_questions <= 0:
            remaining_questions = 1

        avg_remaining = self.remaining / remaining_questions

        if avg_remaining > 30000:
            mode = "normal"
        elif avg_remaining > 20000:
            mode = "compact"
        elif avg_remaining > 12000:
            mode = "aggressive"
        else:
            mode = "minimal"

        config = BUDGET_MODES[mode]

        if mode != "normal":
            logger.info(
                f"  预算模式: {mode} "
                f"(剩余={self.remaining:,}, "
                f"剩余题={remaining_questions}, "
                f"均={avg_remaining:,.0f})"
            )

        return BudgetConfig(
            mode=mode,
            max_evidence_tokens=config["max_evidence_tokens"],
            max_reasoning_tokens=config["max_reasoning_tokens"],
            avg_per_question=config["avg_per_question"],
        )

    def get_summary(self) -> dict:
        """获取预算使用摘要"""
        return {
            "total_budget": self.total_budget,
            "used": self.used,
            "remaining": self.remaining,
            "usage_ratio": self.usage_ratio,
            "questions_answered": len(self.per_question),
        }


def estimate_tokens(text: str) -> int:
    """估算文本的 Token 数量

    中文经验值：1个中文字符约 1.5-2 个 token
    英文：1个单词约 1-1.5 个 token
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    # 中文约 1.5 token/char, 其他约 0.5 token/char
    estimated = int(chinese_chars * 1.5 + other_chars * 0.5)
    return max(1, estimated)


def estimate_tokens_tiktoken(text: str) -> int:
    """使用 tiktoken 精确计算 Token 数 (仅支持 cl100k_base 编码)"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return estimate_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """截断文本到指定 Token 数"""
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text

    # 按比例截断
    ratio = max_tokens / estimated
    target_chars = int(len(text) * ratio * 0.9)  # 留10%余量
    return text[:target_chars]
