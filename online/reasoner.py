"""CoT 推理引擎"""
from config.prompts import (
    REASON_SINGLE_PROMPT, REASON_MULTI_PROMPT, REASON_JUDGE_PROMPT,
)
from online.question_analyzer import AnalyzedQuestion
from utils.api_client import QwenClient, APIResponse
from utils.logger import logger


class Reasoner:
    """基于 CoT 的推理引擎"""

    def __init__(self, client: QwenClient):
        self.client = client

    def reason(
        self,
        question: AnalyzedQuestion,
        evidence: str,
        enable_thinking: bool = False,
    ) -> tuple[str, APIResponse]:
        """对单道题目进行推理

        Returns:
            (原始模型输出, APIResponse)
        """
        # 根据题型选择 prompt
        if question.question_type == "multi":
            prompt = self._build_multi_prompt(question, evidence)
        elif question.question_type == "judge":
            prompt = self._build_judge_prompt(question, evidence)
        else:
            prompt = self._build_single_prompt(question, evidence)

        # 调用 Qwen
        response = self.client.simple_chat(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2000,
            enable_thinking=enable_thinking,
        )

        return response.content, response

    def reason_second_round(
        self,
        question: AnalyzedQuestion,
        evidence: str,
        first_answer: str,
        first_analysis: str,
    ) -> tuple[str, APIResponse]:
        """第二轮定向推理 (不确定时使用)"""
        extra_prompt = f"""上一轮分析结果：
{first_analysis}

上一轮答案：{first_answer}

请根据补充的文档证据，重新审视你的答案是否正确。如果确认无误，保持原答案；如果发现错误，请修正。

补充证据：
{evidence}
"""
        # 在原始 prompt 基础上加入第二轮信息
        if question.question_type == "multi":
            base_prompt = REASON_MULTI_PROMPT.format(
                evidence="[见补充证据]",
                question=question.question,
                **{f"option_{k.lower()}": v for k, v in question.options.items()},
            )
        elif question.question_type == "judge":
            base_prompt = REASON_JUDGE_PROMPT.format(
                evidence="[见补充证据]",
                question=question.question,
                **{f"option_{k.lower()}": v for k, v in question.options.items()},
            )
        else:
            base_prompt = REASON_SINGLE_PROMPT.format(
                evidence="[见补充证据]",
                question=question.question,
                **{f"option_{k.lower()}": v for k, v in question.options.items()},
            )

        full_prompt = base_prompt + "\n\n" + extra_prompt

        response = self.client.simple_chat(
            prompt=full_prompt,
            temperature=0.1,
            max_tokens=2000,
        )
        return response.content, response

    def _build_single_prompt(
        self, question: AnalyzedQuestion, evidence: str,
    ) -> str:
        """构建单选题 prompt"""
        options = question.options
        return REASON_SINGLE_PROMPT.format(
            evidence=evidence,
            question=question.question,
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
            option_c=options.get("C", ""),
            option_d=options.get("D", ""),
        )

    def _build_multi_prompt(
        self, question: AnalyzedQuestion, evidence: str,
    ) -> str:
        """构建多选题 prompt"""
        options = question.options
        return REASON_MULTI_PROMPT.format(
            evidence=evidence,
            question=question.question,
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
            option_c=options.get("C", ""),
            option_d=options.get("D", ""),
        )

    def _build_judge_prompt(
        self, question: AnalyzedQuestion, evidence: str,
    ) -> str:
        """构建判断题 prompt"""
        options = question.options
        return REASON_JUDGE_PROMPT.format(
            evidence=evidence,
            question=question.question,
            option_a=options.get("A", ""),
            option_b=options.get("B", ""),
        )
