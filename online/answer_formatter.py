"""答案后处理与格式校验"""
import re

from online.question_analyzer import AnalyzedQuestion
from utils.logger import logger


def extract_answer(
    model_output: str,
    question: AnalyzedQuestion,
) -> str:
    """从模型输出中提取答案

    Returns:
        格式化的答案字符串 (如 "A", "ABC", "B")
    """
    question_type = question.question_type

    # 策略1: 匹配 "答案：X" 模式
    answer = _extract_from_pattern(model_output, question_type)
    if answer:
        return _validate_answer(answer, question_type)

    # 策略2: 取最后一行的大写字母
    answer = _extract_from_last_line(model_output, question_type)
    if answer:
        return _validate_answer(answer, question_type)

    # 策略3: 扫描全文找大写字母组合
    answer = _extract_all_letters(model_output, question_type)
    if answer:
        return _validate_answer(answer, question_type)

    # 兜底: 返回默认答案
    logger.warning(f"  [{question.qid}] 答案提取失败，使用默认值")
    return _default_answer(question_type)


def _extract_from_pattern(text: str, question_type: str) -> str | None:
    """从 '答案：X' 模式提取"""
    patterns = [
        r"答案[：:]\s*([A-Da-d]+)",       # 答案：A 或 答案: ABC
        r"最终答案[：:]\s*([A-Da-d]+)",    # 最终答案：A
        r"选择[：:]\s*([A-Da-d]+)",        # 选择：A
        r"answer[：:]\s*([A-Da-d]+)",      # answer: A
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def _extract_from_last_line(text: str, question_type: str) -> str | None:
    """从最后一行提取大写字母"""
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue

        # 查找连续的大写字母
        letters = re.findall(r"[A-D]", line)
        if letters:
            result = "".join(letters)
            # 验证长度合理
            if question_type == "single" and len(result) == 1:
                return result
            elif question_type == "judge" and len(result) == 1:
                return result
            elif question_type == "multi" and 1 <= len(result) <= 4:
                return result
    return None


def _extract_all_letters(text: str, question_type: str) -> str | None:
    """从全文中提取答案 (最后出现的合理字母组合)"""
    # 匹配独立的大写字母组合 (周围有非字母字符)
    matches = re.findall(r"(?<![A-Za-z])([A-D]{1,4})(?![A-Za-z])", text)
    if not matches:
        return None

    # 取最后一个匹配
    for match in reversed(matches):
        if question_type == "single" and len(match) == 1:
            return match
        elif question_type == "judge" and len(match) == 1:
            return match
        elif question_type == "multi" and 1 <= len(match) <= 4:
            return match

    return matches[-1] if matches else None


def _validate_answer(answer: str, question_type: str) -> str:
    """校验并格式化答案"""
    # 去重
    letters = list(set(answer))
    # 排序
    letters.sort()
    # 过滤合法字母
    letters = [l for l in letters if l in "ABCD"]

    if not letters:
        return _default_answer(question_type)

    if question_type == "single":
        return letters[0]  # 单选只取第一个
    elif question_type == "judge":
        return letters[0]  # 判断只取第一个
    else:
        return "".join(letters)  # 多选返回所有


def _default_answer(question_type: str) -> str:
    """返回默认答案"""
    if question_type == "single":
        return "A"
    elif question_type == "judge":
        return "A"
    else:
        return "A"
