"""本地评测辅助工具"""
import json
import csv
from pathlib import Path
from collections import defaultdict

from config.settings import SUBMISSION_DIR
from utils.logger import logger


def load_ground_truth(answers_file: str) -> dict[str, str]:
    """加载标准答案"""
    with open(answers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    answers = {}
    for item in data:
        qid = item.get("qid", "")
        answer = item.get("answer", "")
        answers[qid] = answer
    return answers


def load_predictions(csv_path: str) -> dict[str, str]:
    """加载预测答案"""
    predictions = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("qid", "")
            if qid == "summary":
                continue
            predictions[qid] = row.get("answer", "")
    return predictions


def evaluate(
    predictions: dict[str, str],
    ground_truth: dict[str, str],
    questions: list[dict] | None = None,
) -> dict:
    """评测准确率和分项统计"""
    total = 0
    correct = 0
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    wrong_list = []

    # 构建 qid -> question 映射
    q_map = {}
    if questions:
        for q in questions:
            q_map[q["qid"]] = q

    for qid, gt_answer in ground_truth.items():
        pred_answer = predictions.get(qid, "")
        total += 1

        # 标准化比较
        gt_norm = _normalize_answer(gt_answer)
        pred_norm = _normalize_answer(pred_answer)

        is_correct = gt_norm == pred_norm
        if is_correct:
            correct += 1

        # 按 domain 统计
        q_data = q_map.get(qid, {})
        domain = q_data.get("domain", "unknown")
        answer_format = q_data.get("answer_format", "mcq")

        by_domain[domain]["total"] += 1
        if is_correct:
            by_domain[domain]["correct"] += 1

        by_type[answer_format]["total"] += 1
        if is_correct:
            by_type[answer_format]["correct"] += 1

        if not is_correct:
            wrong_list.append({
                "qid": qid,
                "domain": domain,
                "predicted": pred_answer,
                "ground_truth": gt_answer,
            })

    accuracy = correct / total if total > 0 else 0

    # 计算 Token 消耗
    token_stats = _compute_token_stats(predictions)

    result = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_domain": {
            domain: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for domain, stats in by_domain.items()
        },
        "by_type": {
            qtype: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
            }
            for qtype, stats in by_type.items()
        },
        "wrong_list": wrong_list[:20],  # 只记录前 20 个错误
        "token_stats": token_stats,
    }

    return result


def _normalize_answer(answer: str) -> str:
    """标准化答案 (去重、排序、大写)"""
    letters = sorted(set(c.upper() for c in answer if c.upper() in "ABCD"))
    return "".join(letters)


def _compute_token_stats(predictions: dict[str, str]) -> dict:
    """计算 Token 统计"""
    csv_path = SUBMISSION_DIR / "answer.csv"
    if not csv_path.exists():
        return {}

    total_tokens = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("qid") == "summary":
                total_tokens = int(row.get("total_tokens", 0))
                break

    token_budget = 5_000_000
    token_score = max(0, min(1, (token_budget - total_tokens) / token_budget))

    return {
        "total_tokens": total_tokens,
        "token_score": token_score,
    }


def compute_final_score(accuracy: float, total_tokens: int) -> float:
    """计算最终得分"""
    token_budget = 5_000_000
    token_score = max(0, min(1, (token_budget - total_tokens) / token_budget))
    return 100 * accuracy * (0.7 + 0.3 * token_score)


def print_report(result: dict):
    """打印评测报告"""
    logger.info("=" * 60)
    logger.info("评测报告")
    logger.info("=" * 60)

    logger.info(f"总题数: {result['total']}")
    logger.info(f"正确数: {result['correct']}")
    logger.info(f"准确率: {result['accuracy']:.1%}")

    token_stats = result.get("token_stats", {})
    if token_stats:
        total_tokens = token_stats.get("total_tokens", 0)
        token_score = token_stats.get("token_score", 0)
        final_score = compute_final_score(result["accuracy"], total_tokens)
        logger.info(f"Token 消耗: {total_tokens:,}")
        logger.info(f"Token 效率分: {token_score:.3f}")
        logger.info(f"最终得分: {final_score:.2f}")

    logger.info(f"\n按文档类型:")
    for domain, stats in sorted(result["by_domain"].items()):
        logger.info(
            f"  {domain:25s}: {stats['correct']:3d}/{stats['total']:3d} "
            f"= {stats['accuracy']:.1%}"
        )

    logger.info(f"\n按题型:")
    for qtype, stats in sorted(result["by_type"].items()):
        logger.info(
            f"  {qtype:15s}: {stats['correct']:3d}/{stats['total']:3d} "
            f"= {stats['accuracy']:.1%}"
        )

    logger.info("=" * 60)


def quick_eval(questions_file: str, answers_file: str):
    """快速评测 (本地有答案时使用)"""
    ground_truth = load_ground_truth(answers_file)
    predictions = load_predictions(str(SUBMISSION_DIR / "answer.csv"))

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    result = evaluate(predictions, ground_truth, questions)
    print_report(result)
    return result
