"""主入口: 离线构建 + 在线答题"""
import argparse
import json
import csv
import shutil
from pathlib import Path

from config.settings import (
    RAW_DIR, PARSED_DIR, CHUNKS_DIR, INDEX_DIR, CACHE_DIR, SUBMISSION_DIR,
    TOTAL_TOKEN_BUDGET,
)
from utils.logger import logger, setup_logger
from utils.api_client import QwenClient
from utils.token_counter import TokenBudget


def clear_pipeline_data():
    """清除所有中间产物，用于 --force 从头开始"""
    dirs_to_clear = [PARSED_DIR, CHUNKS_DIR, INDEX_DIR]
    for d in dirs_to_clear:
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"  已清除: {d}")
    # 清除摘要缓存
    doc_memory_path = CACHE_DIR / "doc_memory.json"
    if doc_memory_path.exists():
        doc_memory_path.unlink()
        logger.info(f"  已清除: {doc_memory_path}")


def run_offline(force: bool = False):
    """运行离线 Pipeline: 解析+清洗 → 结构化+分块 → 索引

    断点续跑机制：
    - 阶段 A: 解析+清洗 — 检查 PARSED_DIR/{doc_id}.json 是否存在
    - 阶段 B: 结构化+分块 — 检查 CHUNKS_DIR/{doc_id}.json 是否存在
    - 阶段 C: 构建索引 — 检查 INDEX_DIR/search_index.pkl 是否存在

    每个文档处理完立即持久化，中断后重启可跳过已完成的文档。
    """
    from offline.pdf_parser import parse_all_pdfs
    from offline.html_parser import parse_all_html
    from offline.txt_parser import parse_all_txt
    from offline.structure_extractor import extract_structure
    from offline.chunker import chunk_all_documents
    from offline.index_builder import SearchIndex, build_financial_dict

    logger.info("=" * 60)
    logger.info(f"开始离线 Pipeline（断点续跑{' | 强制重跑' if force else ''}）")
    logger.info("=" * 60)

    # ── 阶段 A: 解析 + 清洗 (原子操作，每文档独立持久化到 PARSED_DIR) ──
    logger.info("[A/3] 解析 + 清洗...")

    logger.info("  解析 PDF 文件...")
    pdf_docs = parse_all_pdfs(force=force)

    logger.info("  解析 HTML 文件...")
    html_docs = parse_all_html(force=force)

    logger.info("  解析 TXT 文件...")
    txt_docs = parse_all_txt(force=force)

    # 合并所有已清洗的文档
    parsed_docs = pdf_docs + html_docs + txt_docs
    total = len(parsed_docs)

    if total == 0:
        logger.error("没有解析到任何文档，请检查 data/raw/ 目录")
        return

    # 统计文档类型分布
    type_counts: dict[str, int] = {}
    for d in parsed_docs:
        dt = d["doc_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1
    logger.info(f"  文档总数: {total}, 类型分布: {type_counts}")

    # ── 阶段 B: 结构化提取 + 分块 (每文档独立持久化到 CHUNKS_DIR) ──
    logger.info("[B/3] 结构化提取 + 分块...")

    structures = {}
    for doc_data in parsed_docs:
        structures[doc_data["doc_id"]] = extract_structure(
            doc_data["raw_text"],
            doc_data["doc_id"],
            doc_data["doc_type"],
        )

    all_chunks = chunk_all_documents(parsed_docs, structures, force=force)

    # ── 阶段 C: 构建索引 (从全量 chunks 一次性构建) ──
    index_path = INDEX_DIR / "search_index.pkl"
    if not force and index_path.exists():
        logger.info(f"[C/3] 索引已存在，跳过构建: {index_path}")
    else:
        logger.info("[C/3] 构建索引...")
        build_financial_dict(parsed_docs)
        index = SearchIndex()
        doc_meta = {d["doc_id"]: d.get("metadata", {}) for d in parsed_docs}
        index.build(all_chunks, doc_meta)
        index.save()

    logger.info("离线 Pipeline 完成！")
    logger.info(f"  文档总数: {total}")
    logger.info(f"  文档类型: {type_counts}")
    logger.info(f"  Chunk 数: {sum(len(v) for v in all_chunks.values())}")


def run_summarize(force: bool = False, max_workers: int = 6):
    """运行文档摘要生成（支持断点续跑）

    Args:
        force: 是否强制重新生成所有摘要。
        max_workers: 并发线程数，默认 6，有效范围 1-16。
    """
    from offline.doc_summarizer import generate_all_summaries

    logger.info("=" * 60)
    logger.info(f"开始生成文档摘要{'（强制重跑）' if force else ''}，"
                f"并发线程数: {max_workers}")
    logger.info("=" * 60)

    client = QwenClient()
    doc_memory = generate_all_summaries(client, force=force, max_workers=max_workers)
    logger.info(f"摘要生成完成，消耗: {client.get_usage().summary()}")


def _collect_question_files(questions_path: str) -> list[Path]:
    """收集题目文件路径列表，支持单个文件或目录。

    Args:
        questions_path: 题目文件路径或目录路径。

    Returns:
        按文件名排序的 JSON 文件路径列表。

    Raises:
        FileNotFoundError: 路径不存在时抛出。
        ValueError: 路径存在但既不是文件也不是目录时抛出。
    """
    path = Path(questions_path)

    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if path.is_file():
        return [path]

    if path.is_dir():
        json_files = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
        return json_files

    raise ValueError(f"路径既不是文件也不是目录: {path}")


def run_online(questions_path: str):
    """运行在线答题，支持单个 JSON 文件或包含多个 JSON 文件的目录。

    断点续跑机制：
    - 启动时读取 answer.csv，提取已完成的 qid 列表并跳过。
    - 每处理完一道题目后立即追加写入 CSV（逐题持久化）。
    - 处理完成后清理可能的重复行。

    Args:
        questions_path: 题目文件路径或目录路径。
    """
    from offline.index_builder import SearchIndex
    from memory.doc_memory import DocMemory
    from online.agent_loop import AgentLoop

    # 收集所有题目文件
    try:
        question_files = _collect_question_files(questions_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"{e}")
        return

    if not question_files:
        logger.warning(f"目录中未找到 JSON 文件: {questions_path}")
        return

    # ── 断点续跑：加载已完成的 qid ──
    completed_qids = _load_completed_qids()
    if completed_qids:
        logger.info(f"检测到已有结果: {len(completed_qids)} 道题目已完成，将跳过")

    logger.info("=" * 60)
    logger.info("开始在线答题")
    logger.info(f"  输入路径: {questions_path}")
    logger.info(f"  题目文件数: {len(question_files)}")
    logger.info("=" * 60)

    # 加载索引
    index = SearchIndex()
    index.load()

    # 加载文档记忆
    doc_memory = DocMemory()
    doc_memory.load()

    # 初始化 Agent（全局共享 TokenBudget）
    client = QwenClient()
    token_budget = TokenBudget()
    agent = AgentLoop(client, index, doc_memory, token_budget)

    all_results: list[dict] = []
    total_questions = 0
    skipped_count = 0
    processed_count = 0

    # 定义逐题回调：处理完一题后立即持久化
    def on_question_done(result: dict):
        nonlocal processed_count
        save_submission(
            [], token_budget, append=True, single_result=result,
        )
        processed_count += 1

    for qfile in question_files:
        logger.info(f"处理文件: {qfile.name}")

        # 加载题目
        with open(qfile, "r", encoding="utf-8") as f:
            questions = json.load(f)

        # 过滤已完成的题目
        pending_questions = []
        for q in questions:
            qid = q.get("qid", "")
            if qid and qid in completed_qids:
                skipped_count += 1
                continue
            pending_questions.append(q)

        logger.info(f"  加载 {len(questions)} 道题目，跳过 {len(questions) - len(pending_questions)} 道，待处理 {len(pending_questions)} 道")
        total_questions += len(questions)

        if not pending_questions:
            continue

        # 处理待处理的题目（带逐题回调）
        results = agent.process_all(pending_questions, on_question_done=on_question_done)
        all_results.extend(results)

    # 清理可能的重复行（保留每个 qid 的最后一行）
    _deduplicate_answer_csv()

    # 最终写入 summary 行
    save_submission([], token_budget, append=True, write_summary=True)

    # 输出统计
    logger.info("=" * 60)
    logger.info(f"答题完成！")
    logger.info(f"  处理文件数: {len(question_files)}")
    logger.info(f"  总题目数: {total_questions}")
    logger.info(f"  跳过（已存在）: {skipped_count}")
    logger.info(f"  本次处理: {processed_count}")
    logger.info(f"  {client.get_usage().summary()}")
    logger.info(f"  预算使用: {token_budget.used:,}/{token_budget.total_budget:,}")
    logger.info("=" * 60)


def _load_completed_qids() -> set[str]:
    """读取 answer.csv 中已完成的 qid 列表（跳过 summary 行）"""
    output_path = SUBMISSION_DIR / "answer.csv"
    completed = set()
    if not output_path.exists():
        return completed

    try:
        with open(output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # 跳过表头
            if not header:
                return completed
            for row in reader:
                if len(row) >= 1 and row[0] and row[0] != "summary":
                    completed.add(row[0])
    except Exception as e:
        logger.warning(f"读取已有 answer.csv 失败: {e}")

    return completed


def _deduplicate_answer_csv():
    """清理 answer.csv 中的重复 qid 行，保留每个 qid 的最后一行记录"""
    output_path = SUBMISSION_DIR / "answer.csv"
    if not output_path.exists():
        return

    try:
        # 读取所有行，用 OrderedDict 保留最后出现的记录
        from collections import OrderedDict

        rows = OrderedDict()
        summary_row = None

        with open(output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return

            for row in reader:
                if len(row) >= 1 and row[0]:
                    if row[0] == "summary":
                        summary_row = row
                    else:
                        rows[row[0]] = row

        # 写回文件
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows.values():
                writer.writerow(row)
            if summary_row:
                writer.writerow(summary_row)

        logger.info(f"已清理 answer.csv 重复行，保留 {len(rows)} 条唯一记录")
    except Exception as e:
        logger.warning(f"清理 answer.csv 重复行失败: {e}")


def save_submission(
    results: list[dict],
    token_budget: TokenBudget,
    *,
    append: bool = False,
    write_summary: bool = False,
    single_result: dict | None = None,
):
    """保存提交文件 answer.csv

    Args:
        results: 题目结果列表（批量写入时使用）。
        token_budget: Token 预算对象。
        append: 是否以追加模式写入。False 时覆盖写入并写入表头。
        write_summary: 是否写入 summary 汇总行。
        single_result: 单题结果（逐题持久化时使用），优先级高于 results。
    """
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "answer.csv"

    mode = "a" if append else "w"
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not append:
            # 写入表头
            writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])

        if write_summary:
            # 汇总行
            writer.writerow([
                "summary", "-",
                token_budget.used,
                0,
                token_budget.used,
            ])
        elif single_result is not None:
            # 逐题写入模式
            writer.writerow([
                single_result["qid"],
                single_result["answer"],
                single_result.get("prompt_tokens", 0),
                single_result.get("completion_tokens", 0),
                single_result.get("total_tokens", 0),
            ])
        else:
            # 批量写入模式
            for r in results:
                writer.writerow([
                    r["qid"],
                    r["answer"],
                    r.get("prompt_tokens", 0),
                    r.get("completion_tokens", 0),
                    r.get("total_tokens", 0),
                ])

    if not append or write_summary or single_result is not None:
        logger.info(f"提交文件已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="金融长文本 Agent")
    parser.add_argument("--mode", choices=["offline", "summarize", "online", "all"],
                        default="all", help="运行模式")
    parser.add_argument("--questions", type=str, default="",
                        help="题目文件路径 (online 模式，支持单个 JSON 文件或包含多个 JSON 文件的目录)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理（清除所有中间产物，忽略断点续跑）")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="摘要生成并发线程数（默认6，范围1-16，超出自动修正）")
    args = parser.parse_args()

    # 配置日志文件
    if args.log_file:
        setup_logger(log_file=args.log_file)

    # --force 时先清除中间产物
    if args.force:
        logger.info("--force 模式：清除所有中间产物...")
        clear_pipeline_data()

    if args.mode == "offline":
        run_offline(force=args.force)
    elif args.mode == "summarize":
        run_summarize(force=args.force, max_workers=args.max_workers)
    elif args.mode == "online":
        if not args.questions:
            logger.error("online 模式需要 --questions 参数")
            return
        run_online(args.questions)
    elif args.mode == "all":
        run_offline(force=args.force)
        run_summarize(force=args.force, max_workers=args.max_workers)
        if args.questions:
            run_online(args.questions)
        else:
            logger.warning("未指定 --questions，跳过在线答题")


if __name__ == "__main__":
    main()
