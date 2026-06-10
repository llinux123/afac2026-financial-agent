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


def run_summarize(force: bool = False):
    """运行文档摘要生成（支持断点续跑）"""
    from offline.doc_summarizer import generate_all_summaries

    logger.info("=" * 60)
    logger.info(f"开始生成文档摘要{'（强制重跑）' if force else ''}")
    logger.info("=" * 60)

    client = QwenClient()
    doc_memory = generate_all_summaries(client, force=force)
    logger.info(f"摘要生成完成，消耗: {client.get_usage().summary()}")


def run_online(questions_file: str):
    """运行在线答题"""
    from offline.index_builder import SearchIndex
    from memory.doc_memory import DocMemory
    from online.agent_loop import AgentLoop

    logger.info("=" * 60)
    logger.info("开始在线答题")
    logger.info("=" * 60)

    # 加载索引
    index = SearchIndex()
    index.load()

    # 加载文档记忆
    doc_memory = DocMemory()
    doc_memory.load()

    # 加载题目
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info(f"加载 {len(questions)} 道题目")

    # 初始化 Agent
    client = QwenClient()
    token_budget = TokenBudget()
    agent = AgentLoop(client, index, doc_memory, token_budget)

    # 处理所有题目
    results = agent.process_all(questions)

    # 生成提交文件
    save_submission(results, token_budget)

    # 输出统计
    logger.info("=" * 60)
    logger.info(f"答题完成！")
    logger.info(f"  {client.get_usage().summary()}")
    logger.info(f"  预算使用: {token_budget.used:,}/{token_budget.total_budget:,}")
    logger.info("=" * 60)


def save_submission(results: list[dict], token_budget: TokenBudget):
    """保存提交文件 answer.csv"""
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SUBMISSION_DIR / "answer.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])

        # 汇总行
        writer.writerow([
            "summary", "-",
            token_budget.used,
            0,
            token_budget.used,
        ])

        # 每题数据
        for r in results:
            writer.writerow([
                r["qid"],
                r["answer"],
                r.get("prompt_tokens", 0),
                r.get("completion_tokens", 0),
                r.get("total_tokens", 0),
            ])

    logger.info(f"提交文件已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="金融长文本 Agent")
    parser.add_argument("--mode", choices=["offline", "summarize", "online", "all"],
                        default="all", help="运行模式")
    parser.add_argument("--questions", type=str, default="",
                        help="题目文件路径 (online 模式)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="日志文件路径")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理（清除所有中间产物，忽略断点续跑）")
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
        run_summarize(force=args.force)
    elif args.mode == "online":
        if not args.questions:
            logger.error("online 模式需要 --questions 参数")
            return
        run_online(args.questions)
    elif args.mode == "all":
        run_offline(force=args.force)
        run_summarize(force=args.force)
        if args.questions:
            run_online(args.questions)
        else:
            logger.warning("未指定 --questions，跳过在线答题")


if __name__ == "__main__":
    main()
