"""主入口: 离线构建 + 在线答题"""
import argparse
import json
import csv
from pathlib import Path

from config.settings import (
    RAW_DIR, PARSED_DIR, INDEX_DIR, CACHE_DIR, SUBMISSION_DIR,
    TOTAL_TOKEN_BUDGET,
)
from utils.logger import logger, setup_logger
from utils.api_client import QwenClient
from utils.token_counter import TokenBudget


def run_offline():
    """运行离线 Pipeline: 多格式解析 → 清洗 → 结构化 → 分块 → 索引 → 摘要

    支持的文件格式：
    - PDF 文件 (.pdf/.PDF): 保险条款、金融合同、财务报表、行业研报
    - HTML 文件 (.html): 监管法规（证监会网站导出）
    - TXT 文件 (.txt): 监管法规（纯文本）
    """
    from offline.pdf_parser import parse_all_pdfs
    from offline.html_parser import parse_all_html
    from offline.txt_parser import parse_all_txt
    from offline.text_cleaner import clean_text
    from offline.structure_extractor import extract_structure
    from offline.chunker import chunk_all_documents
    from offline.index_builder import SearchIndex, build_financial_dict

    logger.info("=" * 60)
    logger.info("开始离线 Pipeline（多格式解析）")
    logger.info("=" * 60)

    # 1. 多格式文档解析
    logger.info("[1/5] 多格式文档解析...")

    # 1a. PDF 文件（保险、合同、财报、研报）
    logger.info("  [1a] 解析 PDF 文件...")
    pdf_docs = parse_all_pdfs()
    logger.info(f"  PDF: {len(pdf_docs)} 个文档")

    # 1b. HTML 文件（监管法规 - 证监会网站）
    logger.info("  [1b] 解析 HTML 文件...")
    html_docs = parse_all_html()
    logger.info(f"  HTML: {len(html_docs)} 个文档")

    # 1c. TXT 文件（监管法规 - 纯文本）
    logger.info("  [1c] 解析 TXT 文件...")
    txt_docs = parse_all_txt()
    logger.info(f"  TXT: {len(txt_docs)} 个文档")

    total_raw = len(pdf_docs) + len(html_docs) + len(txt_docs)
    if total_raw == 0:
        logger.error("没有解析到任何文档，请检查 data/raw/ 目录")
        return

    # 2. 文本清洗
    logger.info(f"[2/5] 文本清洗（共 {total_raw} 个文档）...")
    parsed_docs = []

    # 清洗 PDF 文档
    for doc in pdf_docs:
        doc.raw_text = clean_text(doc.raw_text, doc.doc_type)
        parsed_docs.append(doc.to_dict())

    # 清洗 HTML 文档
    for doc in html_docs:
        doc.raw_text = clean_text(doc.raw_text, doc.doc_type)
        parsed_docs.append(doc.to_dict())

    # 清洗 TXT 文档
    for doc in txt_docs:
        doc.raw_text = clean_text(doc.raw_text, doc.doc_type)
        parsed_docs.append(doc.to_dict())

    # 保存清洗后的数据
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    for doc_data in parsed_docs:
        output_path = PARSED_DIR / f"{doc_data['doc_id']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)

    # 统计各类型文档数量
    type_counts: dict[str, int] = {}
    for d in parsed_docs:
        dt = d["doc_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1
    logger.info(f"  文档类型分布: {type_counts}")

    # 3. 结构化提取
    logger.info("[3/5] 结构化提取...")
    structures = {}
    for doc_data in parsed_docs:
        structure = extract_structure(
            doc_data["raw_text"],
            doc_data["doc_id"],
            doc_data["doc_type"],
        )
        structures[doc_data["doc_id"]] = structure

    # 4. 分块
    logger.info("[4/5] 文档分块...")
    all_chunks = chunk_all_documents(parsed_docs, structures)

    # 5. 构建索引
    logger.info("[5/5] 构建索引...")
    build_financial_dict(parsed_docs)  # 先构建自定义词典
    index = SearchIndex()
    doc_meta = {d["doc_id"]: d.get("metadata", {}) for d in parsed_docs}
    index.build(all_chunks, doc_meta)
    index.save()

    logger.info("离线 Pipeline 完成！")
    logger.info(f"  文档总数: {len(parsed_docs)}")
    logger.info(f"  文档类型: {type_counts}")
    logger.info(f"  Chunk 数: {sum(len(v) for v in all_chunks.values())}")


def run_summarize():
    """运行文档摘要生成"""
    from offline.doc_summarizer import generate_all_summaries

    logger.info("=" * 60)
    logger.info("开始生成文档摘要")
    logger.info("=" * 60)

    client = QwenClient()
    doc_memory = generate_all_summaries(client)
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
            token_budget.used,  # 这里用总消耗近似
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
    args = parser.parse_args()

    # 配置日志文件
    if args.log_file:
        setup_logger(log_file=args.log_file)

    if args.mode == "offline":
        run_offline()
    elif args.mode == "summarize":
        run_summarize()
    elif args.mode == "online":
        if not args.questions:
            logger.error("online 模式需要 --questions 参数")
            return
        run_online(args.questions)
    elif args.mode == "all":
        run_offline()
        run_summarize()
        if args.questions:
            run_online(args.questions)
        else:
            logger.warning("未指定 --questions，跳过在线答题")


if __name__ == "__main__":
    main()
