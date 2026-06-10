"""文档级三层摘要生成 (离线阶段)"""
import json
from pathlib import Path

import jieba
from tqdm import tqdm

from config.settings import PARSED_DIR, CACHE_DIR
from config.prompts import DOC_SUMMARY_L1, DOC_SUMMARY_L2, DOC_SUMMARY_L3
from memory.doc_memory import DocMemory
from utils.api_client import QwenClient
from utils.logger import logger


def extract_doc_keywords(text: str, top_n: int = 50) -> list[str]:
    """提取文档 TF-IDF 关键词 (离线，不计 Token)"""
    words = jieba.lcut(text[:20000])
    stop_words = set("的 了 在 是 有 和 就 不 都 一 上 也 很 到 说 要 去 你 会 "
                     "着 没有 看 好 这 那 为 与 及 或 而 但 之 以 于 从 对 等 "
                     "将 被 把 向 比 所 如 果 能 可 并 且 该 此 其 更 最".split())

    # 词频统计
    freq: dict[str, int] = {}
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in stop_words:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, c in sorted_words[:top_n]]


def generate_summaries_for_doc(
    doc_id: str,
    text: str,
    doc_type: str,
    client: QwenClient,
) -> dict:
    """为单个文档生成三层摘要"""
    logger.info(f"生成摘要: {doc_id}")

    keywords = extract_doc_keywords(text)

    # L1 摘要 (200-300 字)
    l1_input = text[:5000]  # 取前 5000 字
    l1_prompt = DOC_SUMMARY_L1.format(text=l1_input)
    l1_response = client.simple_chat(l1_prompt, temperature=0.1, max_tokens=500)
    l1_summary = l1_response.content if l1_response.content else ""

    # L2 摘要 (1000-2000 字)
    l2_input = text[:15000]  # 取前 15000 字
    l2_prompt = DOC_SUMMARY_L2.format(text=l2_input)
    l2_response = client.simple_chat(l2_prompt, temperature=0.1, max_tokens=2000)
    l2_summary = l2_response.content if l2_response.content else ""

    # L3 摘要 (3000-5000 字) - 只对较短文档全文处理
    if len(text) <= 30000:
        l3_input = text
    else:
        # 长文档分段生成
        l3_input = text[:30000]
    l3_prompt = DOC_SUMMARY_L3.format(text=l3_input)
    l3_response = client.simple_chat(l3_prompt, temperature=0.1, max_tokens=4000)
    l3_summary = l3_response.content if l3_response.content else ""

    total_tokens = (l1_response.total_tokens +
                    l2_response.total_tokens +
                    l3_response.total_tokens)
    logger.info(f"  摘要完成: L1={len(l1_summary)}字, L2={len(l2_summary)}字, "
                f"L3={len(l3_summary)}字, tokens={total_tokens}")

    return {
        "l1_summary": l1_summary,
        "l2_summary": l2_summary,
        "l3_summary": l3_summary,
        "keywords": keywords,
    }


def generate_all_summaries(
    client: QwenClient,
    parsed_dir: Path | None = None,
) -> DocMemory:
    """为所有已解析文档生成摘要"""
    parsed_dir = parsed_dir or PARSED_DIR
    doc_memory = DocMemory()

    # 加载所有已解析文档
    parsed_files = sorted(parsed_dir.glob("*.json"))
    if not parsed_files:
        logger.error(f"在 {parsed_dir} 中未找到解析文件")
        return doc_memory

    logger.info(f"开始生成 {len(parsed_files)} 个文档的摘要...")

    for pf in tqdm(parsed_files, desc="生成摘要"):
        with open(pf, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_id = doc_data["doc_id"]
        text = doc_data["raw_text"]
        doc_type = doc_data["doc_type"]

        try:
            result = generate_summaries_for_doc(doc_id, text, doc_type, client)
            doc_memory.add_document(
                doc_id=doc_id,
                l1_summary=result["l1_summary"],
                l2_summary=result["l2_summary"],
                l3_summary=result["l3_summary"],
                keywords=result["keywords"],
                metadata=doc_data.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"生成摘要失败 {doc_id}: {e}")

    # 保存
    doc_memory.save()
    logger.info(f"所有摘要生成完成: {len(doc_memory.memories)} 个文档")
    return doc_memory
