"""文档级三层摘要生成 (离线阶段)

线程池配置常量:
    MAX_WORKERS_DEFAULT: 默认并发线程数，适用于大多数 I/O 密集型 API 调用场景
    MAX_WORKERS_UPPER_LIMIT: 线程数上限，防止系统资源过度消耗。
        设置为 16 是基于以下考量:
        - 每个线程占用约 8MB 栈空间，16 线程 ≈ 128MB 内存开销
        - API 服务端通常有并发请求限制，过高并发会触发限流
        - 文件 I/O 和 JSON 序列化在锁内执行，过高并发会加剧锁竞争
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jieba
from tqdm import tqdm

from config.settings import PARSED_DIR, CACHE_DIR
from config.prompts import DOC_SUMMARY_L1, DOC_SUMMARY_L2, DOC_SUMMARY_L3
from memory.doc_memory import DocMemory
from utils.api_client import QwenClient
from utils.logger import logger

# ============================================================
# 线程池配置常量
# ============================================================
MAX_WORKERS_DEFAULT: int = 6
"""默认并发线程数。

适用于大多数 I/O 密集型 API 调用场景，在并发吞吐与资源开销之间取得平衡。
建议值范围 4-8，当前取中间值 6。
"""

MAX_WORKERS_UPPER_LIMIT: int = 16
"""并发线程数上限。

防止用户设置过高线程数导致系统资源过度消耗。超过此值的输入将被自动截断。
该上限基于以下约束:
  - 内存: 每线程 ~8MB 栈空间，16 线程 ≈ 128MB
  - API 限流: 过高并发易触发服务端限流
  - 锁竞争: doc_memory 的读写需持锁，过高并发加剧竞争
"""


def validate_max_workers(max_workers: int) -> int:
    """验证并修正线程池大小参数。

    Args:
        max_workers: 用户指定的并发线程数。

    Returns:
        经过验证和修正后的有效线程数。

    Raises:
        ValueError: 当 max_workers 不是整数类型时。
    """
    if not isinstance(max_workers, int):
        raise ValueError(
            f"max_workers 必须为整数，当前类型: {type(max_workers).__name__}"
        )
    if max_workers < 1:
        logger.warning(
            f"max_workers={max_workers} 无效（需 >= 1），"
            f"已修正为 {MAX_WORKERS_DEFAULT}"
        )
        return MAX_WORKERS_DEFAULT
    if max_workers > MAX_WORKERS_UPPER_LIMIT:
        logger.warning(
            f"max_workers={max_workers} 超过上限 {MAX_WORKERS_UPPER_LIMIT}，"
            f"已截断为 {MAX_WORKERS_UPPER_LIMIT}"
        )
        return MAX_WORKERS_UPPER_LIMIT
    return max_workers


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
    force: bool = False,
    max_workers: int = MAX_WORKERS_DEFAULT,
) -> DocMemory:
    """为所有已解析文档生成摘要（多线程并发）。

    断点续跑：先加载已有摘要，跳过已完成的文档，每处理完一个立即保存。
    使用 ThreadPoolExecutor 并发处理多个文档，适合 I/O 密集型 API 调用。

    Args:
        client: Qwen API 客户端，用于调用 LLM 生成摘要。
        parsed_dir: 已解析文档目录，默认使用 config.settings.PARSED_DIR。
        force: 是否强制重新生成所有摘要（忽略断点续跑缓存）。
        max_workers: 最大并发线程数，默认 MAX_WORKERS_DEFAULT(6)。
            有效范围: 1 ~ MAX_WORKERS_UPPER_LIMIT(16)。
            超出范围时将自动修正并输出警告日志。
            建议值: 4-8（I/O 密集型场景），过高值可能导致 API 限流或锁竞争。

    Returns:
        DocMemory: 包含所有文档摘要的记忆对象。

    Raises:
        ValueError: 当 max_workers 参数类型不正确时。
    """
    # 验证并修正线程池大小
    max_workers = validate_max_workers(max_workers)
    logger.info(f"线程池大小: {max_workers}")

    parsed_dir = parsed_dir or PARSED_DIR
    doc_memory = DocMemory()

    # 加载已有摘要（断点续跑）
    if not force:
        doc_memory.load()

    # 加载所有已解析文档
    parsed_files = sorted(parsed_dir.glob("*.json"))
    if not parsed_files:
        logger.error(f"在 {parsed_dir} 中未找到解析文件")
        return doc_memory

    # 过滤待处理文档列表（断点续跑：跳过已有摘要）
    files_to_process = [
        pf for pf in parsed_files
        if force or _doc_id_from_path(pf) not in doc_memory.memories
    ]
    skipped = len(parsed_files) - len(files_to_process)

    logger.info(
        f"摘要生成: 共 {len(parsed_files)} 个文档, "
        f"跳过已有: {skipped}, 待处理: {len(files_to_process)}"
    )

    if not files_to_process:
        logger.info("没有需要处理的文档")
        return doc_memory

    # 线程锁：保护 doc_memory 的读写操作
    lock = threading.Lock()

    # 性能监控计数器（线程安全，通过 lock 保护）
    processed_count = 0
    failed_count = 0
    start_time = time.monotonic()

    def process_single_doc(pf: Path) -> str | None:
        """Worker function: 处理单个文档并线程安全地保存。

        处理流程:
        1. 读取并解析文档 JSON（无需锁）
        2. 调用 LLM API 生成摘要（I/O 密集型，无需锁）
        3. 获取锁后写入 doc_memory 并持久化（线程安全临界区）

        Args:
            pf: 待处理的文档 JSON 文件路径。

        Returns:
            成功时返回 doc_id，失败时返回 None。
        """
        nonlocal processed_count, failed_count
        doc_id = _doc_id_from_path(pf)
        try:
            with open(pf, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            text = doc_data["raw_text"]
            doc_type = doc_data["doc_type"]

            # API 调用（I/O 密集型，不需要锁）
            result = generate_summaries_for_doc(doc_id, text, doc_type, client)

            # 线程安全地写入 doc_memory 并持久化
            with lock:
                doc_memory.add_document(
                    doc_id=doc_id,
                    l1_summary=result["l1_summary"],
                    l2_summary=result["l2_summary"],
                    l3_summary=result["l3_summary"],
                    keywords=result["keywords"],
                    metadata=doc_data.get("metadata", {}),
                )
                doc_memory.save()
                processed_count += 1

            return doc_id
        except json.JSONDecodeError as e:
            logger.error(f"文档 JSON 解析失败 {doc_id}: {e}")
            with lock:
                failed_count += 1
            return None
        except KeyError as e:
            logger.error(f"文档缺少必要字段 {doc_id}: {e}")
            with lock:
                failed_count += 1
            return None
        except Exception as e:
            logger.error(f"生成摘要失败 {doc_id}: {e}")
            with lock:
                failed_count += 1
            return None

    # 使用 ThreadPoolExecutor 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_doc, pf): pf
            for pf in files_to_process
        }
        with tqdm(total=len(files_to_process), desc="生成摘要") as pbar:
            for future in as_completed(futures):
                future.result()  # 异常已在 worker 内处理
                pbar.update(1)

    # 性能统计
    elapsed = time.monotonic() - start_time
    logger.info(
        f"摘要生成完成: {processed_count}/{len(files_to_process)} 个文档成功"
        f" (失败: {failed_count}), 耗时: {elapsed:.1f}s"
        f" (本次新处理: {processed_count})"
    )
    if failed_count > 0:
        logger.warning(
            f"有 {failed_count} 个文档处理失败，请检查日志获取详细信息"
        )
    return doc_memory


def _doc_id_from_path(path: Path) -> str:
    """从文件路径提取 doc_id"""
    return path.stem
