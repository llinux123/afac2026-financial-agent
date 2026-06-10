"""TXT 解析模块 - 处理纯文本格式法规文档

适用于从政府网站下载的纯文本文档，如：
- strict_v3_017_中华人民共和国反洗钱法.txt
- strict_v3_008_中国人民银行令〔2025〕第12号.txt
"""
import re
import json
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import logger


@dataclass
class ParsedTXTDocument:
    """解析后的 TXT 文档"""
    doc_id: str
    doc_type: str          # 固定为 "regulatory"
    title: str
    raw_text: str
    metadata: dict = field(default_factory=dict)
    source_file: str = ""

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "total_pages": 1,
            "raw_text": self.raw_text,
            "pages": [{"page_num": 1, "text": self.raw_text, "blocks": []}],
            "tables": [],
            "metadata": self.metadata,
        }


def _extract_title_from_filename(filename: str) -> str:
    """从文件名提取标题

    示例:
    strict_v3_017_中华人民共和国反洗钱法.txt → 中华人民共和国反洗钱法
    strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）.txt
      → 中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）
    """
    # 去除扩展名
    name = Path(filename).stem

    # 去除 strict_v3_NNN_ 前缀
    match = re.match(r'strict_v\d+_\d+_(.+)', name)
    if match:
        return match.group(1)

    return name


def _extract_title_from_content(text: str) -> str:
    """从文本内容提取标题

    法规文件通常在开头有标题行或括号注释。
    """
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if not line:
            continue
        # 跳过括号注释（如通过日期信息）
        if line.startswith('（') and '通过' in line:
            continue
        if line.startswith('('):
            continue
        # 跳过"目录"
        if line == '目录':
            continue
        # 第一个有意义的行通常是标题
        if len(line) >= 4:
            return line[:100]
    return ""


def _clean_txt_content(text: str) -> str:
    """清洗 TXT 内容"""
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        # 保留空行（段落分隔）
        if not line:
            cleaned.append('')
            continue
        cleaned.append(line)

    # 合并连续空行（最多保留一个）
    result = []
    prev_empty = False
    for line in cleaned:
        is_empty = not line
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return '\n'.join(result)


def _extract_txt_metadata(text: str, filename: str) -> dict:
    """从 TXT 内容和文件名提取元数据"""
    metadata = {}

    # 从文件名提取法规编号
    name = Path(filename).stem
    match = re.match(r'strict_v(\d+)_(\d+)', name)
    if match:
        metadata["version"] = match.group(1)
        metadata["serial"] = match.group(2)

    # 提取通过日期（常见于法规开头）
    date_pattern = re.compile(
        r'[（(](\d{4})年(\d{1,2})月(\d{1,2})日.*?[)）]'
    )
    match = date_pattern.search(text[:500])
    if match:
        metadata["pass_date"] = f"{match.group(1)}年{match.group(2)}月{match.group(3)}日"

    # 提取修订日期
    revise_pattern = re.compile(
        r'(\d{4})年(\d{1,2})月(\d{1,2})日.*?修订'
    )
    match = revise_pattern.search(text[:500])
    if match:
        metadata["revise_date"] = f"{match.group(1)}年{match.group(2)}月{match.group(3)}日"

    # 统计章节数
    chapters = re.findall(r'第[一二三四五六七八九十百]+章', text)
    articles = re.findall(r'第[一二三四五六七八九十百千\d]+条', text)
    metadata["chapter_count"] = len(set(chapters))
    metadata["article_count"] = len(set(articles))

    return metadata


def parse_single_txt(txt_path: Path, doc_id: str | None = None) -> ParsedTXTDocument:
    """解析单个 TXT 文件"""
    txt_path = Path(txt_path)
    if doc_id is None:
        doc_id = txt_path.stem

    logger.info(f"解析 TXT: {txt_path.name} (doc_id={doc_id})")

    # 尝试多种编码读取
    raw_text = ""
    for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
        try:
            raw_text = txt_path.read_text(encoding=encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if not raw_text:
        logger.error(f"无法读取文件（所有编码均失败）: {txt_path.name}")
        raw_text = ""

    # 提取标题
    title_from_file = _extract_title_from_filename(txt_path.name)
    title_from_content = _extract_title_from_content(raw_text)
    title = title_from_file if title_from_file else title_from_content

    # 清洗内容
    cleaned_text = _clean_txt_content(raw_text)

    # 提取元数据
    metadata = _extract_txt_metadata(raw_text, txt_path.name)

    result = ParsedTXTDocument(
        doc_id=doc_id,
        doc_type="regulatory",
        title=title,
        raw_text=cleaned_text,
        metadata=metadata,
        source_file=txt_path.name,
    )

    logger.info(
        f"  类型=regulatory, 文本长度={len(cleaned_text)}字, "
        f"标题={title[:50]}, "
        f"章={metadata.get('chapter_count', 0)}, "
        f"条={metadata.get('article_count', 0)}"
    )
    return result


def parse_all_txt(txt_dir: Path | None = None, force: bool = False) -> list[dict]:
    """批量解析目录下所有 TXT 文件

    断点续跑：检查 PARSED_DIR 中是否已有清洗后的结果，有则跳过。
    """
    import json
    from config.settings import RAW_DIR, PARSED_DIR
    from offline.text_cleaner import clean_text

    txt_dir = txt_dir or (RAW_DIR / "regulatory" / "txt")

    if not txt_dir.exists():
        logger.warning(f"TXT 目录不存在: {txt_dir}")
        return []

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"在 {txt_dir} 中未找到 TXT 文件")
        return []

    logger.info(f"发现 {len(txt_files)} 个 TXT 文件")

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    skipped = 0

    for txt_path in txt_files:
        doc_id = txt_path.stem
        output_path = PARSED_DIR / f"{doc_id}.json"

        # 断点续跑：检查是否已有清洗后的结果
        if not force and output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
                documents.append(doc_data)
                skipped += 1
                continue
            except Exception:
                pass  # 文件损坏，重新解析

        try:
            parsed = parse_single_txt(txt_path)
            # 清洗
            parsed.raw_text = clean_text(parsed.raw_text, parsed.doc_type)
            doc_data = parsed.to_dict()
            # 保存清洗后的结果
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
            documents.append(doc_data)
        except Exception as e:
            logger.error(f"TXT 解析失败 '{txt_path.name}': {e}")

    logger.info(
        f"TXT 解析完成: {len(documents)}/{len(txt_files)} 个文档"
        f" (跳过已有: {skipped}, 新解析: {len(documents) - skipped})"
    )
    return documents
