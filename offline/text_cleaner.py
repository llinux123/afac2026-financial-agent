"""文本清洗模块 - 去噪、规范化"""
import re
from config.doc_type_config import HEADER_FOOTER_PATTERNS, DISCLAIMER_PATTERNS
from utils.logger import logger


def clean_text(text: str, doc_type: str = "") -> str:
    """完整的文本清洗流水线"""
    text = remove_headers_footers(text)
    text = remove_watermarks(text)
    text = merge_broken_lines(text)
    text = normalize_whitespace(text)
    text = normalize_punctuation(text)
    if doc_type == "research":
        text = remove_disclaimers(text)
    text = remove_empty_lines(text)
    return text


def remove_headers_footers(text: str) -> str:
    """去除页眉页脚（重复模式检测 + 正则匹配）"""
    lines = text.split("\n")
    if len(lines) < 10:
        return text

    # 方法1: 正则匹配已知的页眉页脚模式
    compiled_patterns = [re.compile(p) for p in HEADER_FOOTER_PATTERNS]
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if any(p.match(stripped) for p in compiled_patterns):
            continue
        cleaned_lines.append(line)

    # 方法2: 统计高频出现的短行（可能是页眉页脚）
    line_counts: dict[str, int] = {}
    for line in cleaned_lines:
        stripped = line.strip()
        if 3 < len(stripped) < 60:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # 出现超过5次的短行很可能是页眉/页脚
    frequent_lines = {
        line for line, count in line_counts.items()
        if count >= 5 and _looks_like_header_footer(line)
    }

    if frequent_lines:
        cleaned_lines = [
            line for line in cleaned_lines
            if line.strip() not in frequent_lines
        ]

    return "\n".join(cleaned_lines)


def _looks_like_header_footer(line: str) -> bool:
    """判断一行是否像页眉/页脚"""
    # 纯数字或页码格式
    if re.match(r"^\d+$", line):
        return True
    if re.match(r"^\d+\s*/\s*\d+$", line):
        return True
    # 包含"第X页"
    if re.search(r"第\s*\d+\s*页", line):
        return True
    # 包含 Copyright 或 版权
    if "copyright" in line.lower() or "版权" in line:
        return True
    return False


def remove_watermarks(text: str) -> str:
    """去除水印文本（全文重复出现的短字符串）"""
    lines = text.split("\n")
    if len(lines) < 20:
        return text

    # 统计每行出现次数
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 2 < len(stripped) < 30:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # 出现超过10次的短行可能是水印
    watermark_threshold = max(10, len(lines) // 10)
    watermarks = {
        line for line, count in line_counts.items()
        if count >= watermark_threshold and len(line) < 20
    }

    if not watermarks:
        return text

    logger.debug(f"检测到水印: {watermarks}")
    cleaned_lines = [line for line in lines if line.strip() not in watermarks]
    return "\n".join(cleaned_lines)


def merge_broken_lines(text: str) -> str:
    """合并中文PDF中常见的行尾断句问题

    规则：如果当前行不以句号/分号/问号/感叹号结尾，
    且下一行不以数字/空格/特殊标记开头，则合并两行。
    """
    lines = text.split("\n")
    if len(lines) < 2:
        return text

    merged = [lines[0]]
    # 中文句末标点
    end_puncts = set("。；！？；：）】」』")
    # 列表/标题开头标记
    list_markers = re.compile(
        r"^[\s ]*([一二三四五六七八九十\d]+[、．.）)]|"
        r"[（(][一二三四五六七八九十\d]+[）)]|"
        r"[①②③④⑤⑥⑦⑧⑨⑩]|"
        r"第[一二三四五六七八九十\d]+[章节条])"
    )

    for i in range(1, len(lines)):
        prev = merged[-1]
        curr = lines[i]

        # 空行不合并
        if not prev.strip() or not curr.strip():
            merged.append(curr)
            continue

        prev_last = prev.rstrip()[-1] if prev.rstrip() else ""
        # 前一行以句末标点结尾，不合并
        if prev_last in end_puncts:
            merged.append(curr)
            continue
        # 当前行以列表标记开头，不合并
        if list_markers.match(curr.strip()):
            merged.append(curr)
            continue
        # 前一行太短（标题），不合并
        if len(prev.strip()) < 10:
            merged.append(curr)
            continue
        # 当前行以表格格式开头，不合并
        if curr.strip().startswith("|") or curr.strip().startswith("┃"):
            merged.append(curr)
            continue

        # 合并：直接拼接（中文无需空格）
        merged[-1] = prev + curr

    return "\n".join(merged)


def normalize_whitespace(text: str) -> str:
    """规范化空白字符"""
    # 多个空格替换为单个空格
    text = re.sub(r"[ \t]+", " ", text)
    # 全角空格替换为半角空格
    text = text.replace("\u3000", " ")
    # 去除行首行尾空格
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def normalize_punctuation(text: str) -> str:
    """统一标点符号（保留中文标点）"""
    # 统一括号（英文括号保留在数字/公式中，其他转中文）
    # 这里不做激进的转换，只做基本统一
    # 保留原文的标点风格
    return text


def remove_disclaimers(text: str) -> str:
    """去除研报中的免责声明段落"""
    compiled = [re.compile(p) for p in DISCLAIMER_PATTERNS]

    lines = text.split("\n")
    result = []
    skip_mode = False

    for line in lines:
        stripped = line.strip()
        # 检测免责声明开始
        if any(p.search(stripped) for p in compiled):
            skip_mode = True
            continue
        # 免责声明通常连续多段，直到遇到新标题才结束
        if skip_mode:
            # 遇到新的章节标题则结束跳过
            if re.match(r"^[一二三四五六七八九十\d]+[、．.]", stripped):
                skip_mode = False
            elif len(stripped) == 0:
                # 空行可能是段落间隔，继续跳过
                continue
            else:
                continue

        result.append(line)

    return "\n".join(result)


def remove_empty_lines(text: str) -> str:
    """去除连续空行（保留最多一个空行）"""
    lines = text.split("\n")
    result = []
    prev_empty = False

    for line in lines:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return "\n".join(result)
