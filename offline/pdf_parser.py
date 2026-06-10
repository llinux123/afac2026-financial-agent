"""PDF 解析模块 - PyMuPDF 为主，pdfplumber 辅助表格提取"""
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from config.settings import PARSED_DIR, DOC_TYPES
from config.doc_type_config import DOC_TYPE_KEYWORDS
from utils.logger import logger


@dataclass
class PageData:
    """单页数据"""
    page_num: int
    text: str
    blocks: list[dict] = field(default_factory=list)  # 文本块(含坐标)


@dataclass
class TableData:
    """表格数据"""
    page_num: int
    table_id: str
    rows: list[list[str]] = field(default_factory=list)
    text: str = ""  # 表格的文本表示


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    doc_type: str
    title: str
    total_pages: int
    raw_text: str
    pages: list[PageData] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "total_pages": self.total_pages,
            "raw_text": self.raw_text,
            "pages": [asdict(p) for p in self.pages],
            "tables": [asdict(t) for t in self.tables],
        }


def _infer_doc_type_from_path(file_path: Path) -> str | None:
    """从文件所在目录推断文档类型

    目录名与 DOC_TYPES 匹配时直接返回，如：
    data/raw/insurance/1.pdf → "insurance"
    data/raw/financial_reports/xxx.PDF → "financial_reports"
    """
    from config.settings import DOC_TYPES
    for parent in [file_path.parent, file_path.parent.parent]:
        dir_name = parent.name.lower()
        if dir_name in DOC_TYPES:
            return dir_name
    return None


def detect_doc_type(text: str, filename: str, file_path: Path | None = None) -> str:
    """根据文本内容、文件名和目录路径自动检测文档类型

    优先级：目录名 > 关键词匹配
    """
    # 优先使用目录名
    if file_path is not None:
        dir_type = _infer_doc_type_from_path(file_path)
        if dir_type:
            return dir_type

    # 降级：关键词匹配
    scores = {}
    combined_text = text[:5000] + " " + filename

    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined_text)
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        logger.warning(f"无法确定文档类型 '{filename}'，默认为 research")
        return "research"
    return best_type


def extract_text_blocks(page: fitz.Page) -> list[dict]:
    """从页面提取文本块（含坐标信息，用于多栏检测）"""
    blocks = []
    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
        if block["type"] == 0:  # 文本块
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")
                text += "\n"
            blocks.append({
                "bbox": block["bbox"],  # (x0, y0, x1, y1)
                "text": text.strip(),
                "font_size": _get_avg_font_size(block),
            })
    return blocks


def _get_avg_font_size(block: dict) -> float:
    """获取文本块的平均字号"""
    sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(span.get("size", 12))
    return sum(sizes) / len(sizes) if sizes else 12.0


def sort_blocks_reading_order(blocks: list[dict]) -> list[dict]:
    """按阅读顺序排序文本块（处理多栏排版）

    策略：先按 y 坐标分行（y 差距小于阈值的视为同一行），
    同行内按 x 坐标从左到右排序。
    """
    if not blocks:
        return blocks

    Y_THRESHOLD = 50  # y 坐标差距小于此值视为同一行

    # 按 y 坐标排序
    sorted_blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    # 分组：y 坐标相近的块归为一行
    rows: list[list[dict]] = []
    current_row = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        if abs(block["bbox"][1] - current_row[-1]["bbox"][1]) < Y_THRESHOLD:
            current_row.append(block)
        else:
            # 行内按 x 坐标排序
            current_row.sort(key=lambda b: b["bbox"][0])
            rows.append(current_row)
            current_row = [block]

    if current_row:
        current_row.sort(key=lambda b: b["bbox"][0])
        rows.append(current_row)

    # 展平
    result = []
    for row in rows:
        result.extend(row)
    return result


def extract_tables_pdfplumber(pdf_path: str) -> list[TableData]:
    """用 pdfplumber 提取表格（适合财报/合同等表格密集文档）"""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for tbl_idx, table_rows in enumerate(page_tables):
                    if not table_rows:
                        continue

                    # 转为文本表示
                    text_lines = []
                    for row in table_rows:
                        cells = [str(cell or "").strip() for cell in row]
                        text_lines.append(" | ".join(cells))

                    tables.append(TableData(
                        page_num=page_idx + 1,
                        table_id=f"t{len(tables) + 1}",
                        rows=[[str(c or "").strip() for c in r] for r in table_rows],
                        text="\n".join(text_lines),
                    ))
    except Exception as e:
        logger.warning(f"pdfplumber 表格提取失败 '{pdf_path}': {e}")

    return tables


def parse_single_pdf(pdf_path: Path, doc_id: str | None = None) -> ParsedDocument:
    """解析单个 PDF 文件"""
    pdf_path = Path(pdf_path)
    if doc_id is None:
        doc_id = pdf_path.stem

    logger.info(f"解析 PDF: {pdf_path.name} (doc_id={doc_id})")

    doc = fitz.open(str(pdf_path))
    pages = []
    all_text_parts = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # 提取文本块（含坐标）
        blocks = extract_text_blocks(page)
        # 按阅读顺序排序（处理多栏）
        blocks = sort_blocks_reading_order(blocks)

        # 拼接页面文本
        page_text = "\n".join(b["text"] for b in blocks if b["text"])

        pages.append(PageData(
            page_num=page_idx + 1,
            text=page_text,
            blocks=[{"bbox": b["bbox"], "font_size": b["font_size"],
                      "text_preview": b["text"][:100]} for b in blocks],
        ))
        all_text_parts.append(page_text)

    doc.close()

    raw_text = "\n\n".join(all_text_parts)

    # 表格密集文档用 pdfplumber 额外提取表格
    tables = []
    if _is_table_heavy(raw_text, pdf_path.name):
        tables = extract_tables_pdfplumber(str(pdf_path))
        logger.info(f"  提取到 {len(tables)} 个表格")

    # 检测文档类型（优先使用目录名）
    doc_type = detect_doc_type(raw_text, pdf_path.name, file_path=pdf_path)

    # 提取标题（取前几行非空文本）
    title = _extract_title(raw_text)

    result = ParsedDocument(
        doc_id=doc_id,
        doc_type=doc_type,
        title=title,
        total_pages=len(pages),
        raw_text=raw_text,
        pages=pages,
        tables=tables,
    )

    logger.info(f"  类型={doc_type}, 页数={len(pages)}, "
                f"文本长度={len(raw_text)}字, 表格={len(tables)}个")
    return result


def _is_table_heavy(text: str, filename: str) -> bool:
    """判断文档是否表格密集"""
    # 通过关键词判断
    table_keywords = ["表格", "合计", "总计", "单位：", "万元", "亿元", "%"]
    score = sum(1 for kw in table_keywords if kw in text[:10000])
    if score >= 3:
        return True

    # 财报和合同类型默认提取表格
    fn_lower = filename.lower()
    if any(kw in fn_lower for kw in ["report", "financial", "contract", "财报", "合同"]):
        return True
    return False


def _extract_title(text: str) -> str:
    """从文本开头提取文档标题"""
    lines = text.strip().split("\n")
    for line in lines[:10]:
        line = line.strip()
        # 跳过空行和短行
        if len(line) < 4:
            continue
        # 跳过页眉类内容
        if re.match(r"^第\s*\d+\s*页", line):
            continue
        if re.match(r"^\d{4}\s*年", line):
            continue
        # 取第一个有意义的长行作为标题
        if len(line) >= 6:
            return line[:100]
    return "未知标题"


def _find_pdf_files(raw_dir: Path) -> list[Path]:
    """查找目录下所有 PDF 文件（支持 .pdf 和 .PDF 扩展名）"""
    pdf_files = []
    for pattern in ["**/*.pdf", "**/*.PDF", "**/*.Pdf"]:
        pdf_files.extend(raw_dir.glob(pattern))
    # 去重并排序
    return sorted(set(pdf_files))


def parse_all_pdfs(raw_dir: Path | None = None, force: bool = False) -> list[dict]:
    """批量解析所有 PDF 文件（支持大小写扩展名）

    断点续跑：检查 PARSED_DIR 中是否已有清洗后的结果，有则跳过。

    Args:
        raw_dir: PDF 文件所在目录
        force: 强制重新解析（忽略已有结果）

    Returns:
        解析后的文档字典列表
    """
    from config.settings import RAW_DIR
    from offline.text_cleaner import clean_text
    raw_dir = raw_dir or RAW_DIR

    pdf_files = _find_pdf_files(raw_dir)
    if not pdf_files:
        logger.error(f"在 {raw_dir} 中未找到 PDF 文件（已尝试 .pdf/.PDF）")
        return []

    logger.info(f"发现 {len(pdf_files)} 个 PDF 文件")

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    skipped = 0

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
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
            parsed = parse_single_pdf(pdf_path)
            # 清洗
            parsed.raw_text = clean_text(parsed.raw_text, parsed.doc_type)
            doc_data = parsed.to_dict()
            # 保存清洗后的结果
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)
            documents.append(doc_data)
        except Exception as e:
            logger.error(f"解析失败 '{pdf_path.name}': {e}")

    logger.info(
        f"PDF 解析完成: {len(documents)}/{len(pdf_files)} 个文档"
        f" (跳过已有: {skipped}, 新解析: {len(documents) - skipped})"
    )
    return documents


def load_parsed_document(doc_id: str) -> ParsedDocument | None:
    """加载已解析的文档"""
    path = PARSED_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc = ParsedDocument(
        doc_id=data["doc_id"],
        doc_type=data["doc_type"],
        title=data["title"],
        total_pages=data["total_pages"],
        raw_text=data["raw_text"],
    )
    return doc
