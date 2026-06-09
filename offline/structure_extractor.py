"""结构化元数据提取模块"""
import re
from dataclasses import dataclass, field, asdict
from config.doc_type_config import SECTION_PATTERNS
from utils.logger import logger


@dataclass
class Section:
    """章节节点"""
    id: str
    title: str
    level: int
    start_pos: int
    end_pos: int
    text: str = ""
    children: list["Section"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "text": self.text[:200],  # 只存前200字
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class DocumentStructure:
    """文档结构"""
    doc_id: str
    doc_type: str
    title: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
        }


def extract_structure(text: str, doc_id: str, doc_type: str) -> DocumentStructure:
    """从文本中提取结构化章节树"""
    title = _extract_title(text)
    metadata = extract_metadata(text, doc_type)

    # 根据文档类型选择正则模式
    patterns = SECTION_PATTERNS.get(doc_type, SECTION_PATTERNS.get("research", []))

    # 扫描所有行，匹配章节标记
    markers: list[tuple[int, str, int, str]] = []  # (pos, title, level, type)
    lines = text.split("\n")
    pos = 0

    for line in lines:
        stripped = line.strip()
        for pattern, level, sec_type in patterns:
            match = re.match(pattern, stripped)
            if match:
                if level <= 2:
                    section_title = match.group(1).strip() if match.lastindex >= 1 else stripped
                elif sec_type == "numbered" and match.lastindex >= 2:
                    section_title = match.group(2).strip()
                else:
                    section_title = stripped
                markers.append((pos, section_title, level, sec_type))
                break  # 只匹配第一个模式
        pos += len(line) + 1  # +1 for \n

    # 构建章节树
    sections = _build_section_tree(markers, text)

    return DocumentStructure(
        doc_id=doc_id,
        doc_type=doc_type,
        title=title,
        sections=sections,
        metadata=metadata,
    )


def _extract_title(text: str) -> str:
    """提取文档标题"""
    lines = text.strip().split("\n")
    for line in lines[:15]:
        line = line.strip()
        if len(line) < 4:
            continue
        if re.match(r"^第\s*\d+\s*页", line):
            continue
        if len(line) >= 6:
            return line[:100]
    return "未知标题"


def _build_section_tree(
    markers: list[tuple[int, str, int, str]],
    text: str,
) -> list[Section]:
    """从标记列表构建章节树"""
    if not markers:
        # 无结构标记时，按段落分段
        return _fallback_sectioning(text)

    sections = []
    total_len = len(text)

    for i, (pos, title, level, sec_type) in enumerate(markers):
        # 结束位置为下一个同级或更高级标记的位置，或文本末尾
        end_pos = total_len
        for j in range(i + 1, len(markers)):
            next_pos, _, next_level, _ = markers[j]
            if next_level <= level:
                end_pos = next_pos
                break

        section_text = text[pos:end_pos].strip()

        section = Section(
            id=f"s{i + 1}",
            title=title,
            level=level,
            start_pos=pos,
            end_pos=end_pos,
            text=section_text[:5000],  # 限制长度
        )

        # 尝试挂载到父节点
        if sections and level > 1:
            parent = _find_parent(sections, level)
            if parent:
                section.id = f"{parent.id}.{len(parent.children) + 1}"
                parent.children.append(section)
            else:
                sections.append(section)
        else:
            sections.append(section)

    return sections


def _find_parent(sections: list[Section], target_level: int) -> Section | None:
    """在已有章节中找到最近的父节点（level 比 target_level 小的）"""
    for section in reversed(sections):
        if section.level < target_level:
            # 递归查找子节点中的父节点
            if section.children:
                child_parent = _find_parent(section.children, target_level)
                if child_parent:
                    return child_parent
            return section
    return None


def _fallback_sectioning(text: str) -> list[Section]:
    """无结构标记时的降级分段策略"""
    paragraphs = re.split(r"\n\s*\n", text)
    sections = []
    pos = 0

    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 20:
            pos += len(para) + 2
            continue

        sections.append(Section(
            id=f"p{i + 1}",
            title=para.strip()[:50],
            level=2,
            start_pos=pos,
            end_pos=pos + len(para),
            text=para.strip()[:5000],
        ))
        pos += len(para) + 2

    return sections[:100]  # 限制数量


def extract_metadata(text: str, doc_type: str) -> dict:
    """提取文档元数据"""
    metadata: dict = {"doc_type": doc_type}

    # 通用实体提取
    metadata["companies"] = _extract_company_names(text)
    metadata["dates"] = _extract_dates(text)
    metadata["amounts"] = _extract_amounts(text)

    # 按类型提取
    if doc_type == "insurance":
        metadata["product"] = _extract_insurance_product(text)
        metadata["key_entities"] = ["投保人", "被保险人", "受益人", "保险金额", "保费"]
    elif doc_type == "regulatory":
        metadata["regulation_id"] = _extract_regulation_id(text)
    elif doc_type == "financial_reports":
        metadata["report_year"] = _extract_report_year(text)
        metadata["company"] = metadata["companies"][0] if metadata["companies"] else ""

    return metadata


def _extract_company_names(text: str) -> list[str]:
    """提取公司名称"""
    patterns = [
        r"([\u4e00-\u9fa5]{2,20}(?:股份有限|有限责任|有限)公司)",
        r"([\u4e00-\u9fa5]{2,15}集团)",
    ]
    companies = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text[:10000]):
            companies.add(match.group(1))
    return list(companies)[:10]


def _extract_dates(text: str) -> list[str]:
    """提取日期"""
    patterns = [
        r"(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)",
        r"(\d{4}\s*年\s*\d{1,2}\s*月)",
        r"(\d{4}\s*年)",
    ]
    dates = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text[:20000]):
            dates.add(match.group(1))
    return sorted(dates)[:20]


def _extract_amounts(text: str) -> list[str]:
    """提取金额"""
    pattern = r"([\d,，]+\.?\d*)\s*([万亿]元|%|‰|个基点)"
    amounts = []
    for match in re.finditer(pattern, text[:20000]):
        amounts.append(f"{match.group(1)}{match.group(2)}")
    return amounts[:30]


def _extract_insurance_product(text: str) -> str:
    """提取保险产品名称"""
    patterns = [
        r"《(.+?(?:保险|寿险|年金).*?)》",
        r"(.{5,30}(?:终身|定期|两全)(?:寿险|保险))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:5000])
        if match:
            return match.group(1)
    return ""


def _extract_regulation_id(text: str) -> str:
    """提取法规文号"""
    pattern = r"[\u4e00-\u9fa5]+[\[〔\(（]\d{4}[\]〕\)）]\s*\d+\s*号"
    match = re.search(pattern, text[:3000])
    return match.group(0) if match else ""


def _extract_report_year(text: str) -> str:
    """提取年报年份"""
    pattern = r"(\d{4})\s*年(?:度)?(?:年度)?(?:报告|年报)"
    match = re.search(pattern, text[:3000])
    return match.group(1) if match else ""
