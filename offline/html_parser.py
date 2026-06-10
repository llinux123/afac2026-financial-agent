"""HTML 解析模块 - 处理监管法规类 HTML 文档

适用于证监会(csrc)等政府网站导出的 HTML 页面。
提取策略：
1. 从 <meta> 标签提取元数据（标题、发布机构、发文日期等）
2. 从 <div class="detail-news"> 提取正文内容
3. 去除导航/页脚/脚本等无关区域
"""
import re
import html as html_module
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import logger


@dataclass
class ParsedHTMLDocument:
    """解析后的 HTML 文档"""
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
            "total_pages": 1,  # HTML 无分页概念
            "raw_text": self.raw_text,
            "pages": [{"page_num": 1, "text": self.raw_text, "blocks": []}],
            "tables": [],
            "metadata": self.metadata,
        }


def _strip_html_tags(html_str: str) -> str:
    """去除 HTML 标签，保留文本内容"""
    # 先把 <br> / <p> / <div> 等块级标签转为换行
    text = re.sub(r'<br\s*/?>', '\n', html_str, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</h[1-6]>', '\n', text, flags=re.IGNORECASE)
    # 去除其余所有标签
    text = re.sub(r'<[^>]+>', '', text)
    # HTML 实体解码
    text = html_module.unescape(text)
    # 规范化空白
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def _extract_meta_from_head(html_str: str) -> dict:
    """从 <head> 中的 <meta> 标签提取元数据"""
    meta = {}
    # meta name="xxx" content="yyy"
    pattern = re.compile(
        r'<meta\s+name=["\']([^"\']+)["\']\s+content=["\']([^"\']*)["\']',
        re.IGNORECASE,
    )
    for match in pattern.finditer(html_str[:5000]):
        name = match.group(1).strip()
        content = match.group(2).strip()
        if content:
            meta[name] = content

    # 提取 <title>
    title_match = re.search(r'<title>([^<]+)</title>', html_str[:3000], re.IGNORECASE)
    if title_match:
        meta["_title"] = html_module.unescape(title_match.group(1).strip())

    return meta


def _extract_metadata_table(html_str: str) -> dict:
    """提取 HTML 中的信息表格（索 引 号、发布机构、发文日期等）"""
    meta = {}
    # 证监会页面有 <table> 包含 th/td 对
    table_match = re.search(
        r'<div class="xxgk-table">(.*?)</div>',
        html_str,
        re.DOTALL | re.IGNORECASE,
    )
    if not table_match:
        return meta

    table_html = table_match.group(1)
    # 提取 th-td 对
    pairs = re.findall(
        r'<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>',
        table_html,
        re.DOTALL | re.IGNORECASE,
    )
    for th_html, td_html in pairs:
        key = _strip_html_tags(th_html).replace(' ', '').replace('\u3000', '').strip()
        value = _strip_html_tags(td_html).strip()
        if key and value:
            meta[key] = value

    return meta


def _extract_main_content(html_str: str) -> str:
    """提取正文主体内容

    证监会页面的正文在 <div class="detail-news"> 内。
    也兼容其他常见 class 名。
    """
    # 按优先级尝试多个内容容器
    content_patterns = [
        r'<div\s+class="detail-news"[^>]*>(.*)',       # 证监会主内容区
        r'<div\s+class="article-content"[^>]*>(.*)',   # 通用文章区
        r'<div\s+class="content"[^>]*>(.*)',           # 通用 content
        r'<div\s+id="content"[^>]*>(.*)',              # id=content
        r'<article[^>]*>(.*)',                          # <article> 标签
    ]

    for pattern in content_patterns:
        match = re.search(pattern, html_str, re.DOTALL | re.IGNORECASE)
        if match:
            content_html = match.group(1)
            # 截断到合理的结束标记
            # 尝试找到对应的结束 div
            content_html = _truncate_to_end(content_html)
            text = _strip_html_tags(content_html)
            # 过滤掉太短的结果（可能是误匹配）
            if len(text) > 100:
                return text

    # 降级：对整个 body 提取
    body_match = re.search(r'<body[^>]*>(.*)</body>', html_str, re.DOTALL | re.IGNORECASE)
    if body_match:
        text = _strip_html_tags(body_match.group(1))
        return text

    return _strip_html_tags(html_str)


def _truncate_to_end(html_str: str) -> str:
    """截断 HTML 到合理的结束位置

    去除页脚、脚本等后续内容。
    """
    # 在这些标记处截断
    end_markers = [
        r'<div\s+class="xxgk-down-box"',     # 下载/打印按钮区
        r'<div\s+class="foot',                # 页脚
        r'<div\s+class="footsj"',
        r'<script\s',                          # 脚本开始
        r'<!--.*?-->',                         # 注释
    ]
    earliest_end = len(html_str)
    for marker in end_markers:
        match = re.search(marker, html_str, re.IGNORECASE)
        if match and match.start() < earliest_end:
            earliest_end = match.start()

    return html_str[:earliest_end]


def _clean_extracted_text(text: str) -> str:
    """清洗提取后的文本"""
    lines = text.split('\n')
    cleaned = []

    # 导航/搜索相关噪声关键词
    noise_patterns = [
        r'当前位置[：:]',
        r'首页\s*>',
        r'请输入关键字',
        r'索\s*引\s*号',
        r'分\s*类',
        r'发布机构',
        r'发文日期',
        r'名\s*称',
        r'文\s*号',
        r'主\s*题\s*词',
        r'相关链接',
        r'【打印】',
        r'【关闭窗口】',
        r'主办.*?证监会',
        r'版权所有',
        r'网站识别码',
        r'京ICP备',
        r'京公网安备',
        r'联系我们',
        r'法律声明',
    ]
    compiled_noise = [re.compile(p) for p in noise_patterns]

    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append('')
            continue
        # 跳过噪声行
        if any(p.search(line) for p in compiled_noise):
            continue
        # 跳过太短的无意义行
        if len(line) < 2:
            continue
        cleaned.append(line)

    # 去除连续空行
    result = []
    prev_empty = False
    for line in cleaned:
        is_empty = not line
        if is_empty and prev_empty:
            continue
        result.append(line)
        prev_empty = is_empty

    return '\n'.join(result)


def parse_single_html(html_path: Path, doc_id: str | None = None) -> ParsedHTMLDocument:
    """解析单个 HTML 文件"""
    html_path = Path(html_path)
    if doc_id is None:
        doc_id = html_path.stem  # 如 csrc_0001

    logger.info(f"解析 HTML: {html_path.name} (doc_id={doc_id})")

    raw_html = html_path.read_text(encoding='utf-8', errors='ignore')

    # 1. 提取元数据
    head_meta = _extract_meta_from_head(raw_html)
    table_meta = _extract_metadata_table(raw_html)
    metadata = {**head_meta, **table_meta}

    # 2. 提取标题
    title = ""
    # 优先从 meta ArticleTitle 获取
    if "ArticleTitle" in metadata:
        title = metadata["ArticleTitle"]
    elif "_title" in metadata:
        title = metadata["_title"]
    else:
        # 从 h2 标签获取
        h2_match = re.search(r'<h2[^>]*>([^<]+)</h2>', raw_html, re.IGNORECASE)
        if h2_match:
            title = html_module.unescape(h2_match.group(1).strip())

    # 3. 提取正文
    raw_text = _extract_main_content(raw_html)
    raw_text = _clean_extracted_text(raw_text)

    # 4. 检测是否有 PDF 附件
    has_pdf_attachment = bool(
        re.search(r'files/[^"\']+\.pdf', raw_html, re.IGNORECASE)
    )
    metadata["has_pdf_attachment"] = has_pdf_attachment

    # 5. 清理元数据中的内部字段
    metadata.pop("_title", None)
    # 只保留有意义的元数据
    clean_meta = {}
    skip_keys = {"viewport", "X-UA-Compatible", "template,templategroup,version",
                 "others", "SiteIDCode", "ColumnType", "ColumnDescription",
                 "ColumnKeywords"}
    for k, v in metadata.items():
        if k not in skip_keys and v:
            clean_meta[k] = v

    result = ParsedHTMLDocument(
        doc_id=doc_id,
        doc_type="regulatory",
        title=title,
        raw_text=raw_text,
        metadata=clean_meta,
        source_file=html_path.name,
    )

    logger.info(
        f"  类型=regulatory, 文本长度={len(raw_text)}字, "
        f"标题={title[:50]}, PDF附件={'有' if has_pdf_attachment else '无'}"
    )
    return result


def parse_all_html(html_dir: Path | None = None) -> list[ParsedHTMLDocument]:
    """批量解析目录下所有 HTML 文件"""
    from config.settings import RAW_DIR
    html_dir = html_dir or (RAW_DIR / "regulatory" / "html")

    if not html_dir.exists():
        logger.error(f"HTML 目录不存在: {html_dir}")
        return []

    html_files = sorted(html_dir.glob("*.html"))
    if not html_files:
        logger.warning(f"在 {html_dir} 中未找到 HTML 文件")
        return []

    logger.info(f"发现 {len(html_files)} 个 HTML 文件")

    from config.settings import PARSED_DIR
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    import json
    documents = []
    for html_path in html_files:
        try:
            parsed = parse_single_html(html_path)
            # 保存解析结果
            output_path = PARSED_DIR / f"{parsed.doc_id}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(parsed.to_dict(), f, ensure_ascii=False, indent=2)
            documents.append(parsed)
        except Exception as e:
            logger.error(f"HTML 解析失败 '{html_path.name}': {e}")

    logger.info(f"成功解析 {len(documents)}/{len(html_files)} 个 HTML 文档")
    return documents
