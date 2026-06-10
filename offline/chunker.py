"""多层次文档分块引擎"""
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config.settings import (
    CHUNK_SIZE_L2_MIN, CHUNK_SIZE_L2_MAX,
    CHUNK_SIZE_L3_MIN, CHUNK_SIZE_L3_MAX,
    CHUNK_SIZE_L4_MAX, CHUNK_OVERLAP,
    CHUNKS_DIR,
)
from config.doc_type_config import CHUNK_STRATEGIES, SECTION_PATTERNS
from offline.structure_extractor import DocumentStructure, Section
from utils.logger import logger


@dataclass
class Chunk:
    """文档分块"""
    chunk_id: str
    doc_id: str
    doc_type: str
    level: str              # L2/L3/L4
    section_path: str       # 章节路径 (如 "第一章 总则 > 第一条 合同构成")
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunker:
    """多层次分块引擎"""

    def __init__(self):
        self.chunks: list[Chunk] = []

    def chunk_document(
        self,
        text: str,
        structure: DocumentStructure,
        doc_id: str,
        doc_type: str,
    ) -> list[Chunk]:
        """对单个文档执行多层次分块"""
        strategy = CHUNK_STRATEGIES.get(doc_type, CHUNK_STRATEGIES["research"])
        self.chunks = []

        # L2: 章节级分块
        l2_chunks = self._chunk_by_sections(text, structure, doc_id, doc_type, "L2")
        self.chunks.extend(l2_chunks)

        # L3: 段落/条款级分块
        l3_chunks = self._chunk_by_articles(text, structure, doc_id, doc_type, "L3")
        self.chunks.extend(l3_chunks)

        # L4: 句子级分块 (仅对有结构的段落进一步切分)
        l4_chunks = self._chunk_sentences(text, structure, doc_id, doc_type)
        self.chunks.extend(l4_chunks)

        logger.info(
            f"  分块完成: L2={len(l2_chunks)}, L3={len(l3_chunks)}, "
            f"L4={len(l4_chunks)}, 总计={len(self.chunks)}"
        )
        return self.chunks

    def _chunk_by_sections(
        self, text: str, structure: DocumentStructure,
        doc_id: str, doc_type: str, level: str,
    ) -> list[Chunk]:
        """L2: 按顶级章节分块"""
        chunks = []
        if not structure.sections:
            # 无结构时按固定长度切分
            return self._chunk_by_length(text, doc_id, doc_type, level,
                                          CHUNK_SIZE_L2_MIN, CHUNK_SIZE_L2_MAX)

        for section in structure.sections:
            section_text = text[section.start_pos:section.end_pos]
            if len(section_text) < 100:
                continue

            # 如果章节太长，按子章节拆分
            if len(section_text) > CHUNK_SIZE_L2_MAX and section.children:
                child_texts = []
                for child in section.children:
                    child_text = text[child.start_pos:child.end_pos]
                    child_texts.append((child.title, child_text))

                # 合并子章节到 L2 大小
                current_text = ""
                current_titles = []
                for title, ct in child_texts:
                    if len(current_text) + len(ct) > CHUNK_SIZE_L2_MAX and current_text:
                        chunks.append(Chunk(
                            chunk_id=f"{doc_id}_{level}_{len(chunks) + 1}",
                            doc_id=doc_id,
                            doc_type=doc_type,
                            level=level,
                            section_path=section.title,
                            text=current_text,
                            metadata={"section": section.title,
                                      "subsections": current_titles},
                        ))
                        current_text = ct
                        current_titles = [title]
                    else:
                        current_text += "\n" + ct
                        current_titles.append(title)

                if current_text:
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_{level}_{len(chunks) + 1}",
                        doc_id=doc_id,
                        doc_type=doc_type,
                        level=level,
                        section_path=section.title,
                        text=current_text,
                        metadata={"section": section.title},
                    ))
            else:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{level}_{len(chunks) + 1}",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    level=level,
                    section_path=section.title,
                    text=section_text,
                    metadata={"section": section.title},
                ))

        return chunks

    def _chunk_by_articles(
        self, text: str, structure: DocumentStructure,
        doc_id: str, doc_type: str, level: str,
    ) -> list[Chunk]:
        """L3: 按条款/段落级分块"""
        chunks = []

        # 递归收集所有叶子节点（最细粒度的章节）
        leaf_sections = []
        self._collect_leaves(structure.sections, leaf_sections, [])

        for i, (section, path_parts) in enumerate(leaf_sections):
            section_text = text[section.start_pos:section.end_pos]
            if len(section_text) < 50:
                continue

            section_path = " > ".join(path_parts + [section.title])

            # 如果条款太长，进一步切分
            if len(section_text) > CHUNK_SIZE_L3_MAX:
                sub_chunks = self._split_text(
                    section_text, CHUNK_SIZE_L3_MAX, CHUNK_OVERLAP
                )
                for j, sub_text in enumerate(sub_chunks):
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_{level}_{len(chunks) + 1}",
                        doc_id=doc_id,
                        doc_type=doc_type,
                        level=level,
                        section_path=section_path,
                        text=sub_text,
                        metadata={
                            "section_path": section_path,
                            "has_numbers": bool(re.search(r"\d", sub_text)),
                            "part": j + 1,
                        },
                    ))
            else:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_{level}_{len(chunks) + 1}",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    level=level,
                    section_path=section_path,
                    text=section_text,
                    metadata={
                        "section_path": section_path,
                        "has_numbers": bool(re.search(r"\d", section_text)),
                    },
                ))

        # 如果没有结构化分块结果，使用文本直接分块
        if not chunks:
            chunks = self._chunk_by_length(
                text, doc_id, doc_type, level,
                CHUNK_SIZE_L3_MIN, CHUNK_SIZE_L3_MAX,
            )

        return chunks

    def _chunk_sentences(
        self, text: str, structure: DocumentStructure,
        doc_id: str, doc_type: str,
    ) -> list[Chunk]:
        """L4: 从较长的 L3 chunk 中提取句子级分块"""
        chunks = []
        # 只对 L3 中较长的 chunk 做句子级切分
        l3_chunks = [c for c in self.chunks if c.level == "L3" and c.char_count > 300]

        for l3 in l3_chunks:
            sentences = _split_sentences(l3.text)
            for j, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) < 30:
                    continue
                if len(sent) > CHUNK_SIZE_L4_MAX:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_L4_{len(chunks) + 1}",
                    doc_id=doc_id,
                    doc_type=doc_type,
                    level="L4",
                    section_path=l3.section_path,
                    text=sent,
                    metadata={
                        "parent_chunk": l3.chunk_id,
                        "has_numbers": bool(re.search(r"\d", sent)),
                    },
                ))

        return chunks[:500]  # 限制 L4 数量

    def _collect_leaves(
        self, sections: list[Section],
        result: list[tuple[Section, list[str]]],
        parent_path: list[str],
    ):
        """递归收集叶子节点"""
        for section in sections:
            if section.children:
                self._collect_leaves(
                    section.children, result,
                    parent_path + [section.title],
                )
            else:
                result.append((section, parent_path))

    def _chunk_by_length(
        self, text: str, doc_id: str, doc_type: str, level: str,
        min_size: int, max_size: int,
    ) -> list[Chunk]:
        """按固定长度分块（降级策略）"""
        chunks = []
        parts = self._split_text(text, max_size, CHUNK_OVERLAP)
        for i, part in enumerate(parts):
            if len(part) < min_size and i > 0:
                # 合并过短的块到前一个
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_{level}_{len(chunks) + 1}",
                doc_id=doc_id,
                doc_type=doc_type,
                level=level,
                section_path="",
                text=part,
                metadata={},
            ))
        return chunks

    def _split_text(self, text: str, max_size: int, overlap: int) -> list[str]:
        """将文本按 max_size 切分，保留 overlap 重叠"""
        if len(text) <= max_size:
            return [text]

        parts = []
        start = 0
        while start < len(text):
            end = start + max_size
            # 尝试在句号处断开
            if end < len(text):
                for punct in ["。", "；", "！", "？", "\n\n", "\n"]:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > max_size * 0.5:
                        end = start + last_punct + len(punct)
                        break

            parts.append(text[start:end])
            start = end - overlap

        return parts


def _split_sentences(text: str) -> list[str]:
    """中文句子切分"""
    # 按中文句末标点切分，保留标点
    pattern = r"(?<=[。！？；])"
    sentences = re.split(pattern, text)
    return [s for s in sentences if s.strip()]


def chunk_all_documents(
    parsed_docs: list[dict],
    structures: dict[str, DocumentStructure],
    force: bool = False,
) -> dict[str, list[Chunk]]:
    """对所有文档执行分块

    断点续跑：检查 CHUNKS_DIR 中是否已有分块结果，有则跳过。
    每个文档分块后立即保存到 CHUNKS_DIR/{doc_id}.json。
    """
    chunker = Chunker()
    all_chunks: dict[str, list[Chunk]] = {}
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    skipped = 0

    for doc_data in parsed_docs:
        doc_id = doc_data["doc_id"]
        doc_type = doc_data["doc_type"]
        chunks_path = CHUNKS_DIR / f"{doc_id}.json"

        # 断点续跑：检查是否已有分块结果
        if not force and chunks_path.exists():
            try:
                chunks = _load_chunks_from_file(chunks_path)
                if chunks:
                    all_chunks[doc_id] = chunks
                    skipped += 1
                    continue
            except Exception:
                pass  # 文件损坏，重新分块

        text = doc_data["raw_text"]
        structure = structures.get(doc_id)

        if not structure:
            logger.warning(f"文档 {doc_id} 无结构信息，使用降级分块")
            from offline.structure_extractor import DocumentStructure as DS
            structure = DS(doc_id=doc_id, doc_type=doc_type, title="")

        chunks = chunker.chunk_document(text, structure, doc_id, doc_type)
        all_chunks[doc_id] = chunks

        # 立即持久化
        _save_chunks(chunks, chunks_path)

    total = sum(len(v) for v in all_chunks.values())
    logger.info(
        f"分块完成: {len(all_chunks)} 个文档, {total} 个 chunks"
        f" (跳过已有: {skipped}, 新分块: {len(all_chunks) - skipped})"
    )
    return all_chunks


def _save_chunks(chunks: list[Chunk], path: Path):
    """保存分块结果到 JSON 文件"""
    data = [c.to_dict() for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_chunks_from_file(path: Path) -> list[Chunk]:
    """从 JSON 文件加载分块结果"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Chunk(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            doc_type=d["doc_type"],
            level=d["level"],
            section_path=d["section_path"],
            text=d["text"],
            metadata=d.get("metadata", {}),
        )
        for d in data
    ]


def load_all_chunks(chunks_dir: Path | None = None) -> dict[str, list[Chunk]]:
    """从 CHUNKS_DIR 加载所有已保存的分块结果"""
    chunks_dir = chunks_dir or CHUNKS_DIR
    all_chunks: dict[str, list[Chunk]] = {}

    if not chunks_dir.exists():
        return all_chunks

    for path in sorted(chunks_dir.glob("*.json")):
        try:
            chunks = _load_chunks_from_file(path)
            if chunks:
                doc_id = chunks[0].doc_id
                all_chunks[doc_id] = chunks
        except Exception as e:
            logger.warning(f"加载分块失败 {path.name}: {e}")

    total = sum(len(v) for v in all_chunks.values())
    logger.info(f"从磁盘加载分块: {len(all_chunks)} 个文档, {total} 个 chunks")
    return all_chunks
