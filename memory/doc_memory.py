"""文档级记忆 - 三层摘要存储"""
import json
from pathlib import Path

from config.settings import CACHE_DIR
from utils.logger import logger


class DocMemory:
    """文档级记忆管理器"""

    def __init__(self):
        self.memories: dict[str, dict] = {}

    def add_document(
        self,
        doc_id: str,
        l1_summary: str = "",
        l2_summary: str = "",
        l3_summary: str = "",
        keywords: list[str] | None = None,
        metadata: dict | None = None,
    ):
        """添加文档记忆"""
        self.memories[doc_id] = {
            "l1_summary": l1_summary,
            "l2_summary": l2_summary,
            "l3_summary": l3_summary,
            "keywords": keywords or [],
            "metadata": metadata or {},
        }

    def get_summary(self, doc_id: str, level: int = 1) -> str:
        """获取指定层级的摘要"""
        mem = self.memories.get(doc_id, {})
        if level == 1:
            return mem.get("l1_summary", "")
        elif level == 2:
            return mem.get("l2_summary", "")
        else:
            return mem.get("l3_summary", "")

    def get_keywords(self, doc_id: str) -> list[str]:
        """获取文档关键词"""
        return self.memories.get(doc_id, {}).get("keywords", [])

    def get_all_summaries(self) -> dict[str, dict]:
        """获取所有文档摘要 (用于 B 榜文档定位)"""
        result = {}
        for doc_id, mem in self.memories.items():
            result[doc_id] = {
                "l1_summary": mem["l1_summary"],
                "l2_summary": mem["l2_summary"],
                "keywords": mem["keywords"],
                "metadata": mem.get("metadata", {}),
            }
        return result

    def save(self, path: str | None = None):
        """持久化记忆"""
        path = path or str(CACHE_DIR / "doc_memory.json")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
        logger.info(f"文档记忆已保存: {path} ({len(self.memories)} 个文档)")

    def load(self, path: str | None = None):
        """加载记忆"""
        path = path or str(CACHE_DIR / "doc_memory.json")
        p = Path(path)
        if not p.exists():
            logger.warning(f"文档记忆文件不存在: {path}")
            return
        with open(p, "r", encoding="utf-8") as f:
            self.memories = json.load(f)
        logger.info(f"文档记忆已加载: {len(self.memories)} 个文档")
