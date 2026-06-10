"""全局配置 - API key、模型参数、路径常量"""
import os
from pathlib import Path

# 加载 .env 文件中的环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"        # 阶段2: 清洗后的解析结果
CHUNKS_DIR = DATA_DIR / "chunks"         # 阶段3-4: 分块结果
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
SUBMISSION_DIR = DATA_DIR / "submission"

# ============================================================
# Qwen API 配置
# ============================================================
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_API_BASE = os.getenv(
    "QWEN_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# API 调用参数
QWEN_TEMPERATURE = 0.1          # 低温度保证确定性
QWEN_MAX_TOKENS = 4096          # 单次最大输出 token
QWEN_MAX_RETRIES = 3            # 重试次数
QWEN_RETRY_DELAY = 2.0          # 重试初始延迟(秒)

# Thinking 模式 (Qwen3 系列支持)
QWEN_ENABLE_THINKING = False    # 默认关闭，复杂题目可开启

# ============================================================
# Token 预算
# ============================================================
TOTAL_TOKEN_BUDGET = 5_000_000  # 总预算 500 万 token

# ============================================================
# 检索参数
# ============================================================
BM25_TOP_K = 20                 # BM25 召回数量
RERANK_TOP_K = 10               # 重排序数量
RERANK_TRIGGER_THRESHOLD = 0.3  # BM25 分差低于此值触发重排序
MAX_EVIDENCE_TOKENS = 15000     # 最大证据 token 数

# 文档定位参数 (B榜)
DOC_LOCATE_TOP_N = 5            # B榜候选文档数量

# ============================================================
# 分块参数
# ============================================================
CHUNK_SIZE_L2_MIN = 2000
CHUNK_SIZE_L2_MAX = 8000
CHUNK_SIZE_L3_MIN = 500
CHUNK_SIZE_L3_MAX = 2000
CHUNK_SIZE_L4_MAX = 500
CHUNK_OVERLAP = 100             # chunk 间重叠字符数

# ============================================================
# 文档类型
# ============================================================
DOC_TYPES = [
    "insurance",          # 保险条款
    "regulatory",         # 监管法规
    "financial_contracts",# 金融合同
    "financial_reports",  # 财务报表
    "research",           # 行业研报
]

# 文档类型中文名映射
DOC_TYPE_NAMES = {
    "insurance": "保险条款",
    "regulatory": "监管法规",
    "financial_contracts": "金融合同",
    "financial_reports": "财务报表",
    "research": "行业研报",
}

# ============================================================
# 压缩参数
# ============================================================
DEDUP_SIMILARITY_THRESHOLD = 0.7   # chunk 去重 Jaccard 阈值
SENTENCE_RELEVANCE_THRESHOLD = 0.1 # 句子相关性阈值
LONG_CHUNK_THRESHOLD = 1000        # 超过此长度的 chunk 触发摘要

# ============================================================
# 记忆参数
# ============================================================
QUERY_CACHE_SIMILARITY = 0.8       # 题目缓存命中阈值
FACT_TABLE_MAX_SIZE = 500          # 事实表最大条目数

# ============================================================
# 动态预算模式
# ============================================================
BUDGET_MODES = {
    "normal": {
        "max_evidence_tokens": 15000,
        "max_reasoning_tokens": 20000,
        "avg_per_question": 30000,
    },
    "compact": {
        "max_evidence_tokens": 10000,
        "max_reasoning_tokens": 15000,
        "avg_per_question": 20000,
    },
    "aggressive": {
        "max_evidence_tokens": 6000,
        "max_reasoning_tokens": 10000,
        "avg_per_question": 12000,
    },
    "minimal": {
        "max_evidence_tokens": 3000,
        "max_reasoning_tokens": 8000,
        "avg_per_question": 8000,
    },
}
