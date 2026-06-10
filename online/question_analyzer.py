"""题目分析模块 - 纯规则实现，0 Token 消耗"""
import re
from dataclasses import dataclass, field


# 中文停用词 (精简版)
_STOP_WORDS = set(
    "的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 "
    "说 要 去 你 会 着 没有 看 好 自己 这 那 她 他 它 们 "
    "为 与 及 或 而 但 之 以 于 从 对 等 将 被 把 向 比 "
    "所 如 果 能 可 并 且 该 此 其 更 最 已 经 还 又 "
    "中 时 后 前 下列 以下 哪项 哪些 哪个 关于 根据 "
    "正确 错误 准确 描述 判断 说法 选项".split()
)

# 文档类型关键词
_DOC_TYPE_HINTS = {
    "insurance": ["保险", "条款", "投保", "理赔", "保费", "被保险人", "等待期", "退保"],
    "regulatory": ["法规", "规定", "管理办法", "监管", "合规", "处罚", "条例"],
    "financial_contracts": ["债券", "合同", "募集", "发行人", "票面", "兑付"],
    "financial_reports": ["年报", "年度报告", "营业收入", "净利润", "财报", "报表"],
    "research": ["研报", "研究报告", "行业分析", "市场规模", "增长率"],
}


@dataclass
class AnalyzedQuestion:
    """分析后的题目"""
    qid: str
    question_type: str          # single / multi / judge
    question: str
    options: dict[str, str]
    answer_format: str          # mcq / multi / tf
    domain: str                 # 文档领域
    doc_ids: list[str]          # 关联文档 (A榜有，B榜空)

    # 分析结果
    target_doc_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    option_keywords: dict[str, list[str]] = field(default_factory=dict)
    entities: dict[str, str] = field(default_factory=dict)
    intent_type: str = "fact_lookup"
    search_queries: list[str] = field(default_factory=list)

    # 原始数据
    raw: dict = field(default_factory=dict)


def analyze_question(question_data: dict) -> AnalyzedQuestion:
    """分析单道题目 (纯规则，不调用 LLM)"""
    qid = question_data.get("qid", "unknown")
    question = question_data.get("question", "")
    options = question_data.get("options", {})
    answer_format = question_data.get("answer_format", "mcq")
    domain = question_data.get("domain", "")
    doc_ids = question_data.get("doc_ids", [])

    # 1. 题型识别
    question_type = _detect_question_type(question, answer_format)

    # 2. 文档类型推断
    target_doc_types = _infer_doc_types(question, options, domain, doc_ids)

    # 3. 关键词提取
    keywords = _extract_keywords(question)
    option_keywords = {
        opt: _extract_keywords(text) for opt, text in options.items()
    }

    # 4. 实体识别
    entities = _extract_entities(question, options)

    # 5. 问题意图分类
    intent_type = _classify_intent(question)

    # 6. 构建检索查询
    search_queries = _build_search_queries(question, options, keywords)

    return AnalyzedQuestion(
        qid=qid,
        question_type=question_type,
        question=question,
        options=options,
        answer_format=answer_format,
        domain=domain,
        doc_ids=doc_ids,
        target_doc_types=target_doc_types,
        keywords=keywords,
        option_keywords=option_keywords,
        entities=entities,
        intent_type=intent_type,
        search_queries=search_queries,
        raw=question_data,
    )


def _detect_question_type(question: str, answer_format: str) -> str:
    """识别题型"""
    # 优先使用 answer_format 字段
    if answer_format == "multi":
        return "multi"
    elif answer_format == "tf":
        return "judge"
    elif answer_format == "mcq":
        return "single"

    # 通过文本规则判断
    multi_hints = ["哪些", "哪几项", "正确的有", "错误的有", "符合.*的有"]
    for hint in multi_hints:
        if re.search(hint, question):
            return "multi"

    judge_hints = ["是否正确", "是否正确", "对还是错", "判断"]
    for hint in judge_hints:
        if hint in question:
            return "judge"

    return "single"


def _infer_doc_types(
    question: str, options: dict, domain: str, doc_ids: list,
) -> list[str]:
    """推断目标文档类型"""
    # 如果有 domain 字段，直接使用
    if domain:
        return [domain]

    # 如果有 doc_ids，从中推断
    if doc_ids:
        types = set()
        for did in doc_ids:
            for dt in ["insurance", "regulatory", "financial_contracts",
                        "financial_reports", "research"]:
                if dt in did or dt.replace("_", "") in did:
                    types.add(dt)
        if types:
            return list(types)

    # 通过关键词推断
    combined = question + " " + " ".join(options.values())
    scores = {}
    for doc_type, hints in _DOC_TYPE_HINTS.items():
        scores[doc_type] = sum(1 for h in hints if h in combined)

    top_types = sorted(scores.items(), key=lambda x: -x[1])
    result = [t for t, s in top_types if s > 0]
    return result[:2] if result else ["research"]


def _extract_keywords(text: str) -> list[str]:
    """基于规则提取关键词"""
    import jieba
    words = jieba.lcut(text)
    # 过滤停用词和短词
    keywords = [
        w.strip() for w in words
        if w.strip() not in _STOP_WORDS
        and len(w.strip()) >= 2
        and not re.match(r"^[A-Za-z]$", w.strip())  # 排除单字母选项
    ]

    # 额外提取：引号内容、专有名词模式
    quoted = re.findall(r"[「《]([^」》]+)[」》]", text)
    keywords.extend(quoted)

    # 提取数字和金额
    numbers = re.findall(r"\d+\.?\d*\s*(?:%|万|亿|元|年|月|日|天|次)", text)
    keywords.extend(numbers)

    # 去重并限制数量
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result[:12]


def _extract_entities(question: str, options: dict) -> dict[str, str]:
    """提取实体信息"""
    combined = question + " " + " ".join(options.values())
    entities = {}

    # 公司名
    company_pattern = r"([\u4e00-\u9fa5]{2,15}(?:股份有限|有限责任|有限)公司)"
    match = re.search(company_pattern, combined)
    if match:
        entities["company"] = match.group(1)

    # 年份
    year_pattern = r"(\d{4})\s*年"
    years = re.findall(year_pattern, combined)
    if years:
        entities["years"] = ",".join(sorted(set(years)))

    # 金额
    amount_pattern = r"([\d,]+\.?\d*)\s*([万亿]?元)"
    match = re.search(amount_pattern, combined)
    if match:
        entities["amount"] = f"{match.group(1)}{match.group(2)}"

    return entities


def _classify_intent(question: str) -> str:
    """分类问题意图"""
    if any(kw in question for kw in ["多少", "比例", "金额", "计算", "增长率"]):
        return "numerical"
    if any(kw in question for kw in ["比较", "对比", "差异", "不同"]):
        return "comparison"
    if any(kw in question for kw in ["是否", "判断", "正确", "错误"]):
        return "verification"
    if any(kw in question for kw in ["条件", "如果", "当", "在.*情况"]):
        return "conditional"
    return "fact_lookup"


def _build_search_queries(
    question: str, options: dict, keywords: list[str],
) -> list[str]:
    """构建多种检索查询"""
    queries = []

    # 查询1: 完整问题
    queries.append(question)

    # 查询2: 关键词组合
    if keywords:
        queries.append(" ".join(keywords[:6]))

    # 查询3: 每个选项的关键信息
    for opt_text in options.values():
        opt_kw = _extract_keywords(opt_text)
        if opt_kw:
            queries.append(" ".join(keywords[:3] + opt_kw[:3]))

    return queries[:6]
