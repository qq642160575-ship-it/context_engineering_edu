"""
F函数上下文工程 - LangChain实现
基于 index.html 中定义的14种策略

依赖安装:
pip install langchain langchain-openai langchain-core
"""

import json
from typing import Optional
from abc import ABC, abstractmethod

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# ============================================================
# 策略1: 全量缓冲 Full Buffer
# 公式: for t=1→T: C_{t+1} = I_t | O_t | C_t
# ============================================================

class FullBuffer:
    """不做任何压缩，保留全部历史"""

    def __init__(self):
        self.messages: list[BaseMessage] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 策略2: 滑动窗口 Sliding Window
# 公式: for t=1→T: C_{t+1} = FIFO(C, K)
# ============================================================

class SlidingWindow:
    """固定容量队列，超出K条丢弃最旧的"""

    def __init__(self, k: int = 10):
        self.k = k
        self.messages: list[BaseMessage] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k:]

    def get(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 策略3: 滚动摘要 Rolling Summary
# 公式: for t=1→T: C_{t+1} = Summary(C) | I | O
# ============================================================

class RollingSummary:
    """定期将历史压缩为摘要"""

    def __init__(self, llm: BaseChatModel, threshold: int = 10):
        self.llm = llm
        self.threshold = threshold
        self.messages: list[BaseMessage] = []
        self.summary: str = ""

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) >= self.threshold:
            self._summarize()

    def _summarize(self) -> None:
        history = "\n".join(m.content for m in self.messages[:-2])
        if not history:
            return
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "将以下对话简洁概括为一段摘要"),
                ("human", "{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            self.summary = chain.invoke({"text": history})
            self.messages = self.messages[-2:]
        except Exception:
            pass

    def get(self) -> list[BaseMessage]:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"历史摘要: {self.summary}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.summary = ""


# ============================================================
# 策略4: 混合缓冲摘要 Summary Buffer
# 公式: C_{t+1} = Summ(C_old) + C_recent
# ============================================================

class SummaryBuffer:
    """摘要层 + 近期原文层"""

    def __init__(self, llm: BaseChatModel, recent_size: int = 5):
        self.llm = llm
        self.recent_size = recent_size
        self.messages: list[BaseMessage] = []
        self.summary: str = ""

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.recent_size:
            self._merge()

    def _merge(self) -> None:
        old = self.messages[:len(self.messages) - self.recent_size]
        old_text = "\n".join(m.content for m in old)
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "总结以下对话"),
                ("human", "{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            new_summary = chain.invoke({"text": old_text})
            self.summary = f"{self.summary}\n{new_summary}".strip() if self.summary else new_summary
            self.messages = self.messages[-self.recent_size:]
        except Exception:
            pass

    def get(self) -> list[BaseMessage]:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"对话摘要:\n{self.summary}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.summary = ""


# ============================================================
# 策略5: Map-Reduce压缩
# 公式: C = Reduce(∪Map(S_i))
# ============================================================

class MapReduce:
    """分块提取要点，再汇总"""

    def __init__(self, llm: BaseChatModel, chunk_size: int = 5):
        self.llm = llm
        self.chunk_size = chunk_size
        self.messages: list[BaseMessage] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get(self) -> list[BaseMessage]:
        if len(self.messages) <= self.chunk_size:
            return self.messages.copy()

        # Map: 分块提取
        summaries = []
        for i in range(0, len(self.messages), self.chunk_size):
            chunk = self.messages[i:i + self.chunk_size]
            text = "\n".join(m.content for m in chunk)
            chain = (
                ChatPromptTemplate.from_messages([
                    ("human", "提取关键信息:\n{text}"),
                ]) | self.llm | StrOutputParser()
            )
            try:
                summaries.append(chain.invoke({"text": text}))
            except Exception:
                summaries.append(text[:200])

        # Reduce: 汇总
        combined = "\n---\n".join(summaries)
        chain = (
            ChatPromptTemplate.from_messages([
                ("human", "合并为简洁摘要:\n{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            final = chain.invoke({"text": combined})
        except Exception:
            final = combined[:500]

        return [SystemMessage(content=f"对话摘要:\n{final}")]

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 策略6: 语义剪枝 LLMLingua
# 公式: C_{t+1} = Prune(C_t, PPL)
# ============================================================

class SemanticPrune:
    """压缩token，去除冗余"""

    def __init__(self, llm: BaseChatModel, ratio: float = 0.3):
        self.llm = llm
        self.ratio = ratio
        self.messages: list[BaseMessage] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def _prune(self, text: str) -> str:
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "将文本压缩，保留核心意思，去除修饰词"),
                ("human", "{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            return chain.invoke({"text": text})
        except Exception:
            return text[:int(len(text) * self.ratio)]

    def get(self) -> list[BaseMessage]:
        result = []
        for msg in self.messages:
            pruned = self._prune(msg.content)
            result.append(msg.__class__(content=pruned))
        return result

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 策略7: 语义权重压缩 Semantic Compaction
# 公式: C = Filter(C, Weights)
# ============================================================

class WeightedFilter:
    """按内容权重过滤"""

    def __init__(self, keywords: Optional[list[str]] = None):
        self.keywords = keywords or [
            "帮我", "请", "必须", "需要", "决定", "喜欢", "讨厌",
            "时间", "地点", "名字", "价格", "方案", "同意", "拒绝",
        ]
        self.messages: list[BaseMessage] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def _filter(self, text: str) -> str:
        lines = text.split("\n")
        kept = [l for l in lines if any(k in l for k in self.keywords) or len(l.strip()) > 15]
        return "\n".join(kept) if kept else text

    def get(self) -> list[BaseMessage]:
        result = []
        for msg in self.messages:
            filtered = self._filter(msg.content)
            result.append(msg.__class__(content=filtered))
        return result

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 策略8: 向量检索记忆 Vector RAG
# 公式: P_{t+1} = Retrieve(I_t, VecDB) ∪ I_t
# ============================================================

class VectorRAG:
    """向量数据库按语义相似度检索"""

    def __init__(self, llm: BaseChatModel, top_k: int = 3):
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_core.documents import Document
        from langchain_openai import OpenAIEmbeddings

        self.llm = llm
        self.top_k = top_k
        self.messages: list[BaseMessage] = []
        self.embeddings = OpenAIEmbeddings()
        self.store = InMemoryVectorStore(self.embeddings)
        self._Document = Document

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        self.store.add_documents([self._Document(page_content=msg.content)])

    def get(self) -> list[BaseMessage]:
        if not self.messages:
            return []

        # 取最后一条用户消息查询
        query = ""
        for m in reversed(self.messages):
            if isinstance(m, HumanMessage):
                query = m.content
                break

        try:
            docs = self.store.similarity_search(query, k=self.top_k)
            context = "\n".join(d.page_content for d in docs)
            return [
                SystemMessage(content=f"相关历史:\n{context}"),
                self.messages[-1],
            ]
        except Exception:
            return self.messages.copy()

    def clear(self) -> None:
        self.messages = []
        self.store = type(self.store)(self.embeddings)


# ============================================================
# 策略9: 外部文件卸载 File Offloading
# 公式: C = {Path} ∪ I_t
# ============================================================

class FileOffload:
    """长内容转储到外部存储"""

    def __init__(self, max_len: int = 500):
        self.max_len = max_len
        self.messages: list[BaseMessage] = []
        self.storage: dict[str, str] = {}
        self._counter = 0

    def add(self, msg: BaseMessage) -> None:
        if len(msg.content) > self.max_len:
            fid = f"file_{self._counter}"
            self.storage[fid] = msg.content
            self._counter += 1
            self.messages.append(msg.__class__(content=f"[文件引用:{fid}]"))
        else:
            self.messages.append(msg)

    def read_file(self, fid: str) -> Optional[str]:
        return self.storage.get(fid)

    def get(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []
        self.storage = {}
        self._counter = 0


# ============================================================
# 策略10: 分层记忆 Hierarchical Memory
# 公式: P ∈ {L1, L2, L3}
# ============================================================

class HierarchicalMemory:
    """L1工作区 + L2近期 + L3长期摘要"""

    def __init__(self, llm: BaseChatModel, l1_size: int = 2, l2_size: int = 10):
        self.llm = llm
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l1: list[BaseMessage] = []   # 工作区
        self.l2: list[BaseMessage] = []   # 近期记忆
        self.l3: str = ""                  # 长期摘要

    def add(self, msg: BaseMessage) -> None:
        self.l1.append(msg)

        # L1满 -> 移到L2
        if len(self.l1) > self.l1_size:
            self.l2.append(self.l1.pop(0))

        # L2满 -> 压缩到L3
        if len(self.l2) > self.l2_size:
            self._compress_l2()

    def _compress_l2(self) -> None:
        text = "\n".join(m.content for m in self.l2)
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "简洁总结以下对话"),
                ("human", "{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            resp = chain.invoke({"text": text})
            self.l3 = f"{self.l3}\n{resp}".strip() if self.l3 else resp
            self.l2 = self.l2[-self.l2_size // 2:]
        except Exception:
            pass

    def get(self) -> list[BaseMessage]:
        result = []
        if self.l3:
            result.append(SystemMessage(content=f"长期记忆:\n{self.l3}"))
        result.extend(self.l2)
        result.extend(self.l1)
        return result

    def clear(self) -> None:
        self.l1 = []
        self.l2 = []
        self.l3 = ""


# ============================================================
# 策略11: 上下文范围限制 Scoped Memory
# 公式: C_i = Scope(C, Role_i)
# ============================================================

class ScopedMemory:
    """Multi-Agent场景，不同角色看到不同上下文"""

    def __init__(self, role_keywords: Optional[dict[str, list[str]]] = None):
        self.role_keywords = role_keywords or {}
        self.messages: list[BaseMessage] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_for_role(self, role: str) -> list[BaseMessage]:
        keywords = self.role_keywords.get(role, [])
        if not keywords:
            return self.messages.copy()

        result = []
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                result.append(msg)
            elif any(k in msg.content for k in keywords):
                result.append(msg)
        return result

    def get(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 策略12: 言语强化 Reflexion
# 公式: M = M ∪ Analyze(O, Error)
# ============================================================

class Reflexion:
    """存储错误反思，避免重复犯错"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.reflections: list[str] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def reflect(self, error: str, action: str) -> str:
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "分析错误并生成避免规则"),
                ("human", "错误:{error}\n失败动作:{action}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            rule = chain.invoke({"error": error, "action": action})
            self.reflections.append(rule)
            return rule
        except Exception as e:
            return str(e)

    def get(self) -> list[BaseMessage]:
        result = []
        if self.reflections:
            refs = "\n".join(self.reflections[-5:])
            result.append(SystemMessage(content=f"经验教训:\n{refs}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.reflections = []


# ============================================================
# 策略13: 事实提取 Fact Extraction
# 公式: M = Extract(I, O) → JSON
# ============================================================

class FactExtraction:
    """从对话提取结构化用户信息"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.facts: dict = {}

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if isinstance(msg, HumanMessage):
            self._extract(msg.content)

    def _extract(self, text: str) -> None:
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "从文本提取用户信息(JSON格式),只返回JSON"),
                ("human", "{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            resp = chain.invoke({"text": text})
            data = json.loads(resp)
            self.facts.update(data)
        except Exception:
            pass

    def get(self) -> list[BaseMessage]:
        result = []
        if self.facts:
            info = json.dumps(self.facts, ensure_ascii=False, indent=2)
            result.append(SystemMessage(content=f"用户信息:\n{info}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.facts = {}


# ============================================================
# 策略14: 知识图谱索引 KG Index
# 公式: C = Query(M, Entity)
# ============================================================

class KnowledgeGraph:
    """提取实体关系，构建图结构"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.entities: list[str] = []
        self.relations: list[dict] = []

    def add(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if isinstance(msg, HumanMessage):
            self._extract_graph(msg.content)

    def _extract_graph(self, text: str) -> None:
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "从文本提取实体和关系,格式:实体:[...];关系:[{{'head':'','rel':'','tail':''}},...]"),
                ("human", "{text}"),
            ]) | self.llm | StrOutputParser()
        )
        try:
            resp = chain.invoke({"text": text})
            if "实体:" in resp:
                ent = resp.split("实体:")[1].split("关系:")[0].strip()
                self.entities.extend(json.loads(ent))
            if "关系:" in resp:
                rel = resp.split("关系:")[1].strip()
                self.relations.extend(json.loads(rel))
        except Exception:
            pass

    def query(self, entity: str) -> list[str]:
        results = [entity]
        for r in self.relations:
            if r.get("head") == entity:
                results.append(f"--{r['rel']}--> {r['tail']}")
            elif r.get("tail") == entity:
                results.append(f"{r['head']} --{r['rel']}-->")
        return results

    def get(self) -> list[BaseMessage]:
        result = []
        if self.entities:
            result.append(SystemMessage(content=f"已知实体: {', '.join(set(self.entities))}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.entities = []
        self.relations = []


# ============================================================
# 工厂函数
# ============================================================

def create(name: str, llm: Optional[BaseChatModel] = None, **kw):
    """统一入口创建策略"""
    m = {
        "full":           lambda: FullBuffer(),
        "sliding":        lambda: SlidingWindow(**kw),
        "rolling":        lambda: RollingSummary(llm=llm, **kw),
        "summary_buffer": lambda: SummaryBuffer(llm=llm, **kw),
        "map_reduce":     lambda: MapReduce(llm=llm, **kw),
        "lingua":         lambda: SemanticPrune(llm=llm, **kw),
        "compaction":     lambda: WeightedFilter(**kw),
        "rag":            lambda: VectorRAG(llm=llm, **kw),
        "offloading":     lambda: FileOffload(**kw),
        "hierarchical":   lambda: HierarchicalMemory(llm=llm, **kw),
        "scoped":         lambda: ScopedMemory(**kw),
        "reflexion":      lambda: Reflexion(llm=llm),
        "fact":           lambda: FactExtraction(llm=llm),
        "kg":             lambda: KnowledgeGraph(llm=llm),
    }
    if name not in m:
        raise ValueError(f"未知策略: {name}\n可选: {list(m.keys())}")
    return m[name]()


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  F函数上下文工程 - 14种策略演示")
    print("=" * 60)

    # 1 FullBuffer
    print("\n[1] FullBuffer 全量缓冲")
    s = FullBuffer()
    s.add(HumanMessage("我叫张三"))
    s.add(AIMessage("你好张三"))
    s.add(HumanMessage("住北京"))
    print(f"    消息数: {len(s.get())}")

    # 2 SlidingWindow
    print("\n[2] SlidingWindow 滑动窗口(K=3)")
    s = SlidingWindow(k=3)
    for i in range(5):
        s.add(HumanMessage(f"msg{i+1}"))
    print(f"    保留: {len(s.get())} 条")

    # 3 RollingSummary
    print("\n[3] RollingSummary 滚动摘要")
    s = RollingSummary(llm=ChatOpenAI(), threshold=3)
    s.add(HumanMessage("讨论Python"))
    s.add(AIMessage("讲了requests库"))
    s.add(HumanMessage("继续讲爬虫"))
    print(f"    触发摘要后: {len(s.get())} 条")

    # 4 SummaryBuffer
    print("\n[4] SummaryBuffer 混合缓冲")
    s = SummaryBuffer(llm=ChatOpenAI(), recent_size=3)
    for i in range(6):
        s.add(HumanMessage(f"消息{i+1}"))
    print(f"    消息数: {len(s.get())}")

    # 5 MapReduce
    print("\n[5] MapReduce 压缩")
    s = MapReduce(llm=ChatOpenAI(), chunk_size=3)
    for i in range(6):
        s.add(HumanMessage(f"信息{i+1}"))
    print(f"    压缩后: {len(s.get())} 条")

    # 6 SemanticPrune
    print("\n[6] SemanticPrune 语义剪枝")
    s = SemanticPrune(llm=ChatOpenAI())
    s.add(HumanMessage("我认为你现在绝对应该立刻去超市买苹果"))
    print(f"    压缩完成")

    # 7 WeightedFilter
    print("\n[7] WeightedFilter 权重过滤")
    s = WeightedFilter()
    s.add(HumanMessage("今天天气不错"))
    s.add(HumanMessage("帮我订明天去北京的机票"))
    s.add(HumanMessage("哈哈好的谢谢"))
    print(f"    原始3条，过滤后: {len(s.get())} 条")

    # 8 VectorRAG
    print("\n[8] VectorRAG 向量检索")
    print("    [需要OpenAI API Key]")

    # 9 FileOffload
    print("\n[9] FileOffload 文件卸载")
    s = FileOffload(max_len=50)
    s.add(HumanMessage("短消息"))
    s.add(HumanMessage("长" * 100))
    print(f"    上下文: {len(s.get())} 条, 文件: {len(s.storage)} 个")

    # 10 HierarchicalMemory
    print("\n[10] HierarchicalMemory 分层记忆")
    s = HierarchicalMemory(llm=ChatOpenAI(), l1_size=2, l2_size=4)
    for i in range(8):
        s.add(HumanMessage(f"消息{i+1}"))
    print(f"    L1:{len(s.l1)}, L2:{len(s.l2)}, L3:{bool(s.l3)}")

    # 11 ScopedMemory
    print("\n[11] ScopedMemory 范围限制")
    s = ScopedMemory({
        "frontend": ["UI", "按钮", "css"],
        "backend": ["API", "数据库"],
    })
    s.add(HumanMessage("把按钮改成蓝色"))
    s.add(HumanMessage("数据库连接超时"))
    print(f"    前端:{len(s.get_for_role('frontend'))}, 后端:{len(s.get_for_role('backend'))}")

    # 12 Reflexion
    print("\n[12] Reflexion 言语强化")
    s = Reflexion(llm=ChatOpenAI())
    s.add(HumanMessage("帮我写贪吃蛇"))
    s.reflections = ["下次写游戏前必须import pygame"]
    print(f"    反思记录: {len(s.reflections)} 条")

    # 13 FactExtraction
    print("\n[13] FactExtraction 事实提取")
    s = FactExtraction(llm=ChatOpenAI())
    s.facts = {"name": "张三", "allergies": ["peanut"]}
    print(f"    已提取事实: {list(s.facts.keys())}")

    # 14 KnowledgeGraph
    print("\n[14] KnowledgeGraph 知识图谱")
    s = KnowledgeGraph(llm=ChatOpenAI())
    s.entities = ["张三", "李四", "星空科技"]
    s.relations = [
        {"head": "张三", "rel": "夫妻", "tail": "李四"},
        {"head": "李四", "rel": "拥有", "tail": "星空科技"},
    ]
    print(f"    查询'张三': {s.query('张三')}")

    print("\n" + "=" * 60)
    print("  策略列表:")
    for i, n in enumerate([
        "full","sliding","rolling","summary_buffer",
        "map_reduce","lingua","compaction","rag",
        "offloading","hierarchical","scoped",
        "reflexion","fact","kg"
    ], 1):
        print(f"    {i:2d}. {n}")
    print("=" * 60)
