"""
LLM 上下文工程 F 函数策略实现
基于 index.html 中定义的 14 种上下文管理策略

依赖: pip install langchain langchain-openai langchain-core
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# ============================================================
# 基类
# ============================================================

class BaseStrategy(ABC):
    """
    上下文管理策略抽象基类
    
    所有策略统一接口:
    - add_message(): 添加消息
    - get_messages(): 获取发送给LLM的上下文
    - clear(): 清空上下文
    """

    @abstractmethod
    def add_message(self, msg: BaseMessage) -> None:
        pass

    @abstractmethod
    def get_messages(self) -> list[BaseMessage]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


# ============================================================
# 一、基础范式 (Basic Paradigm)
# ============================================================

class FullBufferStrategy(BaseStrategy):
    """
    全量缓冲 (Full Buffer)
    ─────────────────────────
    公式: for t=1→T: C_{t+1} = I_t | O_t | C_t
    
    不做任何过滤, 完整保留所有对话历史.
    信息忠实度最高, 但Token消耗不可持续.
    
    适用: Playground测试, 极短对话
    """

    def __init__(self):
        self.messages: list[BaseMessage] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


class SlidingWindowStrategy(BaseStrategy):
    """
    滑动窗口 (Sliding Window)
    ─────────────────────────
    公式: for t=1→T: C_{t+1} = FIFO(C, K)
    
    维护固定容量K的双端队列, 超出后丢弃最旧的消息.
    响应延迟稳定, 但会导致"瞬时失忆".
    
    适用: 实时聊天, 成本敏感场景
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: list[BaseMessage] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_messages(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 二、压缩型 (Compression)
# ============================================================

class RollingSummaryStrategy(BaseStrategy):
    """
    滚动摘要 (Rolling Summary)
    ─────────────────────────
    公式: for t=1→T: C_{t+1} = Summary(C) | I | O
    
    定期将历史对话压缩为摘要.
    长期记忆能力好, 但每次压缩会丢失细节.
    
    适用: 陪伴型AI, 剧情类角色扮演
    """

    def __init__(self, llm: BaseChatModel, threshold: int = 10):
        self.llm = llm
        self.threshold = threshold
        self.messages: list[BaseMessage] = []
        self.summary: str = ""

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) >= self.threshold:
            self._summarize()

    def _summarize(self) -> None:
        """生成摘要并压缩历史"""
        if len(self.messages) < 3:
            return
        
        old_text = "\n".join(m.content for m in self.messages[:-2])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "请将以下对话简洁概括为一段摘要"),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"text": old_text})
            self.summary = resp.content
            self.messages = self.messages[-2:]  # 保留最近2条
        except Exception:
            pass

    def get_messages(self) -> list[BaseMessage]:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"历史摘要: {self.summary}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.summary = ""


class SummaryBufferStrategy(BaseStrategy):
    """
    混合缓冲摘要 (Summary Buffer)
    ─────────────────────────
    公式: C_{t+1} = Summ(C_old) + C_recent
    
    分层架构: 热缓冲区(近期原文) + 冷存储区(历史摘要).
    兼顾微观准确性与宏观记忆度.
    
    适用: 生产级Agent默认方案 (Dify, LangChain推荐)
    """

    def __init__(self, llm: BaseChatModel, recent_size: int = 5):
        self.llm = llm
        self.recent_size = recent_size
        self.messages: list[BaseMessage] = []
        self.summary: str = ""

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.recent_size:
            self._merge()

    def _merge(self) -> None:
        """将老消息压缩为摘要"""
        old_msgs = self.messages[:len(self.messages) - self.recent_size]
        old_text = "\n".join(m.content for m in old_msgs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "总结以下对话要点"),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"text": old_text})
            new_summary = resp.content
            self.summary = (self.summary + "\n" + new_summary).strip() if self.summary else new_summary
            self.messages = self.messages[-self.recent_size:]
        except Exception:
            pass

    def get_messages(self) -> list[BaseMessage]:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"对话摘要:\n{self.summary}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.summary = ""


class MapReduceStrategy(BaseStrategy):
    """
    Map-Reduce 压缩
    ─────────────────────────
    公式: C = Reduce(∪Map(S_i))
    
    借用大数据思想: 先分块提取(Map), 再聚合合成(Reduce).
    可处理超长文本, 但会破坏时序逻辑.
    
    适用: 离线预处理, 不适合实时对话
    """

    def __init__(self, llm: BaseChatModel, chunk_size: int = 5):
        self.llm = llm
        self.chunk_size = chunk_size
        self.messages: list[BaseMessage] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages(self) -> list[BaseMessage]:
        if len(self.messages) <= self.chunk_size:
            return self.messages.copy()

        # Map: 分块提取
        summaries: list[str] = []
        for i in range(0, len(self.messages), self.chunk_size):
            chunk = self.messages[i:i + self.chunk_size]
            chunk_text = "\n".join(m.content for m in chunk)
            
            prompt = ChatPromptTemplate.from_messages([
                ("human", "提取以下对话的关键信息:\n{text}"),
            ])
            chain = prompt | self.llm
            try:
                resp = chain.invoke({"text": chunk_text})
                summaries.append(resp.content)
            except Exception:
                summaries.append(chunk_text[:200])

        # Reduce: 汇总
        combined = "\n---\n".join(summaries)
        prompt = ChatPromptTemplate.from_messages([
            ("human", "合并以下要点为简洁摘要:\n{text}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"text": combined})
            return [SystemMessage(content=f"对话摘要:\n{resp.content}")]
        except Exception:
            return [SystemMessage(content=f"对话摘要:\n{combined[:500]}")]

    def clear(self) -> None:
        self.messages = []


class SemanticPruneStrategy(BaseStrategy):
    """
    语义剪枝 (LLMLingua)
    ─────────────────────────
    公式: C_{t+1} = Prune(C_t, PPL)
    
    利用小模型评估Token信息熵, 移除冗余词汇.
    可压缩5-20倍, 但文本人类难以阅读.
    
    适用: 复杂RAG系统长上下文预处理
    """

    def __init__(self, llm: BaseChatModel, ratio: float = 0.3):
        self.llm = llm
        self.ratio = ratio
        self.messages: list[BaseMessage] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def _prune(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "将文本压缩, 保留核心意思, 去除修饰词"),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"text": text})
            return resp.content
        except Exception:
            return text[:int(len(text) * self.ratio)]

    def get_messages(self) -> list[BaseMessage]:
        result = []
        for msg in self.messages:
            pruned = self._prune(msg.content)
            if isinstance(msg, HumanMessage):
                result.append(HumanMessage(content=pruned))
            elif isinstance(msg, AIMessage):
                result.append(AIMessage(content=pruned))
            else:
                result.append(msg)
        return result

    def clear(self) -> None:
        self.messages = []


class SemanticCompactionStrategy(BaseStrategy):
    """
    语义权重压缩 (Semantic Compaction)
    ─────────────────────────
    公式: C = Filter(C, Weights)
    
    预定义内容权重体系, 过滤闲聊保留高价值指令.
    任务聚焦度高, 但会失去人情味.
    
    适用: 任务型Agent (订票、点餐助手)
    """

    def __init__(self, high_value_keywords: Optional[list[str]] = None):
        self.messages: list[BaseMessage] = []
        self.high_value = high_value_keywords or [
            "帮我", "请", "必须", "需要", "决定", "喜欢", "讨厌",
            "时间", "地点", "名字", "价格", "方案", "同意", "拒绝",
        ]

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def _filter(self, text: str) -> str:
        lines = text.split("\n")
        kept = []
        for line in lines:
            has_keyword = any(kw in line for kw in self.high_value)
            is_substantial = len(line.strip()) > 15
            if has_keyword or is_substantial:
                kept.append(line)
        return "\n".join(kept) if kept else text

    def get_messages(self) -> list[BaseMessage]:
        result = []
        for msg in self.messages:
            filtered = self._filter(msg.content)
            if isinstance(msg, HumanMessage):
                result.append(HumanMessage(content=filtered))
            elif isinstance(msg, AIMessage):
                result.append(AIMessage(content=filtered))
            else:
                result.append(msg)
        return result

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 三、架构型 (Architecture)
# ============================================================

class VectorRAGStrategy(BaseStrategy):
    """
    向量检索记忆 (Vector RAG)
    ─────────────────────────
    公式: P_{t+1} = Retrieve(I_t, VecDB) ∪ I_t
    
    将对话转为高维向量存储, 按语义相似度检索.
    记忆容量无限, 但语义相似≠逻辑相关.
    
    适用: MemGPT, Zep 等长期记忆系统
    """

    def __init__(self, embeddings: Any = None, top_k: int = 3):
        self.messages: list[BaseMessage] = []
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.store = InMemoryVectorStore(embedding=self.embeddings)
        self.top_k = top_k

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        doc = Document(page_content=msg.content, metadata={"role": "user" if isinstance(msg, HumanMessage) else "assistant"})
        self.store.add_documents([doc])

    def get_messages(self) -> list[BaseMessage]:
        if not self.messages:
            return []

        # 取最后一条用户消息作为查询
        last_query = ""
        for m in reversed(self.messages):
            if isinstance(m, HumanMessage):
                last_query = m.content
                break

        if not last_query:
            return self.messages.copy()

        try:
            docs = self.store.similarity_search(last_query, k=self.top_k)
            context = "\n".join(d.page_content for d in docs)
            return [
                SystemMessage(content=f"相关历史:\n{context}"),
                self.messages[-1],
            ]
        except Exception:
            return self.messages.copy()

    def clear(self) -> None:
        self.messages = []
        self.store = InMemoryVectorStore(embedding=self.embeddings)


class FileOffloadingStrategy(BaseStrategy):
    """
    外部文件卸载 (File Offloading)
    ─────────────────────────
    公式: C = {Path} ∪ I_t
    
    长内容转储到外部存储, 上下文只留引用.
    规避Token窗口限制, 但引入IO延迟.
    
    适用: 代码仓库分析, 大数据处理
    """

    def __init__(self, max_len: int = 500):
        self.max_len = max_len
        self.messages: list[BaseMessage] = []
        self.storage: dict[str, str] = {}
        self._counter = 0

    def add_message(self, msg: BaseMessage) -> None:
        if len(msg.content) > self.max_len:
            file_id = f"file_{self._counter}"
            self.storage[file_id] = msg.content
            self._counter += 1
            ref = f"[文件引用: {file_id}]"
            if isinstance(msg, HumanMessage):
                self.messages.append(HumanMessage(content=ref))
            elif isinstance(msg, AIMessage):
                self.messages.append(AIMessage(content=ref))
            else:
                self.messages.append(SystemMessage(content=ref))
        else:
            self.messages.append(msg)

    def read_file(self, file_id: str) -> Optional[str]:
        """读取卸载的文件内容"""
        return self.storage.get(file_id)

    def get_messages(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []
        self.storage = {}
        self._counter = 0


class HierarchicalMemoryStrategy(BaseStrategy):
    """
    分层记忆架构 (Hierarchical Memory)
    ─────────────────────────
    公式: P ∈ {L1, L2, L3}
    
    对标计算机缓存体系:
    - L1: 工作区 (当前轮次)
    - L2: 近期记忆 (滑动窗口)
    - L3: 长期存储 (压缩摘要)
    
    通用人工智能Agent的底层基础设施.
    """

    def __init__(self, llm: BaseChatModel, l1_size: int = 2, l2_size: int = 10):
        self.llm = llm
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l1: list[BaseMessage] = []   # 工作区
        self.l2: list[BaseMessage] = []   # 近期记忆
        self.l3: str = ""                 # 长期摘要

    def add_message(self, msg: BaseMessage) -> None:
        # L1: 工作区
        self.l1.append(msg)

        # L1满了 -> 移到L2
        if len(self.l1) > self.l1_size:
            moved = self.l1.pop(0)
            self.l2.append(moved)

        # L2满了 -> 压缩到L3
        if len(self.l2) > self.l2_size:
            self._compress_l2()

    def _compress_l2(self) -> None:
        """L2压缩到L3"""
        if not self.l2:
            return
        text = "\n".join(m.content for m in self.l2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "简洁总结以下对话"),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"text": text})
            self.l3 = (self.l3 + "\n" + resp.content).strip() if self.l3 else resp.content
            self.l2 = self.l2[-self.l2_size // 2:]
        except Exception:
            pass

    def get_messages(self) -> list[BaseMessage]:
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


class ScopedMemoryStrategy(BaseStrategy):
    """
    上下文范围限制 (Scoped Memory)
    ─────────────────────────
    公式: C_i = Scope(C, Role_i)
    
    Multi-Agent协作场景: 为不同角色Agent剪裁独立的上下文视图.
    通过信息隔离降低认知负担.
    
    适用: CrewAI, AutoGen 多智能体框架
    """

    def __init__(self, role_keywords: Optional[dict[str, list[str]]] = None):
        self.role_keywords = role_keywords or {}
        self.messages: list[BaseMessage] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages_for_role(self, role: str) -> list[BaseMessage]:
        """获取特定角色可见的消息"""
        keywords = self.role_keywords.get(role, [])
        if not keywords:
            return self.messages.copy()

        result = []
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                result.append(msg)  # system消息始终可见
            elif any(kw in msg.content for kw in keywords):
                result.append(msg)
        return result

    def get_messages(self) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 四、结构化 (Structured)
# ============================================================

class ReflexionStrategy(BaseStrategy):
    """
    言语强化 (Reflexion)
    ─────────────────────────
    公式: M = M ∪ Analyze(O, Error)
    
    存储"做错了什么以及为什么错".
    下次执行相同任务时, 这些"教训"作为前置约束.
    
    最接近人类成长学习机制的范式.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.reflections: list[str] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def reflect_on_error(self, error: str, action: str) -> str:
        """记录错误反思"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "分析错误并生成避免规则"),
            ("human", "错误: {error}\n失败动作: {action}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"error": error, "action": action})
            self.reflections.append(resp.content)
            return resp.content
        except Exception as e:
            return str(e)

    def get_messages(self) -> list[BaseMessage]:
        result = []
        if self.reflections:
            refs = "\n".join(self.reflections[-5:])
            result.append(SystemMessage(content=f"经验教训:\n{refs}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.reflections = []


class FactExtractionStrategy(BaseStrategy):
    """
    事实提取 (Fact Extraction)
    ─────────────────────────
    公式: M = Extract(I, O) → JSON
    
    从对话中提取实体和属性, 转化为结构化JSON.
    数据确定性高, 可与CRM等系统对接.
    
    适用: 个性化AI伴侣, 健康顾问
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.facts: dict[str, Any] = {}

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if isinstance(msg, HumanMessage):
            self._extract(msg.content)

    def _extract(self, text: str) -> None:
        """从文本中提取结构化信息"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "从以下文本提取用户信息(JSON格式), 只返回JSON\n示例: {{\"name\":\"\",\"allergies\":[],\"preferences\":[]}}"),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        try:
            import json
            resp = chain.invoke({"text": text})
            extracted = json.loads(resp.content)
            self.facts.update(extracted)
        except Exception:
            pass

    def get_facts(self) -> dict[str, Any]:
        return self.facts.copy()

    def get_messages(self) -> list[BaseMessage]:
        result = []
        if self.facts:
            import json
            info = json.dumps(self.facts, ensure_ascii=False, indent=2)
            result.append(SystemMessage(content=f"用户信息:\n{info}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.facts = {}


class KnowledgeGraphStrategy(BaseStrategy):
    """
    知识图谱索引 (KG Index)
    ─────────────────────────
    公式: C = Query(M, Entity)
    
    解析对话为节点(实体)和边(关系)构成的图.
    能回答存在深度隐性逻辑联系的复杂问题.
    
    记忆的终极形态: 从"检索"到"联想推理".
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.messages: list[BaseMessage] = []
        self.entities: list[str] = []
        self.relations: list[dict[str, str]] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if isinstance(msg, HumanMessage):
            self._extract_graph(msg.content)

    def _extract_graph(self, text: str) -> None:
        """从文本提取实体和关系"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "从文本提取实体和关系\n格式: 实体: [E1,E2]; 关系: [{\"head\":\"\",\"rel\":\"\",\"tail\":\"\"}]"),
            ("human", "{text}"),
        ])
        chain = prompt | self.llm
        try:
            resp = chain.invoke({"text": text})
            content = resp.content
            if "实体:" in content:
                ent_str = content.split("实体:")[1].split("关系:")[0].strip()
                self.entities.extend(eval(ent_str))
            if "关系:" in content:
                rel_str = content.split("关系:")[1].strip()
                self.relations.extend(eval(rel_str))
        except Exception:
            pass

    def query(self, entity: str, max_hops: int = 2) -> list[str]:
        """按实体查询, 多跳遍历关系"""
        results = [entity]
        for _ in range(max_hops):
            for rel in self.relations:
                if rel.get("head") == entity and rel["tail"] not in results:
                    results.append(f"--{rel['rel']}--> {rel['tail']}")
                elif rel.get("tail") == entity and rel["head"] not in results:
                    results.append(f"{rel['head']} --{rel['rel']}-->")
        return results

    def get_messages(self) -> list[BaseMessage]:
        result = []
        if self.entities:
            result.append(SystemMessage(
                content=f"已知实体: {', '.join(set(self.entities))}"
            ))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.entities = []
        self.relations = []


# ============================================================
# 工厂函数
# ============================================================

def create_strategy(
    name: str,
    llm: Optional[BaseChatModel] = None,
    **kwargs,
) -> BaseStrategy:
    """
    根据名称创建策略实例
    
    Args:
        name: 策略名称 (full/sliding/rolling/summary_buffer/map_reduce/
              lingua/compaction/rag/offloading/hierarchical/scoped/
              reflexion/fact/kg)
        llm: 语言模型实例 (部分策略需要)
        **kwargs: 策略参数
    
    Returns:
        策略实例
    
    Example:
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> strategy = create_strategy("summary_buffer", llm=llm, recent_size=5)
    """
    strategies = {
        "full":           FullBufferStrategy,
        "sliding":        SlidingWindowStrategy,
        "rolling":        lambda: RollingSummaryStrategy(llm=llm, **kwargs),
        "summary_buffer": lambda: SummaryBufferStrategy(llm=llm, **kwargs),
        "map_reduce":     lambda: MapReduceStrategy(llm=llm, **kwargs),
        "lingua":         lambda: SemanticPruneStrategy(llm=llm, **kwargs),
        "compaction":     SemanticCompactionStrategy,
        "rag":            VectorRAGStrategy,
        "offloading":     FileOffloadingStrategy,
        "hierarchical":   lambda: HierarchicalMemoryStrategy(llm=llm, **kwargs),
        "scoped":         ScopedMemoryStrategy,
        "reflexion":      lambda: ReflexionStrategy(llm=llm),
        "fact":           lambda: FactExtractionStrategy(llm=llm),
        "kg":             lambda: KnowledgeGraphStrategy(llm=llm),
    }

    if name not in strategies:
        raise ValueError(f"未知策略: {name}\n可选: {list(strategies.keys())}")

    factory = strategies[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory(**kwargs)


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    from unittest.mock import MagicMock

    print("=" * 60)
    print("  LLM 上下文工程 - 14种 F 函数策略实现")
    print("=" * 60)

    # --- 1. FullBuffer 全量缓冲 ---
    print("\n[1] FullBuffer 全量缓冲")
    strategy = FullBufferStrategy()
    strategy.add_message(HumanMessage(content="我叫张三"))
    strategy.add_message(AIMessage(content="你好张三!"))
    strategy.add_message(HumanMessage(content="我住北京"))
    print(f"    消息数: {len(strategy.get_messages())}")

    # --- 2. SlidingWindow 滑动窗口 ---
    print("\n[2] SlidingWindow 滑动窗口 (K=3)")
    strategy = SlidingWindowStrategy(window_size=3)
    for i in range(5):
        strategy.add_message(HumanMessage(content=f"消息{i+1}"))
    print(f"    保留消息数: {len(strategy.get_messages())}")

    # --- 3. SummaryBuffer 混合缓冲摘要 ---
    print("\n[3] SummaryBuffer 混合缓冲摘要")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="这是摘要")
    strategy = SummaryBufferStrategy(llm=mock_llm, recent_size=3)
    for i in range(6):
        strategy.add_message(HumanMessage(content=f"消息{i+1}"))
    print(f"    消息数: {len(strategy.get_messages())}")
    print(f"    包含摘要: {bool(strategy.summary)}")

    # --- 4. FileOffloading 文件卸载 ---
    print("\n[4] FileOffloading 文件卸载")
    strategy = FileOffloadingStrategy(max_len=50)
    strategy.add_message(HumanMessage(content="短消息"))
    strategy.add_message(HumanMessage(content="长" * 100))
    print(f"    上下文消息数: {len(strategy.get_messages())}")
    print(f"    文件存储数: {len(strategy.storage)}")

    # --- 5. ScopedMemory 范围限制 ---
    print("\n[5] ScopedMemory 范围限制")
    strategy = ScopedMemoryStrategy(role_keywords={
        "frontend": ["UI", "按钮", "css"],
        "backend": ["API", "数据库"],
    })
    strategy.add_message(HumanMessage(content="把按钮改成蓝色"))
    strategy.add_message(HumanMessage(content="数据库连接超时"))
    strategy.add_message(HumanMessage(content="API返回500错误"))
    front_msgs = strategy.get_messages_for_role("frontend")
    back_msgs = strategy.get_messages_for_role("backend")
    print(f"    前端Agent可见: {len(front_msgs)} 条")
    print(f"    后端Agent可见: {len(back_msgs)} 条")

    # --- 6. KnowledgeGraph 知识图谱 ---
    print("\n[6] KnowledgeGraph 知识图谱")
    strategy = KnowledgeGraphStrategy(llm=mock_llm)
    strategy.entities = ["张三", "李四", "星空科技"]
    strategy.relations = [
        {"head": "张三", "rel": "夫妻", "tail": "李四"},
        {"head": "李四", "rel": "拥有", "tail": "星空科技"},
    ]
    results = strategy.query("张三")
    print(f"    查询'张三': {results}")

    # --- 7. Reflexion 言语强化 ---
    print("\n[7] Reflexion 言语强化")
    strategy = ReflexionStrategy(llm=mock_llm)
    strategy.reflections = ["下次写游戏前必须import pygame"]
    print(f"    反思记录: {len(strategy.reflections)} 条")
    msgs = strategy.get_messages()
    print(f"    包含经验教训: {any('经验教训' in m.content for m in msgs if isinstance(m, SystemMessage))}")

    # --- 8. FactExtraction 事实提取 ---
    print("\n[8] FactExtraction 事实提取")
    strategy = FactExtractionStrategy(llm=mock_llm)
    strategy.facts = {"name": "张三", "allergies": ["peanut"]}
    print(f"    已提取: {list(strategy.facts.keys())}")

    # --- 工厂函数演示 ---
    print("\n[9] 工厂函数 create_strategy")
    for name in ["full", "sliding", "compaction", "offloading", "scoped"]:
        s = create_strategy(name, window_size=5, max_len=100, recent_size=3)
        print(f"    create_strategy('{name}') -> {type(s).__name__}")

    print("\n" + "=" * 60)
    print("  14种策略清单:")
    strategies_list = [
        ("full",           "全量缓冲"),
        ("sliding",        "滑动窗口"),
        ("rolling",        "滚动摘要"),
        ("summary_buffer", "混合缓冲摘要"),
        ("map_reduce",     "Map-Reduce"),
        ("lingua",         "语义剪枝"),
        ("compaction",     "语义权重压缩"),
        ("rag",            "向量RAG"),
        ("offloading",     "文件卸载"),
        ("hierarchical",   "分层记忆"),
        ("scoped",         "范围限制"),
        ("reflexion",      "言语强化"),
        ("fact",           "事实提取"),
        ("kg",             "知识图谱"),
    ]
    for i, (en, zh) in enumerate(strategies_list, 1):
        print(f"    {i:2d}. {en:20s} {zh}")
    print("=" * 60)
