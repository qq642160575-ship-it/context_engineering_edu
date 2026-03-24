"""
LLM 上下文工程 F 函数策略实现
基于 index.html 中的 14 种上下文管理策略
依赖: langchain, langchain-openai, langchain-community, faiss-cpu
安装: pip install langchain langchain-openai langchain-community faiss-cpu
"""

from typing import Any, Optional
from abc import ABC, abstractmethod

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ============================================================
# 抽象基类
# ============================================================

class ContextStrategy(ABC):
    """上下文管理策略基类"""

    @abstractmethod
    def add_message(self, msg: BaseMessage) -> None:
        """添加消息"""
        pass

    @abstractmethod
    def get_messages(self, config: Optional[RunnableConfig] = None) -> list[BaseMessage]:
        """获取发送给LLM的消息列表"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空上下文"""
        pass


# ============================================================
# 1. 基础范式 (Basic)
# ============================================================

class FullBufferStrategy(ContextStrategy):
    """
    全量缓冲 (Full Buffer)
    公式: C_{t+1} = I_t | O_t | C_t
    完整保留所有对话历史
    """

    def __init__(self):
        self.messages: list[BaseMessage] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages(self, config=None) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


class SlidingWindowStrategy(ContextStrategy):
    """
    滑动窗口 (Sliding Window)
    公式: C_{t+1} = FIFO(C, K)
    只保留最近K条消息
    """

    def __init__(self, llm: BaseChatModel, window_size: int = 10):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.window_size = window_size

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages(self, config=None) -> list[BaseMessage]:
        # 使用langchain的trim_messages进行滑动窗口
        return trim_messages(
            self.messages,
            token_counter=len,  # 按消息条数裁剪
            max_tokens=self.window_size,
            strategy="last",
            start_on="human",
            include_system=False,
        )

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 2. 压缩型 (Compression)
# ============================================================

class RollingSummaryStrategy(ContextStrategy):
    """
    滚动摘要 (Rolling Summary)
    公式: C_{t+1} = Summary(C) | I | O
    将历史对话压缩为摘要
    """

    def __init__(self, llm: BaseChatModel, threshold: int = 10):
        self.messages: list[BaseMessage] = []
        self.summary: str = ""
        self.llm = llm
        self.threshold = threshold
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "请将以下对话简洁概括为一段摘要"),
            ("human", "{conversation}"),
        ])

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) >= self.threshold:
            self._summarize()

    def _summarize(self) -> None:
        """生成摘要并更新"""
        if not self.messages:
            return

        conv_text = "\n".join(m.content for m in self.messages[:-2])
        chain = self.prompt | self.llm
        try:
            result = chain.invoke({"conversation": conv_text})
            self.summary = result.content
            self.messages = self.messages[-2:]  # 保留最近2条
        except Exception:
            pass

    def get_messages(self, config=None) -> list[BaseMessage]:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"历史摘要: {self.summary}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.summary = ""


class SummaryBufferStrategy(ContextStrategy):
    """
    混合缓冲摘要 (Summary Buffer)
    公式: C_{t+1} = Summ(C_old) + C_recent
    分层架构：摘要层 + 近期原文层
    LangChain 内置实现: ConversationSummaryBufferMemory
    """

    def __init__(self, llm: BaseChatModel, recent_size: int = 5):
        self.messages: list[BaseMessage] = []
        self.summary: str = ""
        self.llm = llm
        self.recent_size = recent_size

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if len(self.messages) > self.recent_size:
            self._merge_to_summary()

    def _merge_to_summary(self) -> None:
        """将旧消息合并到摘要"""
        old_msgs = self.messages[:len(self.messages) - self.recent_size]
        old_text = "\n".join(m.content for m in old_msgs)

        try:
            resp = self.llm.invoke(f"请总结以下对话:\n{old_text}")
            new_summary = resp.content
            self.summary = (self.summary + "\n" + new_summary).strip() if self.summary else new_summary
            self.messages = self.messages[-self.recent_size:]
        except Exception:
            pass

    def get_messages(self, config=None) -> list[BaseMessage]:
        result = []
        if self.summary:
            result.append(SystemMessage(content=f"对话摘要:\n{self.summary}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.summary = ""


class MapReduceStrategy(ContextStrategy):
    """
    Map-Reduce 压缩
    公式: C = Reduce(∪Map(S_i))
    分块并行提取，然后合并总结
    """

    def __init__(self, llm: BaseChatModel, chunk_size: int = 5):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.chunk_size = chunk_size
        self.final_summary: str = ""

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages(self, config=None) -> list[BaseMessage]:
        if len(self.messages) <= self.chunk_size:
            return self.messages.copy()

        # Map: 分块提取要点
        summaries: list[str] = []
        for i in range(0, len(self.messages), self.chunk_size):
            chunk = self.messages[i:i + self.chunk_size]
            chunk_text = "\n".join(m.content for m in chunk)
            try:
                resp = self.llm.invoke(f"提取以下对话的关键信息:\n{chunk_text}")
                summaries.append(resp.content)
            except Exception:
                summaries.append(chunk_text[:200])

        # Reduce: 汇总
        combined = "\n---\n".join(summaries)
        try:
            resp = self.llm.invoke(f"合并以下要点为简洁摘要:\n{combined}")
            self.final_summary = resp.content
        except Exception:
            self.final_summary = combined[:500]

        return [SystemMessage(content=f"对话摘要:\n{self.final_summary}")]

    def clear(self) -> None:
        self.messages = []
        self.final_summary = ""


class LLMLinguaStrategy(ContextStrategy):
    """
    语义剪枝 (LLMLingua)
    公式: C_{t+1} = Prune(C_t, PPL)
    基于信息熵压缩token
    """

    def __init__(self, llm: BaseChatModel, compression_ratio: float = 0.3):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.compression_ratio = compression_ratio

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def _prune(self, text: str) -> str:
        """语义压缩"""
        prompt = (
            f"请将以下文本压缩到原来的{self.compression_ratio*100:.0f}%，"
            f"保留核心意思，去除修饰词:\n{text}"
        )
        try:
            resp = self.llm.invoke(prompt)
            return resp.content
        except Exception:
            return text[:int(len(text) * self.compression_ratio)]

    def get_messages(self, config=None) -> list[BaseMessage]:
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


class SemanticCompactionStrategy(ContextStrategy):
    """
    语义权重压缩 (Semantic Compaction)
    公式: C = Filter(C, Weights)
    基于内容权重过滤对话
    """

    def __init__(self, high_value_keywords: list[str] | None = None):
        self.messages: list[BaseMessage] = []
        self.high_value = high_value_keywords or [
            "帮我", "请", "必须", "需要", "决定", "喜欢", "讨厌",
            "时间", "地点", "名字", "价格", "方案", "同意", "拒绝",
        ]

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def _filter(self, content: str) -> str:
        """根据权重过滤，保留高价值内容"""
        lines = content.split("\n")
        filtered = []
        for line in lines:
            # 高价值行保留，纯语气词丢弃
            if any(kw in line for kw in self.high_value):
                filtered.append(line)
            elif len(line.strip()) > 15:
                filtered.append(line)
        return "\n".join(filtered) if filtered else content

    def get_messages(self, config=None) -> list[BaseMessage]:
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
# 3. 架构型 (Architecture)
# ============================================================

class VectorRAGStrategy(ContextStrategy):
    """
    向量检索记忆 (Vector RAG)
    公式: P_{t+1} = Retrieve(I_t, VecDB) ∪ I_t
    使用向量数据库按语义相似度检索
    """

    def __init__(self, llm: BaseChatModel, embeddings: Any = None, top_k: int = 3):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self.top_k = top_k

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        # 存入向量库
        doc = Document(page_content=msg.content, metadata={"type": "message"})
        self.vectorstore.add_documents([doc])

    def get_messages(self, config=None) -> list[BaseMessage]:
        if not self.messages:
            return []

        # 获取最后一条用户消息作为查询
        last_query = ""
        for m in reversed(self.messages):
            if isinstance(m, HumanMessage):
                last_query = m.content
                break

        if not last_query:
            return self.messages.copy()

        # 向量检索
        try:
            docs = self.vectorstore.similarity_search(last_query, k=self.top_k)
            context = "\n".join(d.page_content for d in docs)
            return [
                SystemMessage(content=f"相关历史:\n{context}"),
                self.messages[-1],  # 只返回当前问题
            ]
        except Exception:
            return self.messages.copy()

    def clear(self) -> None:
        self.messages = []
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)


class FileOffloadingStrategy(ContextStrategy):
    """
    外部文件卸载 (File Offloading)
    公式: C = {Path} ∪ I_t
    长内容转储到外部存储，上下文只留引用
    """

    def __init__(self, max_inline_tokens: int = 500):
        self.messages: list[BaseMessage] = []
        self.file_storage: dict[str, str] = {}
        self.max_inline_tokens = max_inline_tokens
        self._counter = 0

    def add_message(self, msg: BaseMessage) -> None:
        # 超长内容卸载到文件存储
        if len(msg.content) > self.max_inline_tokens:
            file_id = f"file_{self._counter}"
            self.file_storage[file_id] = msg.content
            self._counter += 1

            ref_msg = f"[文件引用: {file_id}]"
            if isinstance(msg, HumanMessage):
                self.messages.append(HumanMessage(content=ref_msg))
            elif isinstance(msg, AIMessage):
                self.messages.append(AIMessage(content=ref_msg))
            else:
                self.messages.append(SystemMessage(content=ref_msg))
        else:
            self.messages.append(msg)

    def get_file(self, file_id: str) -> str | None:
        return self.file_storage.get(file_id)

    def get_messages(self, config=None) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []
        self.file_storage = {}
        self._counter = 0


class HierarchicalMemoryStrategy(ContextStrategy):
    """
    分层记忆架构 (Hierarchical Memory)
    公式: P ∈ {L1, L2, L3}
    L1=工作区, L2=近期记忆, L3=长期摘要
    """

    def __init__(self, llm: BaseChatModel, l1_size: int = 2, l2_size: int = 10):
        self.llm = llm
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l1: list[BaseMessage] = []  # L1 工作区
        self.l2: list[BaseMessage] = []  # L2 近期记忆
        self.l3: str = ""               # L3 长期摘要

    def add_message(self, msg: BaseMessage) -> None:
        # 入 L1
        self.l1.append(msg)

        # L1 满 -> 移到 L2
        if len(self.l1) > self.l1_size:
            moved = self.l1.pop(0)
            self.l2.append(moved)

        # L2 满 -> 压缩到 L3
        if len(self.l2) > self.l2_size:
            self._compress_l2_to_l3()

    def _compress_l2_to_l3(self) -> None:
        if not self.l2:
            return
        old_text = "\n".join(m.content for m in self.l2)
        try:
            resp = self.llm.invoke(f"简洁总结:\n{old_text}")
            self.l3 = (self.l3 + "\n" + resp.content).strip() if self.l3 else resp.content
            self.l2 = self.l2[-self.l2_size // 2:]
        except Exception:
            pass

    def get_messages(self, config=None) -> list[BaseMessage]:
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


class ScopedMemoryStrategy(ContextStrategy):
    """
    上下文范围限制 (Scoped Memory)
    公式: C_i = Scope(C, Role_i)
    Multi-Agent场景下为不同角色过滤上下文
    """

    def __init__(self, role_keywords: dict[str, list[str]] | None = None):
        self.messages: list[BaseMessage] = []
        self.role_keywords = role_keywords or {}

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def get_messages_for_role(self, role: str) -> list[BaseMessage]:
        """获取特定角色可见的消息"""
        keywords = self.role_keywords.get(role, [])
        if not keywords:
            return self.messages.copy()

        result = []
        for msg in self.messages:
            # system消息始终可见
            if isinstance(msg, SystemMessage):
                result.append(msg)
            elif any(kw in msg.content for kw in keywords):
                result.append(msg)
        return result

    def get_messages(self, config=None) -> list[BaseMessage]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages = []


# ============================================================
# 4. 结构化 (Structured)
# ============================================================

class ReflexionStrategy(ContextStrategy):
    """
    言语强化 (Reflexion)
    公式: M = M ∪ Analyze(O, Error)
    存储错误反思，避免重复犯错
    """

    def __init__(self, llm: BaseChatModel):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.reflections: list[str] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)

    def reflect_on_error(self, error: str, action: str) -> str:
        """生成反思规则"""
        prompt = f"分析错误并生成避免规则:\n错误: {error}\n失败动作: {action}"
        try:
            resp = self.llm.invoke(prompt)
            reflection = resp.content
            self.reflections.append(reflection)
            return reflection
        except Exception as e:
            return str(e)

    def get_messages(self, config=None) -> list[BaseMessage]:
        result = []
        if self.reflections:
            refs = "\n".join(self.reflections[-5:])
            result.append(SystemMessage(content=f"经验教训:\n{refs}"))
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages = []
        self.reflections = []


class FactExtractionStrategy(ContextStrategy):
    """
    事实提取 (Fact Extraction)
    公式: M = Extract(I, O) → JSON
    从对话中提取结构化用户信息
    """

    def __init__(self, llm: BaseChatModel):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.facts: dict[str, Any] = {}

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if isinstance(msg, HumanMessage):
            self._extract(msg.content)

    def _extract(self, text: str) -> None:
        prompt = (
            f"从以下文本提取用户信息(JSON格式):\n{text}\n"
            "只返回JSON,不要其他内容.\n"
            "示例格式: {{\"name\":\"\",\"preferences\":[],\"allergies\":[]}}"
        )
        try:
            resp = self.llm.invoke(prompt)
            import json
            extracted = json.loads(resp.content)
            self.facts.update(extracted)
        except Exception:
            pass

    def get_messages(self, config=None) -> list[BaseMessage]:
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


class KnowledgeGraphStrategy(ContextStrategy):
    """
    知识图谱索引 (KG Index)
    公式: C = Query(M, Entity)
    提取实体关系，构建简单图结构
    """

    def __init__(self, llm: BaseChatModel):
        self.messages: list[BaseMessage] = []
        self.llm = llm
        self.entities: list[str] = []
        self.relations: list[dict] = []

    def add_message(self, msg: BaseMessage) -> None:
        self.messages.append(msg)
        if isinstance(msg, HumanMessage):
            self._extract_graph(msg.content)

    def _extract_graph(self, text: str) -> None:
        prompt = (
            f"从文本提取实体和关系:\n{text}\n"
            "格式: 实体: [E1,E2,...]\n关系: [{{\"head\":\"\",\"rel\":\"\",\"tail\":\"\"}},...]"
        )
        try:
            resp = self.llm.invoke(prompt)
            content = resp.content
            if "实体:" in content:
                ent_str = content.split("实体:")[1].split("关系:")[0].strip()
                self.entities.extend(eval(ent_str))
            if "关系:" in content:
                rel_str = content.split("关系:")[1].strip()
                self.relations.extend(eval(rel_str))
        except Exception:
            pass

    def query(self, entity: str) -> list[str]:
        """按实体查询关联信息"""
        results = [entity]
        for rel in self.relations:
            if rel.get("head") == entity:
                results.append(f"{rel['rel']} -> {rel['tail']}")
            elif rel.get("tail") == entity:
                results.append(f"{rel['head']} -> {rel['rel']}")
        return results

    def get_messages(self, config=None) -> list[BaseMessage]:
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
# 工厂函数 & 使用示例
# ============================================================

def create_strategy(
    name: str,
    llm: BaseChatModel | None = None,
    **kwargs,
) -> ContextStrategy:
    """创建上下文管理策略"""
    factories = {
        "full": FullBufferStrategy,
        "sliding": lambda: SlidingWindowStrategy(llm=llm, **kwargs),
        "rolling": lambda: RollingSummaryStrategy(llm=llm, **kwargs),
        "summary_buffer": lambda: SummaryBufferStrategy(llm=llm, **kwargs),
        "map_reduce": lambda: MapReduceStrategy(llm=llm, **kwargs),
        "lingua": lambda: LLMLinguaStrategy(llm=llm, **kwargs),
        "compaction": SemanticCompactionStrategy,
        "rag": lambda: VectorRAGStrategy(llm=llm, **kwargs),
        "offloading": FileOffloadingStrategy,
        "hierarchical": lambda: HierarchicalMemoryStrategy(llm=llm, **kwargs),
        "scoped": ScopedMemoryStrategy,
        "reflexion": lambda: ReflexionStrategy(llm=llm),
        "fact": lambda: FactExtractionStrategy(llm=llm),
        "kg": lambda: KnowledgeGraphStrategy(llm=llm),
    }
    if name not in factories:
        raise ValueError(f"未知策略: {name}. 可选: {list(factories.keys())}")

    factory = factories[name]
    return factory() if callable(factory) else factory


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM 上下文工程 F 函数策略实现 (LangChain)")
    print("=" * 60)

    # 初始化LLM (需要配置 OPENAI_API_KEY)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except Exception:
        llm = None
        print("[提示] 未配置OPENAI_API_KEY, 使用消息模拟演示")

    # --- 1. FullBuffer 全量缓冲 ---
    print("\n[1] FullBuffer 全量缓冲")
    strat = FullBufferStrategy()
    strat.add_message(HumanMessage(content="我叫张三"))
    strat.add_message(AIMessage(content="你好张三!"))
    strat.add_message(HumanMessage(content="我住北京"))
    print(f"    消息数: {len(strat.get_messages())}")

    # --- 2. SlidingWindow 滑动窗口 ---
    print("\n[2] SlidingWindow 滑动窗口 (size=3)")
    from unittest.mock import MagicMock
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock()
    strat = SlidingWindowStrategy(llm=mock_llm, window_size=3)
    for i in range(5):
        strat.add_message(HumanMessage(content=f"消息{i+1}"))
    msgs = strat.get_messages()
    print(f"    保留消息数: {len(msgs)}")

    # --- 3. SummaryBuffer 混合缓冲摘要 ---
    print("\n[3] SummaryBuffer 混合缓冲摘要")
    strat = SummaryBufferStrategy(llm=mock_llm, recent_size=3)
    strat.add_message(HumanMessage(content="消息1"))
    strat.add_message(AIMessage(content="回复1"))
    strat.add_message(HumanMessage(content="消息2"))
    strat.add_message(AIMessage(content="回复2"))
    strat.add_message(HumanMessage(content="消息3"))
    msgs = strat.get_messages()
    print(f"    总消息数(含摘要): {len(msgs)}")

    # --- 4. FileOffloading 文件卸载 ---
    print("\n[4] FileOffloading 文件卸载")
    strat = FileOffloadingStrategy(max_inline_tokens=50)
    strat.add_message(HumanMessage(content="短消息"))
    strat.add_message(HumanMessage(content="长" * 100))  # 触发卸载
    msgs = strat.get_messages()
    print(f"    上下文消息数: {len(msgs)}")
    print(f"    文件存储数: {len(strat.file_storage)}")

    # --- 5. ScopedMemory 范围限制 ---
    print("\n[5] ScopedMemory 范围限制")
    strat = ScopedMemoryStrategy(role_keywords={
        "frontend": ["UI", "按钮", "css"],
        "backend": ["API", "数据库", "sql"],
    })
    strat.add_message(HumanMessage(content="把按钮改成蓝色"))
    strat.add_message(HumanMessage(content="数据库连接超时了"))
    strat.add_message(HumanMessage(content="API返回500错误"))
    front_msgs = strat.get_messages_for_role("frontend")
    back_msgs = strat.get_messages_for_role("backend")
    print(f"    前端Agent可见: {len(front_msgs)} 条")
    print(f"    后端Agent可见: {len(back_msgs)} 条")

    # --- 6. KG 知识图谱 ---
    print("\n[6] KG 知识图谱索引")
    strat = KnowledgeGraphStrategy(llm=mock_llm)
    strat.entities = ["张三", "李四", "星空科技"]
    strat.relations = [
        {"head": "张三", "rel": "夫妻", "tail": "李四"},
        {"head": "李四", "rel": "拥有", "tail": "星空科技"},
    ]
    results = strat.query("张三")
    print(f"    查询 '张三' -> {results}")

    print("\n" + "=" * 60)
    print("  14种策略清单:")
    strategies = [
        "full", "sliding", "rolling", "summary_buffer",
        "map_reduce", "lingua", "compaction", "rag",
        "offloading", "hierarchical", "scoped",
        "reflexion", "fact", "kg",
    ]
    for i, s in enumerate(strategies, 1):
        print(f"    {i:2d}. {s}")
    print("=" * 60)
