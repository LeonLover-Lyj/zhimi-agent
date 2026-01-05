# zhimi/agent.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from zhimi.llm import get_llm
from zhimi.tools.search_tool import build_search_tool, build_simple_search_tool
from zhimi.memory import UserMemory

# 会话存储
SESSION_STORE = {}
# 历史窗口大小（保留最近k轮对话）
HISTORY_WINDOW = 3
# 用户记忆实例字典（支持多用户）
_user_memory_store: dict = {}

class RecentWindowChatHistory(BaseChatMessageHistory):
    """包装聊天历史，只返回最近k轮对话"""
    
    def __init__(self, full_history: InMemoryChatMessageHistory, k: int = 3):
        self.full_history = full_history
        self.k = k
    
    @property
    def messages(self) -> List[BaseMessage]:
        """返回最近k轮对话消息"""
        all_messages = self.full_history.messages
        # 提取最近 2*k 条消息（k轮 = k个用户消息 + k个助手消息）
        if len(all_messages) > 2 * self.k:
            return all_messages[-2 * self.k:]
        return all_messages
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息到完整历史"""
        self.full_history.add_message(message)
    
    def clear(self) -> None:
        """清空完整历史"""
        self.full_history.clear()

def get_session_history(session_id: str):
    """获取或创建会话历史，返回带窗口限制的历史对象"""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    
    full_history = SESSION_STORE[session_id]
    # 返回包装后的历史，只暴露最近k轮
    return RecentWindowChatHistory(full_history, k=HISTORY_WINDOW)

def get_recent_messages(history: InMemoryChatMessageHistory, k: int = 3) -> List[BaseMessage]:
    """从完整历史中提取最近k轮对话（每轮包含用户消息和助手消息）
    
    Args:
        history: 完整的对话历史
        k: 要保留的轮数，默认3轮
    
    Returns:
        最近k轮的对话消息列表
    """
    all_messages = history.messages
    # 提取最近 2*k 条消息（k轮 = k个用户消息 + k个助手消息）
    if len(all_messages) > 2 * k:
        return all_messages[-2 * k:]
    return all_messages

def get_user_memory(user_id: str = "default_user") -> UserMemory:
    """获取用户记忆实例（每个用户独立实例）"""
    global _user_memory_store
    if user_id not in _user_memory_store:
        _user_memory_store[user_id] = UserMemory(user_id=user_id)
    return _user_memory_store[user_id]


def load_agent(user_id: str = "default_user"):
    """加载带有记忆的Agent"""
    llm = get_llm()
    # 注册两个搜索工具：简单关键词检索和混合检索
    tools = [build_simple_search_tool(), build_search_tool()]
    
    # 获取用户记忆
    user_memory = get_user_memory(user_id)
    memory_summary = user_memory.get_memory_summary()
    
    # 构建系统消息
    base_system_message = """你是一个名为「知觅」的智能助手，能使用工具来查询本地文档。

## 重要规则

### 1. 对话历史使用规则
- 优先参考最近2-3轮对话上下文来理解当前问题
- 如果用户的问题涉及之前的对话内容，请结合历史上下文回答
- 保持回答的连贯性和上下文相关性

### 2. 工具调用决策规则
- **不要调用工具的情况**：
  - 常识性问题（如"什么是Python"、"如何计算1+1"）
  - 闲聊、问候（如"你好"、"谢谢"）
  - 通用知识问题（不涉及本地文档或项目）
  
- **必须调用工具的情况**：
  - 问题涉及本地文档、项目说明、技术实现细节
  - 需要查询具体的配置、功能、使用方法等

### 3. 工具选择规则
- **simple_keyword_search（简单关键词检索）**：
  - 适用于：明确的术语、名称、具体关键词查询
  - 使用场景：用户询问具体的名称、术语、关键词
  - 示例："知觅是什么"、"如何安装"、"配置文件位置"
  
- **hybrid_search（混合检索）**：
  - 适用于：需要理解语义、上下文、概念的问题
  - 使用场景：需要理解含义、上下文关系、概念解释
  - 示例："解释一下工作原理"、"它们之间的关系是什么"、"这个概念如何应用"

### 4. 回答要求
- 使用自然、流畅的中文回答
- 如果调用了工具，要基于工具返回的结果进行回答
- 如果工具返回"未找到相关信息"，如实告知用户
- 保持回答的准确性和相关性"""
    
    # 如果有用户记忆，添加到系统消息中
    if memory_summary:
        system_message = base_system_message + "\n\n" + memory_summary + "\n\n### 5. 用户记忆使用规则\n- 在回答时，可以参考用户的偏好和背景信息\n- 根据用户的背景调整回答的详细程度和技术深度\n- 如果用户提到新的偏好或背景信息，可以自然地回应"
    else:
        system_message = base_system_message
    
    # 创建自定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 创建基于工具调用的 Agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # 添加记忆（get_session_history已自动限制为最近3轮）
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,  # 注意：这里传递的是agent_executor
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return agent_with_history


def update_user_memory_from_conversation(user_id: str, messages: List[BaseMessage]) -> bool:
    """
    从对话中自动提取并更新用户记忆
    
    Args:
        user_id: 用户ID
        messages: LangChain消息列表
    
    Returns:
        是否更新成功
    """
    user_memory = get_user_memory(user_id)
    return user_memory.update_from_messages(messages)