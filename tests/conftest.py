# tests/conftest.py
"""pytest配置文件，包含测试fixtures和配置"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

@pytest.fixture
def sample_history():
    """创建示例对话历史"""
    history = InMemoryChatMessageHistory()
    # 添加5轮对话（10条消息）
    for i in range(5):
        history.add_message(HumanMessage(content=f"用户问题{i+1}"))
        history.add_message(AIMessage(content=f"助手回答{i+1}"))
    return history

@pytest.fixture
def empty_history():
    """创建空的对话历史"""
    return InMemoryChatMessageHistory()

