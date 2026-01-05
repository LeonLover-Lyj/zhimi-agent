# tests/test_agent.py
"""知觅Agent自动化测试脚本"""
import pytest
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# 延迟导入，避免导入错误影响其他测试
try:
    from zhimi.agent import (
        RecentWindowChatHistory,
        get_recent_messages,
        get_session_history,
        SESSION_STORE,
        HISTORY_WINDOW
    )
    AGENT_IMPORT_OK = True
except ImportError as e:
    AGENT_IMPORT_OK = False
    AGENT_IMPORT_ERROR = str(e)

from zhimi.tools.search_tool import (
    simple_keyword_search,
    hybrid_search,
    build_simple_search_tool,
    build_search_tool
)


@pytest.mark.skipif(not AGENT_IMPORT_OK, reason=f"无法导入agent模块: {AGENT_IMPORT_ERROR if not AGENT_IMPORT_OK else ''}")
class TestConversationHistory:
    """测试对话历史管理功能"""
    
    def test_recent_window_chat_history(self, sample_history):
        """测试RecentWindowChatHistory类是否正确限制历史窗口"""
        window_history = RecentWindowChatHistory(sample_history, k=3)
        
        # 应该只返回最近3轮（6条消息）
        messages = window_history.messages
        assert len(messages) == 6, f"应该返回6条消息，实际返回{len(messages)}条"
        
        # 验证返回的是最近的消息
        assert messages[0].content == "用户问题3"
        assert messages[-1].content == "助手回答5"
    
    def test_recent_window_with_fewer_messages(self, empty_history):
        """测试当历史消息少于窗口大小时的行为"""
        # 添加2轮对话（4条消息）
        empty_history.add_message(HumanMessage(content="问题1"))
        empty_history.add_message(AIMessage(content="回答1"))
        empty_history.add_message(HumanMessage(content="问题2"))
        empty_history.add_message(AIMessage(content="回答2"))
        
        window_history = RecentWindowChatHistory(empty_history, k=3)
        messages = window_history.messages
        
        # 应该返回所有4条消息（少于窗口大小）
        assert len(messages) == 4
    
    def test_get_recent_messages(self, sample_history):
        """测试get_recent_messages函数"""
        recent = get_recent_messages(sample_history, k=2)
        
        # 应该返回最近2轮（4条消息）
        assert len(recent) == 4
        assert recent[0].content == "用户问题4"
        assert recent[-1].content == "助手回答5"
    
    def test_get_session_history(self):
        """测试get_session_history函数"""
        session_id = "test_session_123"
        
        # 第一次调用应该创建新的历史
        history1 = get_session_history(session_id)
        assert isinstance(history1, RecentWindowChatHistory)
        
        # 第二次调用应该返回同一个历史对象（通过full_history）
        history2 = get_session_history(session_id)
        assert history2.full_history is history1.full_history
        
        # 清理
        if session_id in SESSION_STORE:
            del SESSION_STORE[session_id]
    
    def test_history_add_message(self, empty_history):
        """测试历史消息添加功能"""
        window_history = RecentWindowChatHistory(empty_history, k=3)
        
        # 添加消息
        new_message = HumanMessage(content="新问题")
        window_history.add_message(new_message)
        
        # 验证消息被添加到完整历史
        assert len(empty_history.messages) == 1
        assert empty_history.messages[0].content == "新问题"
        
        # 验证窗口历史也能看到新消息
        window_messages = window_history.messages
        assert len(window_messages) == 1
        assert window_messages[0].content == "新问题"


class TestSearchTools:
    """测试搜索工具功能"""
    
    def test_simple_keyword_search_tool_build(self):
        """测试简单关键词检索工具的构建"""
        tool = build_simple_search_tool()
        
        assert tool.name == "simple_keyword_search"
        assert "简单关键词检索" in tool.description
        assert "明确的术语" in tool.description
    
    def test_hybrid_search_tool_build(self):
        """测试混合检索工具的构建"""
        tool = build_search_tool()
        
        assert tool.name == "hybrid_search"
        assert "混合检索" in tool.description
        assert "语义" in tool.description
    
    def test_simple_keyword_search_without_index(self, monkeypatch):
        """测试无索引时简单关键词检索的防御式处理"""
        # 临时设置faiss为None来模拟无索引情况
        import zhimi.tools.search_tool as search_tool_module
        original_faiss = search_tool_module.faiss
        search_tool_module.faiss = None
        
        try:
            result = simple_keyword_search("测试查询")
            assert "本地知识库尚未构建" in result
        finally:
            # 恢复原始值
            search_tool_module.faiss = original_faiss
    
    def test_hybrid_search_without_index(self, monkeypatch):
        """测试无索引时混合检索的防御式处理"""
        import zhimi.tools.search_tool as search_tool_module
        original_faiss = search_tool_module.faiss
        original_bm25 = search_tool_module.bm25
        search_tool_module.faiss = None
        search_tool_module.bm25 = None
        
        try:
            result = hybrid_search("测试查询")
            assert "本地知识库尚未构建" in result
        finally:
            # 恢复原始值
            search_tool_module.faiss = original_faiss
            search_tool_module.bm25 = original_bm25
    
    @pytest.mark.skipif(
        not Path("memory/faiss_index").exists(),
        reason="需要先构建索引"
    )
    def test_simple_keyword_search_with_index(self):
        """测试有索引时简单关键词检索功能"""
        # 这个测试需要实际的索引文件
        result = simple_keyword_search("知觅")
        
        # 应该返回结果或"未找到"消息，不应该抛出异常
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.skipif(
        not Path("memory/faiss_index").exists(),
        reason="需要先构建索引"
    )
    def test_hybrid_search_with_index(self):
        """测试有索引时混合检索功能"""
        # 这个测试需要实际的索引文件
        result = hybrid_search("知觅")
        
        # 应该返回结果或"未找到"消息，不应该抛出异常
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.skipif(not AGENT_IMPORT_OK, reason=f"无法导入agent模块")
class TestAgentIntegration:
    """测试Agent集成功能"""
    
    @pytest.mark.skip(reason="需要API key和完整的LangChain环境")
    def test_agent_tools_registration(self):
        """测试Agent工具注册"""
        from zhimi.agent import load_agent
        
        # 注意：这个测试可能需要API key，所以可能会失败
        # 但我们至少可以验证工具是否正确注册
        try:
            agent = load_agent()
            # 如果能成功加载，说明工具注册正常
            assert agent is not None
        except Exception as e:
            # 如果因为API key等问题失败，跳过这个测试
            pytest.skip(f"Agent加载失败（可能需要API key）: {e}")
    
    def test_tool_descriptions(self):
        """测试工具描述是否正确"""
        simple_tool = build_simple_search_tool()
        hybrid_tool = build_search_tool()
        
        # 验证工具描述包含关键信息
        assert "简单关键词检索" in simple_tool.description
        assert "明确的术语" in simple_tool.description or "具体关键词" in simple_tool.description
        
        assert "混合检索" in hybrid_tool.description
        assert "语义" in hybrid_tool.description or "上下文" in hybrid_tool.description


class TestToolSelection:
    """测试工具选择机制（需要mock LLM）"""
    
    def test_tool_descriptions_for_selection(self):
        """测试工具描述是否足够清晰，便于LLM选择"""
        simple_tool = build_simple_search_tool()
        hybrid_tool = build_search_tool()
        
        # 简单关键词检索应该明确提到"关键词"、"术语"等
        assert any(keyword in simple_tool.description.lower() 
                  for keyword in ["关键词", "术语", "名称", "具体"])
        
        # 混合检索应该明确提到"语义"、"上下文"、"概念"等
        assert any(keyword in hybrid_tool.description.lower() 
                  for keyword in ["语义", "上下文", "概念", "关系", "解释"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

