"""用户记忆提取模块"""
import json
import re
from typing import Dict, Any, List
from zhimi.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


class MemoryExtractor:
    """用户信息提取器（使用LLM自动提取）"""
    
    EXTRACTION_PROMPT = """你是一个信息提取助手。请从以下对话中提取用户信息，重点关注：
1. **用户偏好**（编程语言、工具、技术栈、话题偏好等）
2. **用户背景**（职业、工作经验、项目经历等）

对话内容：
{conversation}

请以JSON格式返回提取的信息，只返回JSON，不要其他文字说明：
{{
    "preferences": {{
        "programming_languages": [],
        "tools": [],
        "topics": []
    }},
    "background": {{
        "profession": "",
        "experience": "",
        "projects": []
    }}
}}

如果对话中没有相关信息，请返回空值（空字符串或空列表）。"""
    
    def __init__(self):
        """初始化提取器"""
        self.llm = get_llm()
    
    def extract_user_info(self, conversation: List[str]) -> Dict[str, Any]:
        """
        从对话中提取用户信息
        
        Args:
            conversation: 对话消息列表（格式：["用户: xxx", "助手: xxx", ...]）
        
        Returns:
            提取的用户信息字典
        """
        if not conversation:
            return self._get_empty_info()
        
        # 构建对话文本
        conversation_text = "\n".join(conversation[-6:])  # 只分析最近6条消息
        
        # 构建提示词
        prompt = self.EXTRACTION_PROMPT.format(conversation=conversation_text)
        
        try:
            # 调用LLM提取信息
            messages = [
                SystemMessage(content="你是一个信息提取助手，专门从对话中提取用户偏好和背景信息。"),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            # 处理不同LLM返回格式
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            # 解析JSON响应
            extracted_info = self._parse_extraction_result(content)
            return extracted_info
            
        except Exception as e:
            print(f"⚠️ 提取用户信息失败: {e}")
            return self._get_empty_info()
    
    def _parse_extraction_result(self, content: str) -> Dict[str, Any]:
        """
        解析LLM返回的提取结果
        
        Args:
            content: LLM返回的内容
        
        Returns:
            解析后的信息字典
        """
        # 尝试提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content
        
        try:
            extracted = json.loads(json_str)
            
            # 验证结构
            result = self._get_empty_info()
            if "preferences" in extracted:
                result["preferences"].update(extracted["preferences"])
            if "background" in extracted:
                result["background"].update(extracted["background"])
            
            return result
        except json.JSONDecodeError:
            print(f"⚠️ JSON解析失败，原始内容: {content[:200]}")
            return self._get_empty_info()
    
    def _get_empty_info(self) -> Dict[str, Any]:
        """返回空的信息结构"""
        return {
            "preferences": {
                "programming_languages": [],
                "tools": [],
                "topics": []
            },
            "background": {
                "profession": "",
                "experience": "",
                "projects": []
            }
        }
    
    def extract_from_messages(self, messages: List[Any]) -> Dict[str, Any]:
        """
        从LangChain消息对象中提取用户信息
        
        Args:
            messages: LangChain消息列表
        
        Returns:
            提取的用户信息字典
        """
        # 转换为文本格式
        conversation = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                role = "用户" if msg.type == "human" else "助手"
                conversation.append(f"{role}: {msg.content}")
        
        return self.extract_user_info(conversation)

