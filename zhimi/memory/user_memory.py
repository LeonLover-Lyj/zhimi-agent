"""用户记忆管理模块"""
from typing import Dict, Any, List, Optional
from zhimi.memory.memory_storage import UserMemoryStorage
from zhimi.memory.memory_extractor import MemoryExtractor


class UserMemory:
    """用户记忆管理主类"""
    
    def __init__(self, user_id: str = "default_user", storage_path: str = "memory/user_memory.json"):
        """
        初始化用户记忆管理器
        
        Args:
            user_id: 用户ID
            storage_path: 存储文件路径
        """
        self.user_id = user_id
        self.storage = UserMemoryStorage(storage_path)
        self.extractor = MemoryExtractor()
        self._memory_cache: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """加载用户记忆"""
        if self._memory_cache is None:
            self._memory_cache = self.storage.load_memory(self.user_id)
        return self._memory_cache
    
    def save(self) -> bool:
        """保存用户记忆"""
        if self._memory_cache is not None:
            success = self.storage.save_memory(self.user_id, self._memory_cache)
            if success:
                return True
        return False
    
    def update_from_conversation(self, conversation: List[str]) -> bool:
        """
        从对话中自动提取并更新记忆
        
        Args:
            conversation: 对话消息列表
        
        Returns:
            是否更新成功
        """
        # 提取用户信息
        extracted_info = self.extractor.extract_user_info(conversation)
        
        # 检查是否有有效信息
        has_info = (
            any(extracted_info["preferences"].values()) or
            any(extracted_info["background"].values())
        )
        
        if not has_info:
            return False
        
        # 更新记忆
        return self.update_memory(extracted_info)
    
    def update_from_messages(self, messages: List[Any]) -> bool:
        """
        从LangChain消息对象中提取并更新记忆
        
        Args:
            messages: LangChain消息列表
        
        Returns:
            是否更新成功
        """
        extracted_info = self.extractor.extract_from_messages(messages)
        
        # 检查是否有有效信息
        has_info = (
            any(extracted_info["preferences"].values()) or
            any(extracted_info["background"].values())
        )
        
        if not has_info:
            return False
        
        return self.update_memory(extracted_info)
    
    def update_memory(self, memory_updates: Dict[str, Any]) -> bool:
        """
        更新记忆（增量更新）
        
        Args:
            memory_updates: 要更新的记忆片段
        
        Returns:
            是否更新成功
        """
        memory = self.load()
        success = self.storage.update_memory(self.user_id, memory_updates)
        if success:
            self._memory_cache = None  # 清除缓存，下次加载最新数据
        return success
    
    def get_memory_summary(self) -> str:
        """
        获取记忆摘要（用于注入到系统提示词）
        
        Returns:
            格式化的记忆摘要文本
        """
        memory = self.load()
        
        summary_parts = []
        
        # 用户偏好
        prefs = memory.get("preferences", {})
        pref_items = []
        if prefs.get("programming_languages"):
            pref_items.append(f"- 编程语言：{', '.join(prefs['programming_languages'])}")
        if prefs.get("tools"):
            pref_items.append(f"- 工具偏好：{', '.join(prefs['tools'])}")
        if prefs.get("topics"):
            pref_items.append(f"- 话题偏好：{', '.join(prefs['topics'])}")
        
        if pref_items:
            summary_parts.append("### 用户偏好")
            summary_parts.extend(pref_items)
        
        # 用户背景
        bg = memory.get("background", {})
        bg_items = []
        if bg.get("profession"):
            bg_items.append(f"- 职业：{bg['profession']}")
        if bg.get("experience"):
            bg_items.append(f"- 经验：{bg['experience']}")
        if bg.get("projects"):
            bg_items.append(f"- 项目：{', '.join(bg['projects'])}")
        
        if bg_items:
            summary_parts.append("### 用户背景")
            summary_parts.extend(bg_items)
        
        if summary_parts:
            return "## 用户记忆\n\n" + "\n".join(summary_parts)
        return ""
    
    def get_preferences(self) -> Dict[str, Any]:
        """获取用户偏好"""
        memory = self.load()
        return memory.get("preferences", {})
    
    def get_background(self) -> Dict[str, Any]:
        """获取用户背景"""
        memory = self.load()
        return memory.get("background", {})
    
    def clear(self) -> bool:
        """清空用户记忆"""
        success = self.storage.clear_memory(self.user_id)
        if success:
            self._memory_cache = None
        return success
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有记忆"""
        return self.load()

