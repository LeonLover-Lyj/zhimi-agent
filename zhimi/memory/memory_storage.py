"""用户记忆存储模块"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class UserMemoryStorage:
    """用户记忆存储类（JSON文件存储）"""
    
    def __init__(self, storage_path: str = "memory/user_memory.json"):
        """
        初始化记忆存储
        
        Args:
            storage_path: 存储文件路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_memory(self, user_id: str = "default_user") -> Dict[str, Any]:
        """
        加载用户记忆
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户记忆字典，如果不存在则返回默认结构
        """
        if not self.storage_path.exists():
            return self._get_default_memory(user_id)
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 如果数据中没有该用户，返回默认结构
            if user_id not in data:
                return self._get_default_memory(user_id)
            
            return data[user_id]
        except (json.JSONDecodeError, KeyError, IOError) as e:
            print(f"⚠️ 加载记忆失败: {e}，使用默认记忆")
            return self._get_default_memory(user_id)
    
    def save_memory(self, user_id: str, memory: Dict[str, Any]) -> bool:
        """
        保存用户记忆
        
        Args:
            user_id: 用户ID
            memory: 记忆字典
        
        Returns:
            是否保存成功
        """
        try:
            # 加载所有用户数据
            all_data = {}
            if self.storage_path.exists():
                try:
                    with open(self.storage_path, "r", encoding="utf-8") as f:
                        all_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    all_data = {}
            
            # 更新时间戳
            memory["updated_at"] = datetime.now().isoformat()
            
            # 更新该用户的记忆
            all_data[user_id] = memory
            
            # 保存到文件
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            return True
        except IOError as e:
            print(f"❌ 保存记忆失败: {e}")
            return False
    
    def update_memory(self, user_id: str, memory_updates: Dict[str, Any]) -> bool:
        """
        更新用户记忆（增量更新）
        
        Args:
            user_id: 用户ID
            memory_updates: 要更新的记忆片段
        
        Returns:
            是否更新成功
        """
        current_memory = self.load_memory(user_id)
        
        # 深度合并更新
        self._deep_merge(current_memory, memory_updates)
        
        return self.save_memory(user_id, current_memory)
    
    def _deep_merge(self, base: Dict, updates: Dict) -> None:
        """深度合并字典"""
        for key, value in updates.items():
            if key == "updated_at":
                continue  # 跳过时间戳，会在save时更新
            
            if key not in base:
                base[key] = value
                continue
            
            if key == "preferences" or key == "background":
                # 对于偏好和背景，特殊处理
                if isinstance(value, dict) and isinstance(base[key], dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in base[key] and isinstance(base[key][sub_key], list):
                            # 合并列表，去重
                            if isinstance(sub_value, list):
                                # 合并列表并去重
                                combined = base[key][sub_key] + sub_value
                                base[key][sub_key] = list(dict.fromkeys(combined))  # 保持顺序的去重
                            elif sub_value and sub_value not in base[key][sub_key]:
                                base[key][sub_key].append(sub_value)
                        elif sub_key in base[key] and isinstance(base[key][sub_key], str):
                            # 字符串类型：如果新值非空，则更新
                            if sub_value and sub_value.strip():
                                base[key][sub_key] = sub_value
                        else:
                            # 新键或类型不匹配，直接赋值
                            base[key][sub_key] = sub_value
                else:
                    base[key] = value
            elif isinstance(base[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                self._deep_merge(base[key], value)
            else:
                # 直接替换
                base[key] = value
    
    def _get_default_memory(self, user_id: str) -> Dict[str, Any]:
        """获取默认记忆结构"""
        return {
            "user_id": user_id,
            "preferences": {
                "programming_languages": [],
                "tools": [],
                "topics": []
            },
            "background": {
                "profession": "",
                "experience": "",
                "projects": []
            },
            "updated_at": datetime.now().isoformat()
        }
    
    def clear_memory(self, user_id: str) -> bool:
        """
        清空用户记忆
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否清空成功
        """
        return self.save_memory(user_id, self._get_default_memory(user_id))

