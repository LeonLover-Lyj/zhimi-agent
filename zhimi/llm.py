# zhimi/llm.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# 默认使用硅基流动上的 Qwen2.5-7B-Instruct
DEFAULT_MODEL = "Qwen2.5-7B-Instruct"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


def get_llm(model_name: str = None) -> BaseChatModel:
    """获取LLM实例（使用硅基流动 OpenAI 兼容接口）
    
    Args:
        model_name: 可选，自定义模型名称；不传则优先用环境变量 LLM_MODEL，其次用默认模型
    
    Returns:
        BaseChatModel: LangChain ChatModel 实例（ChatOpenAI 封装的硅基流动模型）
    
    Note:
        - 需要配置 SILICONFLOW_API_KEY 环境变量
        - 默认模型：Qwen/Qwen2.5-7B-Instruct（硅基流动免费模型）
        - 硅基流动控制台：https://cloud.siliconflow.cn/
    """
    api_key = os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 SILICONFLOW_API_KEY，请在 .env 文件中配置。\n"
            "硅基流动控制台：https://cloud.siliconflow.cn/"
        )
    
    # 确定模型名称：参数 > 环境变量 > 默认值
    if model_name:
        selected_model = model_name
    else:
        selected_model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    
    return ChatOpenAI(
        model=selected_model,
        api_key=api_key,
        base_url=SILICONFLOW_BASE_URL,
        temperature=0.2,
    )