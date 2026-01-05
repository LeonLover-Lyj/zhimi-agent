# zhimi/asr.py
import os
import io
from typing import Optional, Union
from pathlib import Path
from dotenv import load_dotenv
import requests
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Python 3.13+ 移除了 audioop，pydub 需要它
    PYDUB_AVAILABLE = False
    AudioSegment = None

load_dotenv()

# TeleAI API 配置
TELEAI_API_KEY = os.getenv("TELEAI_API_KEY")
TELEAI_API_URL = os.getenv("TELEAI_API_URL", "https://api.teleai.com/v1/asr")
TELEAI_MODEL = os.getenv("TELEAI_MODEL", "TeleAI/TeleSpeechASR")

# 支持的音频格式
SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]


class ASRError(Exception):
    """语音识别错误"""
    pass


def transcribe_audio(
    audio_data: Union[bytes, str, Path],
    audio_format: Optional[str] = None
) -> str:
    """
    使用 TeleAI/TeleSpeechASR API 将语音转换为文本
    
    Args:
        audio_data: 音频数据，可以是：
            - bytes: 音频文件的二进制数据
            - str: 音频文件路径
            - Path: 音频文件路径对象
        audio_format: 音频格式（如 "wav", "mp3"），如果不提供则自动检测
    
    Returns:
        str: 识别出的文本
    
    Raises:
        ASRError: 当 API 调用失败或配置错误时
    """
    if not TELEAI_API_KEY:
        raise ASRError(
            "未找到 TELEAI_API_KEY，请在 .env 文件中配置。\n"
            "TeleAI 控制台：https://teleai.com/"
        )
    
    # 处理不同类型的输入
    if isinstance(audio_data, (str, Path)):
        audio_path = Path(audio_data)
        if not audio_path.exists():
            raise ASRError(f"音频文件不存在: {audio_path}")
        
        # 读取音频文件
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # 自动检测格式
        if not audio_format:
            audio_format = audio_path.suffix.lower().lstrip(".")
    elif isinstance(audio_data, bytes):
        audio_bytes = audio_data
        if not audio_format:
            # 默认假设为 wav 格式
            audio_format = "wav"
    else:
        raise ASRError(f"不支持的音频数据类型: {type(audio_data)}")
    
    # 验证格式
    if audio_format and f".{audio_format}" not in SUPPORTED_FORMATS:
        raise ASRError(
            f"不支持的音频格式: {audio_format}。"
            f"支持的格式: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {TELEAI_API_KEY}",
    }
    
    # 尝试多种请求格式（根据实际 API 文档调整）
    try:
        # 方式1: multipart/form-data（常见格式）
        files = {
            "file": (f"audio.{audio_format}", audio_bytes, f"audio/{audio_format}")
        }
        data = {
            "model": TELEAI_MODEL,
        }
        
        response = requests.post(
            TELEAI_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            # 尝试多种可能的响应字段名
            text = (
                result.get("text") or
                result.get("transcription") or
                result.get("result") or
                result.get("data", {}).get("text") or
                ""
            )
            if text:
                return text
            else:
                raise ASRError(f"API 响应格式异常，未找到文本字段。响应: {result}")
        else:
            error_msg = f"API 请求失败 (状态码: {response.status_code})"
            try:
                error_detail = response.json()
                error_msg += f"\n错误详情: {error_detail}"
            except:
                error_msg += f"\n响应内容: {response.text[:200]}"
            raise ASRError(error_msg)
            
    except requests.exceptions.RequestException as e:
        raise ASRError(f"网络请求失败: {str(e)}")
    except Exception as e:
        if isinstance(e, ASRError):
            raise
        raise ASRError(f"语音识别失败: {str(e)}")


def convert_audio_format(
    audio_data: bytes,
    input_format: str,
    output_format: str = "wav",
    sample_rate: int = 16000
) -> bytes:
    """
    转换音频格式
    
    Args:
        audio_data: 输入音频数据（bytes）
        input_format: 输入格式（如 "wav", "mp3"）
        output_format: 输出格式（默认 "wav"）
        sample_rate: 采样率（默认 16000 Hz）
    
    Returns:
        bytes: 转换后的音频数据
    """
    if not PYDUB_AVAILABLE:
        # Python 3.13+ 中 pydub 不可用，直接返回原始数据
        # 假设 API 可以处理原始格式
        return audio_data
    
    try:
        # 使用 pydub 转换格式
        audio = AudioSegment.from_file(
            io.BytesIO(audio_data),
            format=input_format
        )
        
        # 设置采样率
        audio = audio.set_frame_rate(sample_rate)
        
        # 转换为单声道（如果需要）
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # 导出为指定格式
        output_buffer = io.BytesIO()
        audio.export(output_buffer, format=output_format)
        
        return output_buffer.getvalue()
    except Exception as e:
        raise ASRError(f"音频格式转换失败: {str(e)}")

