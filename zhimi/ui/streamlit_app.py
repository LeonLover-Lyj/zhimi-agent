# zhimi/ui/streamlit_app.py
import sys
import os
from pathlib import Path

# 确保项目根目录在 Python 路径中
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# 切换到项目根目录，确保 .env 文件能被正确加载
os.chdir(project_root)

# 加载环境变量（必须在导入其他模块之前）
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from zhimi.agent import (
    load_agent, 
    SESSION_STORE, 
    HISTORY_WINDOW,
    get_user_memory,
    update_user_memory_from_conversation
)
from zhimi.asr import transcribe_audio, ASRError

st.set_page_config(page_title="知觅 Agent", page_icon="🌿")
st.title("🌿 知觅 – Qwen + 本地知识库")

# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []

SESSION_ID = "default_streamlit"


def process_user_input(prompt: str, session_id: str):
    """处理用户文本输入"""
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # 调用Agent
    with st.spinner("正在思考中..."):
        try:
            response = st.session_state.agent.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )
            assistant_response = response.get("output", "抱歉，我无法回答这个问题。")
            
            # 添加助手回复到历史
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.chat_message("assistant").write(assistant_response)
            
            # 自动更新用户记忆（从对话历史中提取）
            update_memory_if_needed(session_id)
            
        except Exception as e:
            handle_agent_error(e)


def process_audio_input(audio_data: bytes, session_id: str, audio_format: str):
    """处理用户语音输入"""
    with st.spinner("正在识别语音..."):
        try:
            # 调用 ASR API 进行语音识别
            transcribed_text = transcribe_audio(audio_data, audio_format)
            
            if transcribed_text:
                # 显示识别结果
                st.success(f"✅ 识别结果：{transcribed_text}")
                
                # 将识别文本作为用户输入处理
                process_user_input(transcribed_text, session_id)
            else:
                st.warning("⚠️ 未能识别出文本内容，请重试。")
                
        except ASRError as e:
            error_msg = f"❌ **语音识别失败**\n\n{str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)
        except Exception as e:
            error_msg = f"❌ **处理语音时发生错误**\n\n{str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)


def update_memory_if_needed(session_id: str):
    """在需要时更新用户记忆"""
    if session_id in SESSION_STORE:
        full_history = SESSION_STORE[session_id]
        if len(full_history.messages) >= 2:  # 至少有一轮对话
            # 只在对话轮数达到一定数量时更新（避免频繁调用LLM）
            # 每2轮对话（4条消息）更新一次记忆
            if len(full_history.messages) % 4 == 0:
                try:
                    # 异步更新记忆（不阻塞UI）
                    memory_updated = update_user_memory_from_conversation(session_id, full_history.messages)
                    if memory_updated:
                        # 重新加载Agent以更新系统提示词中的记忆
                        st.session_state.agent = load_agent(session_id)
                except Exception as e:
                    # 记忆更新失败不影响对话，静默处理
                    pass


def handle_agent_error(e: Exception):
    """处理 Agent 调用错误"""
    error_str = str(e)
    error_msg = "❌ 发生错误"
    
    # 检查是否是API访问权限错误
    if "AccessDenied" in error_str or "拒绝访问模型" in error_str or "403" in error_str:
        error_msg = """❌ **API访问权限错误**

**问题：** 当前账户无法访问配置的模型

**解决方案：**
1. 检查 `.env` 文件中的 `LLM_MODEL` 设置是否正确
2. 确认模型名称格式：`Qwen/Qwen2.5-7B-Instruct`
3. 在硅基流动控制台检查账户余额和API配额
4. 访问 https://cloud.siliconflow.cn/ 查看模型访问权限

**当前配置的模型：** 请检查 `.env` 文件中的 `LLM_MODEL` 设置"""
    elif "SILICONFLOW_API_KEY" in error_str or "API" in error_str or "api_key" in error_str.lower():
        error_msg = """❌ **API配置错误**

**问题：** API key未配置或无效

**解决方案：**
1. 检查 `.env` 文件中是否配置了 `SILICONFLOW_API_KEY`
2. 确认API key是否正确（以 `sk-` 开头）
3. 确认 `.env` 文件在项目根目录（`E:\\zhimi-agent\\.env`）
4. 访问 https://cloud.siliconflow.cn/ 获取或查看API key
5. 重启 Streamlit 应用（修改 `.env` 后需要重启）"""
    else:
        error_msg = f"❌ **发生错误**\n\n{error_str}"
    
    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    st.chat_message("assistant").write(error_msg)
    st.error(f"错误详情：{error_str}")


# 侧边栏：显示对话统计信息和用户记忆
with st.sidebar:
    st.header("📊 对话统计")
    
    # 获取对话历史轮数
    if SESSION_ID in SESSION_STORE:
        full_history = SESSION_STORE[SESSION_ID]
        total_messages = len(full_history.messages)
        # 计算对话轮数（每轮包含用户消息和助手消息）
        total_turns = total_messages // 2
        st.metric("总对话轮数", total_turns)
        st.metric("当前使用历史窗口", f"最近 {HISTORY_WINDOW} 轮")
    else:
        st.metric("总对话轮数", 0)
        st.metric("当前使用历史窗口", f"最近 {HISTORY_WINDOW} 轮")
    
    st.divider()
    
    # 显示用户记忆
    st.header("🧠 用户记忆")
    user_memory = get_user_memory(SESSION_ID)
    memory_data = user_memory.get_all()
    
    # 显示偏好
    prefs = memory_data.get("preferences", {})
    if any(prefs.values()):
        with st.expander("📌 用户偏好", expanded=False):
            if prefs.get("programming_languages"):
                st.write(f"**编程语言：** {', '.join(prefs['programming_languages'])}")
            if prefs.get("tools"):
                st.write(f"**工具偏好：** {', '.join(prefs['tools'])}")
            if prefs.get("topics"):
                st.write(f"**话题偏好：** {', '.join(prefs['topics'])}")
    else:
        st.caption("暂无偏好信息")
    
    # 显示背景
    bg = memory_data.get("background", {})
    if bg.get("profession") or bg.get("experience") or bg.get("projects"):
        with st.expander("👤 用户背景", expanded=False):
            if bg.get("profession"):
                st.write(f"**职业：** {bg['profession']}")
            if bg.get("experience"):
                st.write(f"**经验：** {bg['experience']}")
            if bg.get("projects"):
                st.write(f"**项目：** {', '.join(bg['projects'])}")
    else:
        st.caption("暂无背景信息")
    
    # 清空记忆按钮
    if st.button("🗑️ 清空记忆", use_container_width=True):
        user_memory.clear()
        st.success("记忆已清空")
        st.rerun()
    
    st.divider()
    st.info("💡 提示：Agent会自动从对话中提取并记住你的偏好和背景信息")

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 加载Agent（延迟加载，避免重复初始化）
if "agent" not in st.session_state:
    with st.spinner("正在初始化Agent..."):
        st.session_state.agent = load_agent(SESSION_ID)

# 输入方式选择
input_tab1, input_tab2 = st.tabs(["📝 文本输入", "🎤 语音输入"])

# 文本输入标签页
with input_tab1:
    prompt = st.chat_input("例如：知觅支持哪些功能？")
    if prompt:
        process_user_input(prompt, SESSION_ID)

# 语音输入标签页
with input_tab2:
    st.markdown("### 方式一：浏览器录音")
    audio_bytes = audio_recorder(
        text="点击开始录音",
        recording_color="#e74c3c",
        neutral_color="#34495e",
        icon_name="microphone",
        icon_size="2x",
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("🎯 识别并发送", type="primary", use_container_width=True):
            process_audio_input(audio_bytes, SESSION_ID, "wav")
    
    st.divider()
    st.markdown("### 方式二：上传音频文件")
    uploaded_file = st.file_uploader(
        "选择音频文件",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        help="支持格式：WAV, MP3, M4A, OGG, FLAC, WEBM"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=uploaded_file.type)
        audio_data = uploaded_file.read()
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if st.button("🎯 识别并发送", type="primary", use_container_width=True, key="upload_recognize"):
            process_audio_input(audio_data, SESSION_ID, file_extension)
