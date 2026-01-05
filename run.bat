@echo off
setlocal
set VENV_NAME=zhimi-agent

if not exist "%VENV_NAME%" (
    echo ❌ 未找到虚拟环境，请先执行构建步骤
    exit /b 1
)

call %VENV_NAME%\Scripts\activate.bat

if not exist memory\faiss_index (
    python scripts\index_local_docs.py --dir data
) else (
    dir /b memory\faiss_index\index.faiss >nul 2>&1
    if errorlevel 1 (
        python scripts\index_local_docs.py --dir data
    )
)

streamlit run zhimi\ui\streamlit_app.py

