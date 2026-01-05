#!/usr/bin/env bash
set -e
VENV_NAME=zhimi-agent

if [ ! -d "$VENV_NAME" ]; then
  echo "❌ 未找到虚拟环境，请先执行 STEP_003"
  exit 1
fi

source $VENV_NAME/bin/activate

if [ ! -d memory/faiss_index ] || [ -z "$(ls -A memory/faiss_index)" ]; then
  python scripts/index_local_docs.py --dir data
fi

streamlit run zhimi/ui/streamlit_app.py

