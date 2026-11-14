#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  echo '首次使用，正在准备运行环境…'
  python3 -m venv .venv || { echo '请先安装 Python 3.10 或以上版本。'; read; exit 1; }
fi
if ! .venv/bin/python -c 'import fastapi, uvicorn, tradingagents.graph.trading_graph' >/dev/null 2>&1; then
  echo '首次使用，正在安装依赖，可能需要几分钟…'
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e '.[web]' || { read; exit 1; }
  else
    .venv/bin/python -m ensurepip --upgrade
    .venv/bin/python -m pip install -e '.[web]' || { read; exit 1; }
  fi
fi
.venv/bin/python scripts/local_web.py
if [[ $? -ne 0 ]]; then
  echo '启动遇到问题，请保留此窗口中的提示。按回车关闭。'
  read
fi
