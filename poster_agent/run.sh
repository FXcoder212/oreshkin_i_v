#!/usr/bin/env bash
# Запуск из исходников (Linux/macOS): при первом запуске создаёт окружение.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m poster_agent "$@"
