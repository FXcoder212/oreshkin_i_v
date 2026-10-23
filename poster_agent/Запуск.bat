@echo off
chcp 65001 >nul
rem Запуск из исходников: при первом запуске создаёт окружение и ставит зависимости.
cd /d "%~dp0"
if not exist .venv (
    echo Первый запуск: устанавливаю зависимости...
    py -3 -m venv .venv || python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip >nul
    .venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
)
start "" .venv\Scripts\pythonw -m poster_agent
