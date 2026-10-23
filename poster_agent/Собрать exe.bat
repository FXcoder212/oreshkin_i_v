@echo off
chcp 65001 >nul
rem Сборка PosterAgent.exe на Windows. Результат: dist\PosterAgent.exe
cd /d "%~dp0"
if not exist .venv-build (
    py -3 -m venv .venv-build || python -m venv .venv-build
)
.venv-build\Scripts\python -m pip install -r packaging\requirements-build.txt || (pause & exit /b 1)
.venv-build\Scripts\python packaging\make_icon.py
.venv-build\Scripts\pyinstaller --noconfirm --clean packaging\PosterAgent.spec || (pause & exit /b 1)
echo.
echo Готово: dist\PosterAgent.exe
pause
