@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 請先建立專案的 .venv Python 環境。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u scripts\demo.py --publish
pause
