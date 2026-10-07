@echo off
rem ============================================================
rem  WhatBird launcher
rem
rem  Interpreter selection order:
rem    1) WHATBIRD_PYTHON environment variable
rem    2) .venv\Scripts\python.exe            (project venv, recommended)
rem    3) %USERPROFILE%\whatbird-venv\Scripts\python.exe  (fallback)
rem    4) python on PATH
rem
rem  First-time setup (see README section 2):
rem    cd /d <project dir>
rem    python -m venv .venv
rem    .venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-dev.txt
rem
rem  Usage:
rem    whatbird.cmd config doctor
rem    whatbird.cmd config set --api-key -
rem    whatbird.cmd identify D:\photos\bird.jpg
rem    whatbird.cmd serve --port 8000
rem
rem  NOTE: keep this file ASCII-only and CRLF-terminated.
rem        cmd.exe reads .cmd with the OEM code page (GBK on Chinese
rem        Windows); non-ASCII text makes it mis-parse the script.
rem ============================================================
setlocal

set "PY=%WHATBIRD_PYTHON%"
if defined PY goto run

if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if defined PY goto run

if exist "%USERPROFILE%\whatbird-venv\Scripts\python.exe" set "PY=%USERPROFILE%\whatbird-venv\Scripts\python.exe"
if defined PY goto run

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] No usable Python found.
  echo         Create the project venv first, see README section 2,
  echo         or set WHATBIRD_PYTHON to a full interpreter path.
  exit /b 2
)
set "PY=python"

:run
"%PY%" -m whatbird.cli %*
exit /b %ERRORLEVEL%
