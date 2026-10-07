@echo off
rem ============================================================
rem  WhatBird 启动器
rem
rem  解释器挑选顺序：
rem    1) 环境变量 WHATBIRD_PYTHON 指定的解释器
rem    2) 项目内 .venv\Scripts\python.exe        （推荐）
rem    3) %USERPROFILE%\whatbird-venv\Scripts\python.exe（备用）
rem    4) PATH 上的 python
rem
rem  首次使用请先创建项目虚拟环境（见 README 第 2 节）：
rem    cd /d <本项目目录>
rem    python -m venv .venv
rem    .venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-dev.txt
rem
rem  用法：
rem    whatbird.cmd config doctor
rem    whatbird.cmd config set --api-key -
rem    whatbird.cmd identify D:\photos\bird.jpg
rem    whatbird.cmd serve --port 8000
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
  echo [错误] 找不到可用的 Python。
  echo        请先创建项目虚拟环境（README 第 2 节），
  echo        或设置 WHATBIRD_PYTHON 指向解释器的完整路径。
  exit /b 2
)
set "PY=python"

:run
"%PY%" -m whatbird.cli %*
exit /b %ERRORLEVEL%
