@echo off
rem ============================================================
rem  WhatBird 启动器
rem
rem  为什么特意不用项目内的 .venv？
rem  本机实测：位于工作区内的解释器被限制为「只能写工作区」，
rem  配置写不进用户目录，只能降级到项目内位置；
rem  而项目外的解释器不受此限，能正常写入 ~/.whatbird/config.json。
rem
rem  解释器挑选顺序：
rem    1) 环境变量 WHATBIRD_PYTHON 指定的解释器
rem    2) %USERPROFILE%\whatbird-venv\Scripts\python.exe（推荐：项目外独立环境）
rem    3) PATH 上的 python
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

if exist "%USERPROFILE%\whatbird-venv\Scripts\python.exe" set "PY=%USERPROFILE%\whatbird-venv\Scripts\python.exe"
if defined PY goto run

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 找不到可用的 Python。
  echo        请安装 Python，或设置 WHATBIRD_PYTHON 指向解释器的完整路径。
  exit /b 2
)
set "PY=python"

:run
"%PY%" -m whatbird.cli %*
exit /b %ERRORLEVEL%
