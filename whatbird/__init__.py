"""whatbird：基于火山引擎方舟 Doubao-Seed-2.1 的鸟类识别应用。

分层约定（对应 Agent.md 第 4 条「前后端分离」）：

- ``whatbird.core``  纯业务核心层，不依赖任何 Web 框架，可被任意前端复用
- ``whatbird.cli``   CLI 前端  ->  core
- ``whatbird.web``   Web 前端  ->  core
"""

__version__ = "0.1.0"
