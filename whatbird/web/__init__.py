"""Web 前端层（后端）：``REST API -> core``。

只做三件事：参数收发、异常翻译成 HTTP 状态码、托管静态前端。
任何识别逻辑都不在这里实现，保证与 CLI 共用同一份 core 能力。

注意：这里刻意**不**写 ``from .app import app``。那样做会把包属性 ``app``
指向 FastAPI 实例，遮蔽同名子模块 ``whatbird.web.app``，导致
``import whatbird.web.app as mod`` 拿到的是实例而不是模块。
"""

from .app import app as fastapi_app
from .app import create_app

__all__ = ["create_app", "fastapi_app"]
