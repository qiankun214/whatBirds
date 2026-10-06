"""FastAPI 后端：``REST API -> core``，并托管静态前端（前后端分离）。

接口一览（详见 ``/docs`` 自动生成的 OpenAPI 文档）：

===========================  ======  =================================================
方法                          路径     说明
===========================  ======  =================================================
GET                          /api/health          健康检查
GET                          /api/config          读取配置（API Key 已脱敏）
PUT                          /api/config          写入配置（持久化到用户目录）
POST                         /api/config/verify   验证配置能否调通方舟
POST                         /api/recognize       上传图片文件识别（multipart）
POST                         /api/recognize-path  传入服务器上的 jpg 路径识别
GET                          /                    静态前端页面
===========================  ======  =================================================

安全说明（本地工具，仍需防护）：
- 所有会改动状态或消耗额度的接口都要求请求头 ``X-WhatBird-Client: 1``。
  跨站请求无法自定义该头（会触发 CORS 预检而被浏览器拦下），可挡住恶意网页
  向本机服务偷偷提交请求。
- 本应用不开启 CORS，前端与后端同源，无需跨域。
- 配置里的 API Key 永远以脱敏形式返回，不会下发到浏览器明文。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..core import (
    BirdRecognizer,
    ConfigError,
    ImageInputError,
    ModelCallError,
    ResponseParseError,
    WhatBirdError,
    default_config_file,
    load_config,
    save_config,
)
from ..core.config import ArkConfig

#: 上传大小上限：20 MB（远超模型需要，纯粹防误操作）
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

STATIC_DIR = Path(__file__).parent / "static"

CLIENT_HEADER = "X-WhatBird-Client"


class UTF8JSONResponse(JSONResponse):
    """显式声明 ``charset=utf-8`` 的 JSON 响应。

    JSON 标准默认就是 UTF-8，浏览器与 httpx 都能正确处理；但 Windows PowerShell 5.1
    等老客户端在响应头没有 charset 时按本地代码页（GBK）解码，中文会变成乱码。
    本地工具的用户很可能直接用 PowerShell 调接口，所以这里补上。
    """

    media_type = "application/json; charset=utf-8"


# --------------------------------------------------------------------- 出入参
class ConfigUpdate(BaseModel):
    """配置更新请求体；字段为 ``None`` 表示不修改。"""

    api_key: Optional[str] = Field(default=None, description="方舟 API Key；留空或回传脱敏串则不修改")
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: Optional[float] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_image_side: Optional[int] = None
    jpeg_quality: Optional[int] = None


class PathRecognizeRequest(BaseModel):
    """按服务器本地路径识别。"""

    path: str = Field(description="服务器上可访问的 jpg / jpeg 文件路径")


def require_client_header(x_whatbird_client: Optional[str] = Header(default=None)) -> None:
    """CSRF 防护：要求调用方带上自定义请求头。"""
    if x_whatbird_client != "1":
        raise HTTPException(
            status_code=403,
            detail=f"缺少请求头 {CLIENT_HEADER}: 1（跨站请求会被浏览器拦截）",
        )


# --------------------------------------------------------------------- 应用
def create_app() -> FastAPI:
    app = FastAPI(
        title="WhatBird API",
        version=__version__,
        description="基于火山引擎方舟 Doubao-Seed-2.1 的鸟类识别服务（core 之上的 REST 层）",
        default_response_class=UTF8JSONResponse,
    )

    # ----------------------------------------------------------- 错误翻译
    @app.exception_handler(WhatBirdError)
    async def _whatbird_error_handler(_: Request, exc: WhatBirdError) -> JSONResponse:
        if isinstance(exc, ConfigError):
            status = 400
        elif isinstance(exc, ImageInputError):
            status = 400
        elif isinstance(exc, (ModelCallError, ResponseParseError)):
            status = 502
        else:
            status = 400
        return UTF8JSONResponse(
            status_code=status,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": exc.message,
                    "hint": exc.hint,
                }
            },
        )

    # ----------------------------------------------------------- 基础
    @app.get("/api/health", tags=["system"])
    def health() -> dict[str, Any]:
        config = load_config()
        return {
            "status": "ok",
            "version": __version__,
            "configured": config.configured,
        }

    # ----------------------------------------------------------- 配置
    @app.get("/api/config", tags=["config"])
    def get_config() -> dict[str, Any]:
        config = load_config()
        return {
            "config": config.masked_dict(),
            "config_file": str(default_config_file()),
            "api_key_masked": True,
        }

    @app.put("/api/config", tags=["config"], dependencies=[Depends(require_client_header)])
    def put_config(payload: ConfigUpdate) -> dict[str, Any]:
        current = load_config()
        changes = payload.model_dump(exclude_none=True)

        # 前端会把脱敏后的 Key 回传，这种值直接忽略，避免把 "abcd****wxyz" 存进去
        api_key = changes.get("api_key")
        if api_key is not None and ("****" in api_key or not api_key.strip()):
            changes.pop("api_key")

        updated = current.replace(**changes)
        path = save_config(updated)
        return {
            "config": updated.masked_dict(),
            "config_file": str(path),
            "message": "配置已保存，后续调用立即生效",
        }

    @app.post("/api/config/verify", tags=["config"], dependencies=[Depends(require_client_header)])
    def verify_config() -> dict[str, Any]:
        recognizer = BirdRecognizer(load_config())
        reply = recognizer.verify_connection()
        return {"ok": True, "message": "连接正常", "model_reply": reply.strip()[:120]}

    # ----------------------------------------------------------- 识别
    @app.post("/api/recognize", tags=["recognize"], dependencies=[Depends(require_client_header)])
    def recognize_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        data = file.file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"图片过大（{len(data) / 1024 / 1024:.1f} MB），上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
            )
        recognizer = BirdRecognizer(load_config())
        result = recognizer.recognize_bytes(data, source=file.filename or "<upload>")
        return {"result": result.to_dict()}

    @app.post(
        "/api/recognize-path",
        tags=["recognize"],
        dependencies=[Depends(require_client_header)],
    )
    def recognize_path(payload: PathRecognizeRequest) -> dict[str, Any]:
        recognizer = BirdRecognizer(load_config())
        result = recognizer.recognize(payload.path)
        return {"result": result.to_dict()}

    # ----------------------------------------------------------- 静态前端
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
