"""FastAPI 后端：``REST API -> core``，并托管静态前端（前后端分离）。

接口一览（详见 ``/docs`` 自动生成的 OpenAPI 文档）：

===========================  ======  =================================================
方法                          路径     说明
===========================  ======  =================================================
GET                          /api/health          健康检查
GET                          /api/config          读取配置（API Key 已脱敏）+ 默认值 + 文件位置
PUT                          /api/config          写入配置（持久化到用户目录）
POST                         /api/config/verify   验证配置能否调通方舟（可带「草稿」参数先测后存）
POST                         /api/recognize       上传单张图片识别（multipart）
POST                         /api/recognize-path  传入服务器上的 jpg 路径识别
POST                         /api/batch/start-upload  批量：上传多张图片，返回任务号
POST                         /api/batch/start-paths   批量：传入多个服务器路径，返回任务号
GET                          /api/batch/{job_id}      批量：查询进度与已完成的结果
POST                         /api/batch/{job_id}/stop 批量：终止任务
GET                          /                    静态前端页面（含配置向导）
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
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..core import (
    BirdRecognizer,
    ConfigError,
    ImageInputError,
    ImageSource,
    ModelCallError,
    ResponseParseError,
    WhatBirdError,
    config_file_status,
    load_config,
    save_config,
)
from ..core.config import ArkConfig, mask_secret
from .jobs import BatchJobManager

#: 上传大小上限：20 MB（远超模型需要，纯粹防误操作）
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: 一次批处理最多多少张（防止一次点错把额度跑光）
MAX_BATCH_IMAGES = 50

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
    allow_project_fallback: Optional[bool] = Field(
        default=None,
        description="是否同意把配置保存到项目内兜底位置（默认不同意，不会自动降级）",
    )


class PathRecognizeRequest(BaseModel):
    """按服务器本地路径识别。"""

    path: str = Field(description="服务器上可访问的 jpg / jpeg 文件路径")


class BatchPathsRequest(BaseModel):
    """批量：按服务器本地路径列表依次识别。"""

    paths: List[str] = Field(description="服务器上可访问的 jpg / jpeg 路径列表，按顺序识别")


def require_client_header(x_whatbird_client: Optional[str] = Header(default=None)) -> None:
    """CSRF 防护：要求调用方带上自定义请求头。"""
    if x_whatbird_client != "1":
        raise HTTPException(
            status_code=403,
            detail=f"缺少请求头 {CLIENT_HEADER}: 1（跨站请求会被浏览器拦截）",
        )


def merge_config_update(current: ArkConfig, payload: Optional["ConfigUpdate"]) -> ArkConfig:
    """把请求体里的改动合并到当前配置上。

    - 字段为 ``None`` 表示不修改；
    - ``api_key`` 为空串或脱敏串（含 ``****``）时视为「不修改」，
      这样配置向导里「留空表示沿用已保存的 Key」才能安全工作。
    """
    if payload is None:
        return current

    changes = payload.model_dump(exclude_none=True)
    changes.pop("allow_project_fallback", None)  # 这是控制项，不是配置项
    api_key = changes.get("api_key")
    if api_key is not None and ("****" in api_key or not api_key.strip()):
        changes.pop("api_key")

    return current.replace(**changes) if changes else current


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
        """给配置向导用的完整状态：当前值（脱敏）、默认值、配置文件位置与可写性。"""
        config = load_config()
        status = config_file_status()
        return {
            "config": config.masked_dict(),
            "config_file": status["path"],
            "config_file_exists": status["exists"],
            "config_loaded_from": status["loaded_from"],
            "config_writable": status["writable"],
            "config_file_error": status["error"],
            "config_location": {
                "kind": status["kind"],
                "label": status["label"],
                "is_fallback": status["is_fallback"],
                "needs_consent": status["needs_consent"],
                "project_candidate": status["project_candidate"],
                "warning": status["warning"],
                "preferred_path": status["preferred_path"],
                "preferred_error": status["preferred_error"],
                "interpreter": status["interpreter"],
                "candidates": status["candidates"],
            },
            "defaults": ArkConfig().to_dict(),
            "api_key_masked": True,
        }

    @app.put("/api/config", tags=["config"], dependencies=[Depends(require_client_header)])
    def put_config(payload: ConfigUpdate) -> dict[str, Any]:
        consent = payload.allow_project_fallback
        updated = merge_config_update(load_config(), payload)
        path = save_config(updated, allow_project=consent)
        status = config_file_status(allow_project=consent)
        return {
            "config": updated.masked_dict(),
            "config_file": str(path),
            "config_writable": status["writable"],
            "config_location": {
                "kind": status["kind"],
                "label": status["label"],
                "is_fallback": status["is_fallback"],
                "needs_consent": status["needs_consent"],
                "project_candidate": status["project_candidate"],
                "warning": status["warning"],
            },
            "message": "配置已保存，后续调用立即生效",
        }

    @app.post("/api/config/verify", tags=["config"], dependencies=[Depends(require_client_header)])
    def verify_config(payload: Optional[ConfigUpdate] = None) -> dict[str, Any]:
        """测试连接。

        可以带上一份「草稿」参数：此时只测试草稿与已保存配置的合并结果，
        **不会写盘**。配置向导因此支持「先测通、再保存」。
        """
        candidate = merge_config_update(load_config(), payload)
        recognizer = BirdRecognizer(candidate)
        reply = recognizer.verify_connection()
        return {
            "ok": True,
            "message": "连接正常",
            "model_reply": reply.strip()[:120],
            "saved": False,
            "tested": {
                "base_url": candidate.base_url,
                "model": candidate.model,
                "api_key": mask_secret(candidate.api_key),
            },
        }

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

    # ----------------------------------------------------------- 批量识别
    jobs = BatchJobManager()

    @app.post(
        "/api/batch/start-upload",
        tags=["batch"],
        dependencies=[Depends(require_client_header)],
    )
    async def start_batch_upload(files: List[UploadFile] = File(...)) -> dict[str, Any]:
        """批量：上传多张图片，立即返回任务号，识别在后台依次进行。"""
        if not files:
            raise HTTPException(status_code=400, detail="请至少选择一张 jpg 图片")
        if len(files) > MAX_BATCH_IMAGES:
            raise HTTPException(status_code=400, detail=f"一次最多 {MAX_BATCH_IMAGES} 张图片")

        sources: list[ImageSource] = []
        for upload in files:
            data = await upload.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"图片过大（{len(data) / 1024 / 1024:.1f} MB）：{upload.filename}",
                )
            sources.append(ImageSource.from_bytes(upload.filename or "<upload>", data))

        job = jobs.start(BirdRecognizer(load_config()), sources)
        return {"job_id": job.id, "total": job.total, "status": job.status}

    @app.post(
        "/api/batch/start-paths",
        tags=["batch"],
        dependencies=[Depends(require_client_header)],
    )
    async def start_batch_paths(payload: BatchPathsRequest) -> dict[str, Any]:
        """批量：传入服务器上的多个路径，立即返回任务号。"""
        paths = [item.strip() for item in payload.paths if item and item.strip()]
        if not paths:
            raise HTTPException(status_code=400, detail="请至少提供一个 jpg 文件路径")
        if len(paths) > MAX_BATCH_IMAGES:
            raise HTTPException(status_code=400, detail=f"一次最多 {MAX_BATCH_IMAGES} 张图片")

        sources = [ImageSource.from_path(path) for path in paths]
        job = jobs.start(BirdRecognizer(load_config()), sources)
        return {"job_id": job.id, "total": job.total, "status": job.status}

    @app.get("/api/batch/{job_id}", tags=["batch"])
    async def batch_status(job_id: str) -> dict[str, Any]:
        """批量：查询进度与已完成的结果（前端轮询这个接口）。"""
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        return job.to_dict()

    @app.post(
        "/api/batch/{job_id}/stop",
        tags=["batch"],
        dependencies=[Depends(require_client_header)],
    )
    async def batch_stop(job_id: str) -> dict[str, Any]:
        """批量：终止任务。正在等待的上游请求会被中断，未开始的图片不会发出。"""
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")

        stopped = jobs.stop(job)
        return {
            "job_id": job.id,
            "stopped": stopped,
            "status": job.status,
            "message": "已发送终止请求" if stopped else "该任务已经结束",
        }

    # ----------------------------------------------------------- 静态前端
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
