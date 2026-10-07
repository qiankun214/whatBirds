"""批处理任务管理：启动、查询进度、终止。

为什么放在 Web 层而不是 core 层：
core 提供的是「依次识别多张图片」的能力（``iter_batch`` / ``aiter_batch``）；
而「把一次批量识别变成可轮询进度、可随时终止的后台任务」属于交互/传输层的事。
CLI 直接用 core 的同步生成器，不需要这一层。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core import BatchItem, BirdRecognizer, ImageSource, WhatBirdError

#: 进程内最多保留多少个任务（超出后丢弃最早结束的）
MAX_JOBS = 20


@dataclass
class BatchJob:
    """一次批量识别的状态快照。"""

    id: str
    total: int
    sources: list[ImageSource]
    status: str = "pending"  # pending / running / done / stopped / failed
    items: list[BatchItem] = field(default_factory=list)
    current_index: int = 0
    current_source: str = ""
    error_message: str = ""
    error_hint: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    task: "asyncio.Task[None] | None" = None

    @property
    def completed(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.ok)

    @property
    def failed(self) -> int:
        return self.completed - self.succeeded

    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "stopped": self.status == "stopped",
            "total": self.total,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "current_index": self.current_index,
            "current_source": self.current_source,
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "error": (
                {"message": self.error_message, "hint": self.error_hint}
                if self.error_message
                else None
            ),
            "items": [item.to_dict() for item in self.items],
        }


class BatchJobManager:
    """内存里的任务表。

    本地单用户工具，用进程内字典即可：任务随进程生命周期存在，
    只保留最近 ``max_jobs`` 个（优先丢弃已经结束的）。
    """

    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: dict[str, BatchJob] = {}
        self._max_jobs = max_jobs

    def get(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def start(self, recognizer: BirdRecognizer, sources: list[ImageSource]) -> BatchJob:
        """创建任务并在当前事件循环里启动（必须在 async 上下文调用）。"""
        job = BatchJob(id=uuid.uuid4().hex[:12], total=len(sources), sources=sources)
        self._jobs[job.id] = job
        self._trim()
        job.task = asyncio.create_task(self._run(job, recognizer))
        return job

    def stop(self, job: BatchJob) -> bool:
        """请求终止任务；返回是否真的发出了取消。"""
        if job.task is None or job.task.done():
            return False

        job.task.cancel()
        if job.status == "pending":  # 还没开始跑就被取消：不会有协程来改状态
            job.status = "stopped"
            job.finished_at = time.time()
        return True

    async def _run(self, job: BatchJob, recognizer: BirdRecognizer) -> None:
        job.status = "running"

        def on_start(index: int, total: int, name: str) -> None:
            job.current_index = index
            job.current_source = name

        try:
            async for item in recognizer.aiter_batch(job.sources, on_start=on_start):
                job.items.append(item)
            job.status = "done"
        except asyncio.CancelledError:
            # 用户点了「终止」：已完成的图片保留，未开始的不会发请求
            job.status = "stopped"
        except WhatBirdError as exc:
            job.status = "failed"
            job.error_message = exc.message
            job.error_hint = exc.hint
        except Exception as exc:  # noqa: BLE001 - 兜底，避免任务静默死掉
            job.status = "failed"
            job.error_message = f"批处理任务异常：{type(exc).__name__}"
            job.error_hint = str(exc)[:300]
        finally:
            job.finished_at = time.time()
            job.current_index = 0
            job.current_source = ""

    def _trim(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        finished = [
            job
            for job in self._jobs.values()
            if job.status in {"done", "stopped", "failed"}
        ]
        finished.sort(key=lambda job: job.started_at)
        for job in finished[: max(0, len(self._jobs) - self._max_jobs)]:
            self._jobs.pop(job.id, None)
