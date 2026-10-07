"""core 层的门面（Facade）：上层只需要认识 :class:`BirdRecognizer`。

调用链：
    用户给的 jpg（路径或字节）
      -> ImageSource.prepare()         校验 + 压缩 + base64
      -> ark_client.ArkVisionClient    调用 Doubao-Seed-2.1 图片理解
      -> prompt.parse_json_object()    解析 JSON
      -> BirdIdentification            强类型结果

能力分层：
- 单张：``recognize`` / ``recognize_bytes``（同步）与 ``arecognize`` / ``arecognize_bytes``（异步）
- 批量：``iter_batch``（同步生成器，CLI 用）与 ``aiter_batch``（异步生成器，Web 任务用）

批处理是「一张失败不影响其余」：单张出错会记成一条 error 结果继续往下跑；
只有配置本身有问题（ConfigError）才直接抛出，因为那会让每一张都同样失败。
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Sequence

from .ark_client import ArkVisionClient
from .config import ArkConfig, load_config
from .errors import ConfigError, ModelCallError, WhatBirdError
from .image_input import (
    DEFAULT_MAX_SIDE,
    DEFAULT_QUALITY,
    ImageSource,
    PreparedImage,
    prepare_image,
    prepare_image_bytes,
)
from .models import BatchItem, BatchReport, BirdIdentification
from .prompt import USER_PROMPT, build_system_prompt, parse_json_object

#: 每张图片开始识别时的回调：(序号, 总数, 名称)
StartCallback = Callable[[int, int, str], None]


class BirdRecognizer:
    """鸟类识别器：一次构造，多次识别（单张或批量）。"""

    def __init__(
        self,
        config: ArkConfig | None = None,
        *,
        client: ArkVisionClient | None = None,
    ) -> None:
        self._config = config or load_config()
        self._client = client
        self._system_prompt = build_system_prompt()

    # ------------------------------------------------------------- 属性
    @property
    def config(self) -> ArkConfig:
        return self._config

    @property
    def client(self) -> ArkVisionClient:
        """惰性创建客户端：这样「只看配置」的场景不会依赖 openai。"""
        if self._client is None:
            self._client = ArkVisionClient(self._config)
        return self._client

    # ------------------------------------------------------------- 单张：同步
    def recognize(
        self,
        image_path: str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """识别本地 jpg 文件路径，返回鸟类分类学信息。"""
        return self.recognize_source(
            ImageSource.from_path(image_path),
            max_image_side=max_image_side,
            quality=quality,
        )

    def recognize_bytes(
        self,
        data: bytes,
        *,
        source: str = "<upload>",
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """识别内存中的图片字节（Web 上传用）。"""
        return self.recognize_source(
            ImageSource.from_bytes(source, data),
            max_image_side=max_image_side,
            quality=quality,
        )

    def recognize_source(
        self,
        source: ImageSource | str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """识别一个统一来源（路径或字节）。"""
        self._config.validate()  # 先校验配置，避免白读一遍图片
        prepared = ImageSource.coerce(source).prepare(
            max_image_side=max_image_side or self._config.max_image_side,
            quality=quality or self._config.jpeg_quality,
        )
        return self._run(prepared)

    # ------------------------------------------------------------- 单张：异步
    async def arecognize(
        self,
        image_path: str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """异步版单张识别（可被取消）。"""
        return await self.arecognize_source(
            ImageSource.from_path(image_path),
            max_image_side=max_image_side,
            quality=quality,
        )

    async def arecognize_bytes(
        self,
        data: bytes,
        *,
        source: str = "<upload>",
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """异步版内存字节识别（Web 上传用）。"""
        return await self.arecognize_source(
            ImageSource.from_bytes(source, data),
            max_image_side=max_image_side,
            quality=quality,
        )

    async def arecognize_source(
        self,
        source: ImageSource | str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """异步版统一来源识别。"""
        self._config.validate()
        prepared = ImageSource.coerce(source).prepare(
            max_image_side=max_image_side or self._config.max_image_side,
            quality=quality or self._config.jpeg_quality,
        )
        return await self._arun(prepared)

    # ------------------------------------------------------------- 批量
    def iter_batch(
        self,
        sources: Iterable[ImageSource | str],
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        stop_event: Any = None,
        on_start: StartCallback | None = None,
    ) -> Iterator[BatchItem]:
        """**同步**依次识别多张图片，每完成一张就 yield 一条结果。

        同步生成器天然支持「随时停止」：调用方不再取下一个值即可。
        另外可用 ``stop_event``（带 ``is_set()`` 的对象）让回调也能主动喊停。
        """
        items: Sequence[ImageSource] = [ImageSource.coerce(source) for source in sources]
        total = len(items)

        for index, source in enumerate(items, start=1):
            if stop_event is not None and stop_event.is_set():
                return
            if on_start is not None:
                on_start(index, total, source.name)
            yield self._batch_item_sync(
                source,
                index=index,
                total=total,
                max_image_side=max_image_side,
                quality=quality,
            )

    async def aiter_batch(
        self,
        sources: Iterable[ImageSource | str],
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        on_start: StartCallback | None = None,
    ) -> AsyncIterator[BatchItem]:
        """**异步**依次识别多张图片，每完成一张就 yield 一条结果。

        异步生成器跑在可取消的任务里：任务被 cancel 时，正在等待的上游请求
        会随 httpx 一起中断，未开始的图片根本不会发出请求。
        """
        items: Sequence[ImageSource] = [ImageSource.coerce(source) for source in sources]
        total = len(items)

        for index, source in enumerate(items, start=1):
            if on_start is not None:
                on_start(index, total, source.name)
            item = await self._batch_item_async(
                source,
                index=index,
                total=total,
                max_image_side=max_image_side,
                quality=quality,
            )
            yield item

    def run_batch(
        self,
        sources: Iterable[ImageSource | str],
        *,
        stop_event: Any = None,
        on_start: StartCallback | None = None,
    ) -> BatchReport:
        """把 :meth:`iter_batch` 跑完并汇总成 :class:`BatchReport`（CLI / 测试用）。"""
        materialised = [ImageSource.coerce(source) for source in sources]
        report = BatchReport(total=len(materialised))
        started = time.perf_counter()
        try:
            for item in self.iter_batch(materialised, stop_event=stop_event, on_start=on_start):
                report.items.append(item)
        finally:
            report.elapsed_seconds = time.perf_counter() - started
        report.stopped = report.completed < report.total
        return report

    # ------------------------------------------------------------- 连通性
    def verify_connection(self) -> str:
        """验证当前配置能否成功调用方舟（会消耗极少量 token）。"""
        self._config.validate()
        return self.client.ping()

    async def averify_connection(self) -> str:
        """异步版连通性验证。"""
        self._config.validate()
        return await self.client.aping()

    # ------------------------------------------------------------- 内部
    def _run(self, prepared: PreparedImage) -> BirdIdentification:
        started = time.perf_counter()
        text = self.client.complete_with_image(
            system_prompt=self._system_prompt,
            user_prompt=USER_PROMPT,
            image_data_url=prepared.data_url,
        )
        return self._build(prepared, text, time.perf_counter() - started)

    async def _arun(self, prepared: PreparedImage) -> BirdIdentification:
        started = time.perf_counter()
        text = await self.client.acomplete_with_image(
            system_prompt=self._system_prompt,
            user_prompt=USER_PROMPT,
            image_data_url=prepared.data_url,
        )
        return self._build(prepared, text, time.perf_counter() - started)

    def _build(self, prepared: PreparedImage, text: str, elapsed: float) -> BirdIdentification:
        if not text:
            raise ModelCallError("模型返回了空内容", hint="请重试")
        payload = parse_json_object(text)
        return BirdIdentification.from_model_output(
            payload,
            source_image=prepared.source,
            model=self._config.model,
            elapsed_seconds=elapsed,
            raw_response=text,
        )

    # -- 批量里单张的包装：把单张异常收进结果，配置错误仍然直接抛出 ----------
    def _batch_item_sync(
        self,
        source: ImageSource,
        *,
        index: int,
        total: int,
        max_image_side: int | None,
        quality: int | None,
    ) -> BatchItem:
        started = time.perf_counter()
        try:
            identification = self.recognize_source(
                source,
                max_image_side=max_image_side,
                quality=quality,
            )
        except ConfigError:
            raise
        except WhatBirdError as exc:
            return _error_item(source, index, total, exc, time.perf_counter() - started)
        return BatchItem(
            index=index,
            total=total,
            source=source.name,
            status="ok",
            identification=identification,
            elapsed_seconds=time.perf_counter() - started,
        )

    async def _batch_item_async(
        self,
        source: ImageSource,
        *,
        index: int,
        total: int,
        max_image_side: int | None,
        quality: int | None,
    ) -> BatchItem:
        started = time.perf_counter()
        try:
            identification = await self.arecognize_source(
                source,
                max_image_side=max_image_side,
                quality=quality,
            )
        except ConfigError:
            raise
        except WhatBirdError as exc:
            return _error_item(source, index, total, exc, time.perf_counter() - started)
        return BatchItem(
            index=index,
            total=total,
            source=source.name,
            status="ok",
            identification=identification,
            elapsed_seconds=time.perf_counter() - started,
        )


def _error_item(
    source: ImageSource,
    index: int,
    total: int,
    exc: WhatBirdError,
    elapsed: float,
) -> BatchItem:
    return BatchItem(
        index=index,
        total=total,
        source=source.name,
        status="error",
        error_message=exc.message,
        error_hint=exc.hint,
        elapsed_seconds=elapsed,
    )


def identify(image_path: str, config: ArkConfig | None = None) -> BirdIdentification:
    """一次性识别的便捷函数：``identify(r"D:\\pics\\bird.jpg")``。"""
    return BirdRecognizer(config).recognize(image_path)
