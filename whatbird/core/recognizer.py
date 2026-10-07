"""core 层的门面（Facade）：上层只需要认识 :class:`BirdRecognizer`。

调用链：
    用户给的 jpg（路径或字节）
      -> ImageSource.prepare()         校验 + 压缩 + base64
      -> ark_client.ArkVisionClient    调用 Doubao-Seed-2.1 图片理解
      -> prompt.parse_json_object()    解析 JSON
      -> BirdIdentification            强类型结果

能力分层：
- 单张：``recognize`` / ``recognize_bytes``（同步）与 ``arecognize`` / ``arecognize_bytes``（异步）
- 批量：``iter_batch``（同步生成器，CLI 用）与 ``aiter_batch``（异步生成器，可并发、可取消）
- 模式：``full``（完整信息）/ ``fast``（只要名称+分类学，输出短、更快）

批处理是「一张失败不影响其余」：单张出错会记成一条 error 结果继续往下跑；
只有配置本身有问题（ConfigError）才直接抛出，因为那会让每一张都同样失败。

关于速度：视觉模型的输出是逐 token 生成的，所以**输出长度几乎线性决定单张耗时**；
批量则受「串行」影响。因此这里提供两个关键杠杆：``mode='fast'``（缩短输出）
与并发（``aiter_batch(concurrency=N)``）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Sequence

from .ark_client import ArkVisionClient, normalize_reply
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
from .prompt import (
    MODE_FAST,
    MODE_FULL,
    PromptSet,
    build_prompt_set,
    normalize_mode,
    parse_json_object,
)

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
        self._prompt_cache: dict[str, PromptSet] = {}

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

    @property
    def default_mode(self) -> str:
        """没显式指定时用哪个模式（由配置里的 fast_mode 决定）。"""
        return MODE_FAST if self._config.fast_mode else MODE_FULL

    def prompts(self, mode: str | None = None) -> PromptSet:
        """取（并缓存）某个模式对应的提示词与输出上限。"""
        resolved = normalize_mode(mode if mode is not None else self.default_mode)
        if resolved not in self._prompt_cache:
            self._prompt_cache[resolved] = build_prompt_set(
                resolved,
                config_max_tokens=self._config.max_tokens,
            )
        return self._prompt_cache[resolved]

    # ------------------------------------------------------------- 单张：同步
    def recognize(
        self,
        image_path: str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
    ) -> BirdIdentification:
        """识别本地 jpg 文件路径，返回鸟类分类学信息。"""
        return self.recognize_source(
            ImageSource.from_path(image_path),
            max_image_side=max_image_side,
            quality=quality,
            mode=mode,
        )

    def recognize_bytes(
        self,
        data: bytes,
        *,
        source: str = "<upload>",
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
    ) -> BirdIdentification:
        """识别内存中的图片字节（Web 上传用）。"""
        return self.recognize_source(
            ImageSource.from_bytes(source, data),
            max_image_side=max_image_side,
            quality=quality,
            mode=mode,
        )

    def recognize_source(
        self,
        source: ImageSource | str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
    ) -> BirdIdentification:
        """识别一个统一来源（路径或字节）。"""
        self._config.validate()  # 先校验配置，避免白读一遍图片
        prepared = ImageSource.coerce(source).prepare(
            max_image_side=max_image_side or self._config.max_image_side,
            quality=quality or self._config.jpeg_quality,
        )
        return self._run(prepared, mode=mode)

    # ------------------------------------------------------------- 单张：异步
    async def arecognize(
        self,
        image_path: str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
    ) -> BirdIdentification:
        """异步版单张识别（可被取消）。"""
        return await self.arecognize_source(
            ImageSource.from_path(image_path),
            max_image_side=max_image_side,
            quality=quality,
            mode=mode,
        )

    async def arecognize_bytes(
        self,
        data: bytes,
        *,
        source: str = "<upload>",
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
    ) -> BirdIdentification:
        """异步版内存字节识别（Web 上传用）。"""
        return await self.arecognize_source(
            ImageSource.from_bytes(source, data),
            max_image_side=max_image_side,
            quality=quality,
            mode=mode,
        )

    async def arecognize_source(
        self,
        source: ImageSource | str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
    ) -> BirdIdentification:
        """异步版统一来源识别。"""
        self._config.validate()
        prepared = ImageSource.coerce(source).prepare(
            max_image_side=max_image_side or self._config.max_image_side,
            quality=quality or self._config.jpeg_quality,
        )
        return await self._arun(prepared, mode=mode)

    # ------------------------------------------------------------- 批量
    def iter_batch(
        self,
        sources: Iterable[ImageSource | str],
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
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
                mode=mode,
            )

    async def aiter_batch(
        self,
        sources: Iterable[ImageSource | str],
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
        mode: str | None = None,
        concurrency: int | None = None,
        on_start: StartCallback | None = None,
    ) -> AsyncIterator[BatchItem]:
        """**异步**批量识别，每完成一张就 yield 一条结果。

        - ``concurrency=1``：严格串行；
        - ``concurrency>1``：最多同时进行这么多张，**完成即产出**（结果顺序不保证，
          调用方按 ``item.index`` 排列即可）。批量提速主要靠它。

        任务被 cancel 时，所有在跑的请求会一起中断，未开始的图片不会发出请求。
        """
        items: Sequence[ImageSource] = [ImageSource.coerce(source) for source in sources]
        total = len(items)
        limit = max(1, int(concurrency if concurrency is not None else self._config.concurrency))

        async def one(index: int, source: ImageSource) -> BatchItem:
            if on_start is not None:
                on_start(index, total, source.name)
            return await self._batch_item_async(
                source,
                index=index,
                total=total,
                max_image_side=max_image_side,
                quality=quality,
                mode=mode,
            )

        if limit == 1 or total <= 1:
            for index, source in enumerate(items, start=1):
                yield await one(index, source)
            return

        semaphore = asyncio.Semaphore(limit)

        async def limited(index: int, source: ImageSource) -> BatchItem:
            async with semaphore:
                return await one(index, source)

        tasks = [asyncio.create_task(limited(i, s)) for i, s in enumerate(items, start=1)]
        try:
            for future in asyncio.as_completed(tasks):
                yield await future
        finally:
            # 被终止或提前退出：把还在跑的任务全部取消（httpx 会随之关掉上游连接）
            for task in tasks:
                if not task.done():
                    task.cancel()

    def run_batch(
        self,
        sources: Iterable[ImageSource | str],
        *,
        mode: str | None = None,
        stop_event: Any = None,
        on_start: StartCallback | None = None,
    ) -> BatchReport:
        """把 :meth:`iter_batch` 跑完并汇总成 :class:`BatchReport`（CLI / 测试用）。"""
        materialised = [ImageSource.coerce(source) for source in sources]
        report = BatchReport(total=len(materialised), mode=self.prompts(mode).mode)
        started = time.perf_counter()
        try:
            for item in self.iter_batch(
                materialised,
                mode=mode,
                stop_event=stop_event,
                on_start=on_start,
            ):
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
    def _run(self, prepared: PreparedImage, *, mode: str | None = None) -> BirdIdentification:
        prompts = self.prompts(mode)
        started = time.perf_counter()
        reply = self.client.complete_with_image(
            system_prompt=prompts.system,
            user_prompt=prompts.user,
            image_data_url=prepared.data_url,
            max_tokens=prompts.max_tokens,
        )
        text, usage = normalize_reply(reply)
        return self._build(
            prepared,
            text,
            time.perf_counter() - started,
            mode=prompts.mode,
            usage=usage,
        )

    async def _arun(self, prepared: PreparedImage, *, mode: str | None = None) -> BirdIdentification:
        prompts = self.prompts(mode)
        started = time.perf_counter()
        reply = await self.client.acomplete_with_image(
            system_prompt=prompts.system,
            user_prompt=prompts.user,
            image_data_url=prepared.data_url,
            max_tokens=prompts.max_tokens,
        )
        text, usage = normalize_reply(reply)
        return self._build(
            prepared,
            text,
            time.perf_counter() - started,
            mode=prompts.mode,
            usage=usage,
        )

    def _build(
        self,
        prepared: PreparedImage,
        text: str,
        elapsed: float,
        *,
        mode: str = MODE_FULL,
        usage: dict[str, int] | None = None,
    ) -> BirdIdentification:
        if not text:
            raise ModelCallError("模型返回了空内容", hint="请重试")
        payload = parse_json_object(text)
        return BirdIdentification.from_model_output(
            payload,
            source_image=prepared.source,
            model=self._config.model,
            mode=mode,
            elapsed_seconds=elapsed,
            usage=usage,
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
        mode: str | None,
    ) -> BatchItem:
        started = time.perf_counter()
        try:
            identification = self.recognize_source(
                source,
                max_image_side=max_image_side,
                quality=quality,
                mode=mode,
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
        mode: str | None,
    ) -> BatchItem:
        started = time.perf_counter()
        try:
            identification = await self.arecognize_source(
                source,
                max_image_side=max_image_side,
                quality=quality,
                mode=mode,
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
