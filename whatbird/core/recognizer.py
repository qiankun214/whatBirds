"""core 层的门面（Facade）：上层只需要认识 :class:`BirdRecognizer`。

调用链：
    用户给的 jpg 路径
      -> image_input.prepare_image()   校验 + 压缩 + base64
      -> ark_client.ArkVisionClient    调用 Doubao-Seed-2.1 图片理解
      -> prompt.parse_json_object()    解析 JSON
      -> BirdIdentification            强类型结果
"""

from __future__ import annotations

import time

from .ark_client import ArkVisionClient
from .config import ArkConfig, load_config
from .errors import ModelCallError
from .image_input import (
    DEFAULT_MAX_SIDE,
    DEFAULT_QUALITY,
    PreparedImage,
    prepare_image,
    prepare_image_bytes,
)
from .models import BirdIdentification
from .prompt import USER_PROMPT, build_system_prompt, parse_json_object


class BirdRecognizer:
    """鸟类识别器：一次构造，多次识别。"""

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

    # ------------------------------------------------------------- 主能力
    def recognize(
        self,
        image_path: str,
        *,
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """识别本地 jpg 文件路径，返回鸟类分类学信息。"""
        self._config.validate()  # 先校验配置，避免白读一遍图片
        prepared = prepare_image(
            image_path,
            max_image_side=max_image_side or self._config.max_image_side,
            quality=quality or self._config.jpeg_quality,
        )
        return self._run(prepared)

    def recognize_bytes(
        self,
        data: bytes,
        *,
        source: str = "<upload>",
        max_image_side: int | None = None,
        quality: int | None = None,
    ) -> BirdIdentification:
        """识别内存中的图片字节（Web 上传用）。"""
        self._config.validate()
        prepared = prepare_image_bytes(
            data,
            source=source,
            max_image_side=max_image_side or self._config.max_image_side,
            quality=quality or self._config.jpeg_quality,
        )
        return self._run(prepared)

    def verify_connection(self) -> str:
        """验证当前配置能否成功调用方舟（会消耗极少量 token）。"""
        self._config.validate()
        return self.client.ping()

    # ------------------------------------------------------------- 内部
    def _run(self, prepared: PreparedImage) -> BirdIdentification:
        self._config.validate()

        started = time.perf_counter()
        text = self.client.complete_with_image(
            system_prompt=self._system_prompt,
            user_prompt=USER_PROMPT,
            image_data_url=prepared.data_url,
        )
        elapsed = time.perf_counter() - started

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


def identify(image_path: str, config: ArkConfig | None = None) -> BirdIdentification:
    """一次性识别的便捷函数：``identify(r"D:\\pics\\bird.jpg")``。"""
    return BirdRecognizer(config).recognize(image_path)
