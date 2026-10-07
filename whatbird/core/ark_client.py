"""火山方舟（Ark）客户端：通过 OpenAI 兼容端点调用 Doubao-Seed-2.1。

选用 openai SDK + 方舟兼容端点的原因（用户确认的选型）：
- 方舟提供 OpenAI 兼容的 ``/chat/completions``，图片以 ``image_url`` 传入；
- openai SDK 自带超时、重试与异常体系，省掉手写 HTTP 的样板代码；
- 未来若更换模型供应商，只要改 base_url / model 即可。

同时提供**同步**与**异步**两套调用：
- 同步（``complete_with_image``）给 CLI 与单图接口用，简单直接；
- 异步（``acomplete_with_image``）给批处理任务用，可以被 ``asyncio`` 取消 ——
  这是「终止识别」能真正中断上游请求的关键：任务被取消时 httpx 会关掉连接。

本模块是 core 层里唯一与外部服务通信的地方，方便替换与测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ArkConfig
from .errors import ConfigError, ModelCallError

#: 配置连通性自检时的提示词
PING_SYSTEM_PROMPT = "你是一个连通性自检助手，只需按要求回答。"
PING_USER_PROMPT = "请只回复两个字：正常"


@dataclass(slots=True)
class ModelReply:
    """模型的一次回答 + token 用量（用量用于判断耗时花在哪）。"""

    text: str
    usage: dict[str, int] | None = None


def normalize_reply(reply: Any) -> tuple[str, dict[str, int] | None]:
    """把客户端返回值统一成 ``(文本, 用量)``。

    允许返回纯字符串（测试替身、旧实现都这么做），此时用量为 None。
    """
    if isinstance(reply, ModelReply):
        return reply.text, reply.usage
    if isinstance(reply, str):
        return reply, None
    text = getattr(reply, "text", None)
    if text is None:
        raise ModelCallError("模型客户端返回了无法识别的结果", hint=f"实际类型：{type(reply).__name__}")
    return str(text), getattr(reply, "usage", None)


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    payload: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        if isinstance(value, int):
            payload[key] = value
    return payload or None


# --------------------------------------------------------------------- 消息构造
def build_image_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
) -> list[dict[str, Any]]:
    """构造「一张图片 + 提示词」的对话消息（同步/异步共用）。"""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def build_text_messages(*, system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    """构造纯文本对话消息（用于连通性自检）。"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class ArkVisionClient:
    """封装方舟的「图片 + 文本 -> 文本」调用（同步 + 异步）。"""

    def __init__(self, config: ArkConfig) -> None:
        config.validate()
        self.config = config
        self._sync_client: Any = None
        self._async_client: Any = None

    # ------------------------------------------------------------- 内部构造
    @staticmethod
    def _import_openai() -> Any:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
            raise ConfigError(
                "缺少依赖 openai，无法调用方舟接口",
                hint="请执行：python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai",
            ) from exc
        return openai

    @property
    def sync_client(self) -> Any:
        if self._sync_client is None:
            openai = self._import_openai()
            self._sync_client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=2,
            )
        return self._sync_client

    @property
    def async_client(self) -> Any:
        if self._async_client is None:
            openai = self._import_openai()
            self._async_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=2,
            )
        return self._async_client

    # ------------------------------------------------------------- 对外能力
    def complete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int | None = None,
    ) -> ModelReply:
        """同步：把一张图片（base64 data URL）连同提示词发给模型。"""
        _ensure_data_url(image_data_url)
        messages = build_image_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data_url=image_data_url,
        )
        return self._chat(messages, max_tokens=max_tokens)

    async def acomplete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
        max_tokens: int | None = None,
    ) -> ModelReply:
        """异步：同上，但可以被取消（批处理任务靠它实现「终止」）。"""
        _ensure_data_url(image_data_url)
        messages = build_image_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data_url=image_data_url,
        )
        return await self._achat(messages, max_tokens=max_tokens)

    def complete_text(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 32) -> str:
        """同步纯文本调用（用于配置连通性自检）。"""
        reply = self._chat(
            build_text_messages(system_prompt=system_prompt, user_prompt=user_prompt),
            max_tokens=max_tokens,
        )
        return reply.text

    async def acomplete_text(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 32) -> str:
        """异步纯文本调用。"""
        reply = await self._achat(
            build_text_messages(system_prompt=system_prompt, user_prompt=user_prompt),
            max_tokens=max_tokens,
        )
        return reply.text

    def ping(self) -> str:
        """用一次极小的请求验证 api_key / base_url / model 是否可用。"""
        return self.complete_text(system_prompt=PING_SYSTEM_PROMPT, user_prompt=PING_USER_PROMPT)

    async def aping(self) -> str:
        """异步版连通性自检。"""
        return await self.acomplete_text(system_prompt=PING_SYSTEM_PROMPT, user_prompt=PING_USER_PROMPT)

    # ------------------------------------------------------------- 统一调用
    def _chat(self, messages: list[dict[str, Any]], *, max_tokens: int | None = None) -> ModelReply:
        try:
            response = self.sync_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - 统一翻译成中文业务异常
            raise _translate_error(exc, model=self.config.model) from exc

        return ModelReply(text=_extract_text(response), usage=_extract_usage(response))

    async def _achat(self, messages: list[dict[str, Any]], *, max_tokens: int | None = None) -> ModelReply:
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            # 注意：asyncio.CancelledError 不是 Exception 的子类，会自然向外传播 —— 这正是取消要的
            raise _translate_error(exc, model=self.config.model) from exc

        return ModelReply(text=_extract_text(response), usage=_extract_usage(response))


def _ensure_data_url(image_data_url: str) -> None:
    if not image_data_url.startswith("data:"):
        raise ModelCallError(
            "图片需要以 base64 data URL 传入",
            hint="请使用 core.image_input.prepare_image() 生成图片输入",
        )


def _extract_text(response: Any) -> str:
    try:
        choices = response.choices
    except AttributeError as exc:  # pragma: no cover - 非预期返回结构
        raise ModelCallError("模型返回结构异常，缺少 choices 字段", hint=str(exc)) from exc

    if not choices:
        raise ModelCallError("模型返回了空的候选结果", hint="可稍后重试")

    message = choices[0].message
    content = getattr(message, "content", None)

    # 少数模型会把内容放在 content 数组里
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        content = "\n".join(parts)

    if not content:
        raise ModelCallError(
            "模型返回了空内容",
            hint="可能触发了内容安全策略，或该模型不支持图片输入",
        )
    return str(content).strip()


def _translate_error(exc: Exception, *, model: str) -> ModelCallError:
    """把 openai SDK 的异常翻译成带中文提示的 :class:`ModelCallError`。"""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    detail = _short_detail(exc)

    if name == "APITimeoutError":
        return ModelCallError("调用方舟接口超时", hint=f"可调大 timeout（当前模型 {model}）；{detail}")
    if name == "APIConnectionError":
        return ModelCallError(
            "无法连接方舟服务",
            hint="请检查网络、代理或 base_url 是否正确；海外/内网环境可能需要代理",
        )
    if status == 401 or name == "AuthenticationError":
        return ModelCallError("鉴权失败（HTTP 401）", hint="API Key 不正确或已失效，请重新配置")
    if status == 403 or name == "PermissionDeniedError":
        return ModelCallError("无权访问该模型（HTTP 403）", hint="请确认该 API Key 已开通对应模型/接入点")
    if status == 404 or name == "NotFoundError":
        return ModelCallError(
            f"模型或接入点不存在（HTTP 404）：{model}",
            hint="请填写控制台里的 Model ID（如 doubao-seed-2-1-lite）或推理接入点 ID（ep-xxxxxxxx）",
        )
    if status == 429 or name == "RateLimitError":
        return ModelCallError("触发限流（HTTP 429）", hint="请稍后重试，或降低并发/频率")
    if status == 400 or name == "BadRequestError":
        return ModelCallError(
            "请求被方舟拒绝（HTTP 400）",
            hint=f"常见原因：模型不支持图片输入、图片过大、参数越界。{detail}",
        )
    if status is not None:
        return ModelCallError(f"调用方舟接口失败（HTTP {status}）", hint=detail)
    return ModelCallError(f"调用方舟接口失败：{name}", hint=detail)


def _short_detail(exc: Exception, *, limit: int = 300) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:limit]
