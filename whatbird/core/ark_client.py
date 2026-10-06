"""火山方舟（Ark）客户端：通过 OpenAI 兼容端点调用 Doubao-Seed-2.1。

选用 openai SDK + 方舟兼容端点的原因（用户确认的选型）：
- 方舟提供 OpenAI 兼容的 ``/chat/completions``，图片以 ``image_url`` 传入；
- openai SDK 自带超时、重试与异常体系，省掉手写 HTTP 的样板代码；
- 未来若更换模型供应商，只要改 base_url / model 即可。

本模块是 core 层里唯一与外部服务通信的地方，方便替换与测试。
"""

from __future__ import annotations

from typing import Any

from .config import ArkConfig
from .errors import ConfigError, ModelCallError

#: 配置连通性自检时的提示词
PING_SYSTEM_PROMPT = "你是一个连通性自检助手，只需按要求回答。"
PING_USER_PROMPT = "请只回复两个字：正常"


class ArkVisionClient:
    """封装方舟的「图片 + 文本 -> 文本」调用。"""

    def __init__(self, config: ArkConfig) -> None:
        config.validate()
        self.config = config
        self._client = self._build_client()

    # ------------------------------------------------------------- 内部构造
    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
            raise ConfigError(
                "缺少依赖 openai，无法调用方舟接口",
                hint="请执行：python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openai",
            ) from exc

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=2,
        )

    # ------------------------------------------------------------- 对外能力
    def complete_with_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_url: str,
    ) -> str:
        """把一张图片（base64 data URL）连同提示词发给模型，返回文本回答。"""
        if not image_data_url.startswith("data:"):
            raise ModelCallError(
                "图片需要以 base64 data URL 传入",
                hint="请使用 core.image_input.prepare_image() 生成图片输入",
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self._chat(messages)

    def complete_text(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 32) -> str:
        """纯文本调用（用于配置连通性自检）。"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._chat(messages, max_tokens=max_tokens)

    def ping(self) -> str:
        """用一次极小的请求验证 api_key / base_url / model 是否可用。"""
        return self.complete_text(system_prompt=PING_SYSTEM_PROMPT, user_prompt=PING_USER_PROMPT)

    # ------------------------------------------------------------- 统一调用
    def _chat(self, messages: list[dict[str, Any]], *, max_tokens: int | None = None) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - 统一翻译成中文业务异常
            raise _translate_error(exc, model=self.config.model) from exc

        return _extract_text(response)


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
