"""core 层的统一异常体系。

所有对外抛出的异常都带一段中文 ``hint``，CLI 与 Web 层可以直接展示给用户，
不需要各自重复解释错误含义。
"""

from __future__ import annotations


class WhatBirdError(Exception):
    """所有 whatbird 业务异常的基类。"""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"{self.message}（{self.hint}）" if self.hint else self.message


class ConfigError(WhatBirdError):
    """API 配置缺失、格式错误或无法读写。"""


class ImageInputError(WhatBirdError):
    """输入的图片路径不存在、不是 jpg，或文件已损坏。"""


class ModelCallError(WhatBirdError):
    """调用火山方舟接口失败（网络、鉴权、限流、模型不存在等）。"""


class ResponseParseError(WhatBirdError):
    """模型返回的内容不是可解析的 JSON 结构。"""


class FocusDependencyError(WhatBirdError):
    """对焦分析所需的第三方库未安装（opencv-python-headless / numpy / exifread）。

    独立成一个异常类型，是为了让 CLI 与 Web 都能给出「按清华镜像装哪个包」的中文提示，
    而不是抛裸 ``ImportError`` 让用户看栈。
    """


class DetectorDependencyError(WhatBirdError):
    """鸟类位置检测所需的第三方库或权重不可用（ultralytics / torch(+xpu) / 模型文件）。

    与 :class:`FocusDependencyError` 同样的理由独立成一个类型：检测能力是**可选**的
    （见 requirements-detect.txt），没装时 ``import whatbird.core`` 与其余功能必须照常可用，
    只有真正调用检测时才用这个异常给出中文安装提示。
    """
