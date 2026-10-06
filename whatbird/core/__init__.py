"""core 层：鸟类识别的全部业务能力，不依赖 Web 框架。

上层（CLI / Web / 未来的其他 UI）只允许通过本包暴露的接口调用，
从而保证 Agent.md 第 4 条「UI -> core / CLI -> core」的扩展能力。
"""

from .config import ArkConfig, load_config, save_config, default_config_file
from .errors import (
    ConfigError,
    ImageInputError,
    ModelCallError,
    ResponseParseError,
    WhatBirdError,
)
from .models import BirdIdentification, SimilarSpecies, Taxonomy
from .recognizer import BirdRecognizer, identify

__all__ = [
    "ArkConfig",
    "BirdIdentification",
    "BirdRecognizer",
    "ConfigError",
    "ImageInputError",
    "ModelCallError",
    "ResponseParseError",
    "SimilarSpecies",
    "Taxonomy",
    "WhatBirdError",
    "default_config_file",
    "identify",
    "load_config",
    "save_config",
]
