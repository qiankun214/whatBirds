"""core 层：鸟类识别的全部业务能力，不依赖 Web 框架。

上层（CLI / Web / 未来的其他 UI）只允许通过本包暴露的接口调用，
从而保证 Agent.md 第 4 条「UI -> core / CLI -> core」的扩展能力。
"""

from .config import (
    ArkConfig,
    config_file_status,
    config_locations,
    default_config_file,
    existing_config_file,
    git_ignores,
    interpreter_status,
    load_config,
    migrate_config,
    project_consent_granted,
    resolve_config_location,
    save_config,
)
from .detect import (
    Detection,
    DetectionReport,
    DetectionStyle,
    DetectParams,
    YoloDetector,
    detect_birds,
    render_bird_overlay,
)
from .errors import (
    ConfigError,
    DetectorDependencyError,
    FocusDependencyError,
    ImageInputError,
    ModelCallError,
    ResponseParseError,
    WhatBirdError,
)
from .focus import (
    FocusHit,
    FocusParams,
    FocusReport,
    OverlayStyle,
    analyze_focus,
    overlay_data_url,
    render_focus_overlay,
)
from .image_input import ImageSource
from .models import BatchItem, BatchReport, BirdIdentification, SimilarSpecies, Taxonomy
from .recognizer import BirdRecognizer, identify

__all__ = [
    "ArkConfig",
    "BatchItem",
    "BatchReport",
    "BirdIdentification",
    "BirdRecognizer",
    "ConfigError",
    "Detection",
    "DetectionReport",
    "DetectionStyle",
    "DetectParams",
    "DetectorDependencyError",
    "FocusDependencyError",
    "FocusHit",
    "FocusParams",
    "FocusReport",
    "ImageInputError",
    "ImageSource",
    "ModelCallError",
    "OverlayStyle",
    "ResponseParseError",
    "SimilarSpecies",
    "Taxonomy",
    "WhatBirdError",
    "YoloDetector",
    "analyze_focus",
    "config_file_status",
    "config_locations",
    "default_config_file",
    "detect_birds",
    "existing_config_file",
    "git_ignores",
    "identify",
    "interpreter_status",
    "load_config",
    "migrate_config",
    "overlay_data_url",
    "project_consent_granted",
    "render_bird_overlay",
    "render_focus_overlay",
    "resolve_config_location",
    "save_config",
]
