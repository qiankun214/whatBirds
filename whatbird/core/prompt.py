"""提示词与模型输出解析。

所有与「怎么问模型、怎么解析模型回答」有关的东西都集中在这里，
方便后续换模型 / 调提示词时只改一个文件。

两种模式：
- ``full``（默认）：完整信息 —— 名称、七级分类学、识别依据、相似物种、
  栖息环境、分布范围、保护级别、备注；
- ``fast``：只问名称 + 七级分类学 + 置信度，去掉说明性字段。

为什么要分两种：视觉模型的输出是逐 token 生成的，**输出越长越慢**。
实测输出模板里说明性字段占了大部分 token，fast 模式把输出压到大约四分之一，
单张耗时可降一半以上，代价是没有那些描述性内容。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import ResponseParseError

MODE_FULL = "full"
MODE_FAST = "fast"
MODES = (MODE_FULL, MODE_FAST)

#: fast 模式下的输出上限（结构很小，512 足够，同时封顶最坏情况）
FAST_MAX_TOKENS = 512

_TAXONOMY_TEMPLATE: dict[str, str] = {
    "kingdom": "界（中文名 + 拉丁名）",
    "phylum": "门",
    "class": "纲",
    "order": "目",
    "family": "科",
    "genus": "属",
    "species": "种",
}

#: 完整模式的输出结构（同时用于提示词约束和前端字段对照）
OUTPUT_TEMPLATE: dict[str, Any] = {
    "is_bird": True,
    "confidence": 0.0,
    "chinese_name": "中文名",
    "english_name": "英文名",
    "scientific_name": "拉丁学名（属名 + 种加词）",
    "taxonomy": dict(_TAXONOMY_TEMPLATE),
    "identify_basis": ["支持该判断的形态特征，逐条列出"],
    "similar_species": [{"name": "易混淆物种", "difference": "与本种的区别要点"}],
    "habitat": "栖息环境",
    "distribution": "分布范围",
    "protection_status": "保护级别（如中国三有保护动物 / IUCN 无危）",
    "notes": "其他补充说明",
}

#: 快速模式的输出结构：只要名称与分类学
FAST_OUTPUT_TEMPLATE: dict[str, Any] = {
    "is_bird": True,
    "confidence": 0.0,
    "chinese_name": "中文名",
    "english_name": "英文名",
    "scientific_name": "拉丁学名（属名 + 种加词）",
    "taxonomy": dict(_TAXONOMY_TEMPLATE),
}

_COMMON_RULES = """工作要求：
1. 仔细分析用户提供的鸟类照片，依据体型、喙形、羽色、翼斑、尾羽、脚部等形态特征做出判断。
2. 给出尽可能准确的物种鉴定结果，包括中文名、英文名、拉丁学名，以及完整的七级分类学信息
   （界、门、纲、目、科、属、种），每一级都使用「中文名 + 拉丁名」的写法。
3. 如果照片中不是鸟，或无法判断到种，也必须如实说明：把 is_bird 设为 false（后者可保留
   已能确定的较高层级），并在 notes 中说明原因与可确定的最高分类层级。
4. confidence 表示你对物种判断的把握，取值 0~1 之间的小数。
5. 严格遵守事实，不确定的信息宁可留空，也不要编造。
6. 输出务必精炼：模型是逐 token 生成回答的，写长了会明显变慢。请遵守以下上限
   —— identify_basis 最多 3 条、similar_species 最多 2 个、notes 不超过 60 字、
   habitat / distribution / protection_status 各不超过 40 字；
   分类学各级只写「中文名 拉丁名」，不要附加解释；不确定的字段直接留空，不要展开论述。

输出要求：
- 只输出一个 JSON 对象，不要输出 markdown 代码块标记，不要输出任何解释性文字。
- JSON 的字段必须严格使用下面给定的结构与英文键名：
{schema}
"""

SYSTEM_PROMPT = "你是一位资深的鸟类分类学专家，精通中国及周边地区的鸟类形态识别与分类。\n\n" + _COMMON_RULES

FAST_SYSTEM_PROMPT = (
    "你是一位资深的鸟类分类学专家，精通中国及周边地区的鸟类形态识别与分类。\n"
    "本次只需要快速给出名称与分类学信息，不要输出任何说明性内容"
    "（不要识别依据、不要相似物种、不要栖息环境/分布/保护级别/备注）。\n\n"
    + _COMMON_RULES
)

USER_PROMPT = """请识别这张图片中的鸟类，按要求返回 JSON。

重点给出：
- 中文名 / 英文名 / 拉丁学名
- 分类学信息：界、门、纲、目、科、属、种（务必包含「科 family」与「属 genus」）
- 判断依据（识别特征）、易混淆物种及区别、栖息环境、分布范围、保护级别
"""

FAST_USER_PROMPT = """请识别这张图片中的鸟类，只返回名称与分类学信息的 JSON，字段值尽量简短，不要展开说明。"""


@dataclass(frozen=True, slots=True)
class PromptSet:
    """一次调用要用的提示词与输出上限。"""

    mode: str
    system: str
    user: str
    max_tokens: int | None  # None 表示用配置里的 max_tokens


def build_prompt_set(mode: str = MODE_FULL, *, config_max_tokens: int | None = None) -> PromptSet:
    """按模式渲染提示词。``config_max_tokens`` 为 None 时使用默认上限。"""
    template = FAST_OUTPUT_TEMPLATE if mode == MODE_FAST else OUTPUT_TEMPLATE
    system_template = FAST_SYSTEM_PROMPT if mode == MODE_FAST else SYSTEM_PROMPT
    user = FAST_USER_PROMPT if mode == MODE_FAST else USER_PROMPT

    schema = json.dumps(template, ensure_ascii=False, indent=2)
    return PromptSet(
        mode=mode,
        system=system_template.format(schema=schema),
        user=user,
        max_tokens=FAST_MAX_TOKENS if mode == MODE_FAST else config_max_tokens,
    )


def normalize_mode(mode: str | None) -> str:
    """把外部传入的模式规整为 full / fast。"""
    if mode is None:
        return MODE_FULL
    text = str(mode).strip().lower()
    if text in ("fast", "quick", "简单", "快速"):
        return MODE_FAST
    return MODE_FULL


def build_system_prompt(mode: str = MODE_FULL) -> str:
    """渲染系统提示词（把输出模板写进提示词，保证字段名一致）。"""
    return build_prompt_set(mode).system


def parse_json_object(text: str) -> dict[str, Any]:
    """从模型返回的文本里稳健地提取 JSON 对象。

    兼容三种常见情况：纯 JSON、被 ``` 代码块包裹、JSON 前后混有说明文字。
    """
    if not text or not text.strip():
        raise ResponseParseError(
            "模型没有返回任何内容",
            hint="可重试一次；若持续为空，请确认该模型/接入点支持图片输入",
        )

    cleaned = _strip_code_fence(text.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ResponseParseError(
                "模型返回的内容不是 JSON",
                hint=f"原始返回片段：{cleaned[:200]!r}",
            ) from None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ResponseParseError(
                "模型返回的 JSON 无法解析",
                hint=f"{exc}；片段：{cleaned[start : start + 200]!r}",
            ) from exc

    if not isinstance(data, dict):
        raise ResponseParseError(
            f"模型返回的 JSON 顶层应为对象，实际是 {type(data).__name__}",
            hint=f"片段：{cleaned[:200]!r}",
        )
    return data


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
