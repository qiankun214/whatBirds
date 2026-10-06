"""提示词与模型输出解析。

所有与「怎么问模型、怎么解析模型回答」有关的东西都集中在这里，
方便后续换模型 / 调提示词时只改一个文件。
"""

from __future__ import annotations

import json
from typing import Any

from .errors import ResponseParseError

#: 模型必须输出的 JSON 结构（同时用于提示词约束和前端字段对照）
OUTPUT_TEMPLATE: dict[str, Any] = {
    "is_bird": True,
    "confidence": 0.0,
    "chinese_name": "中文名",
    "english_name": "英文名",
    "scientific_name": "拉丁学名（属名 + 种加词）",
    "taxonomy": {
        "kingdom": "界（中文名 + 拉丁名）",
        "phylum": "门",
        "class": "纲",
        "order": "目",
        "family": "科",
        "genus": "属",
        "species": "种",
    },
    "identify_basis": ["支持该判断的形态特征，逐条列出"],
    "similar_species": [{"name": "易混淆物种", "difference": "与本种的区别要点"}],
    "habitat": "栖息环境",
    "distribution": "分布范围",
    "protection_status": "保护级别（如中国三有保护动物 / IUCN 无危）",
    "notes": "其他补充说明",
}

SYSTEM_PROMPT = """你是一位资深的鸟类分类学专家，精通中国及周边地区的鸟类形态识别与分类。

工作要求：
1. 仔细分析用户提供的鸟类照片，依据体型、喙形、羽色、翼斑、尾羽、脚部等形态特征做出判断。
2. 给出尽可能准确的物种鉴定结果，包括中文名、英文名、拉丁学名，以及完整的七级分类学信息
   （界、门、纲、目、科、属、种），每一级都使用「中文名 + 拉丁名」的写法。
3. 如果照片中不是鸟，或无法判断到种，也必须如实说明：把 is_bird 设为 false（后者可保留
   已能确定的较高层级），并在 notes 中说明原因与可确定的最高分类层级。
4. confidence 表示你对物种判断的把握，取值 0~1 之间的小数。
5. 严格遵守事实，不确定的信息宁可留空，也不要编造。

输出要求：
- 只输出一个 JSON 对象，不要输出 markdown 代码块标记，不要输出任何解释性文字。
- JSON 的字段必须严格使用下面给定的结构与英文键名：
{schema}
"""

USER_PROMPT = """请识别这张图片中的鸟类，按要求返回 JSON。

重点给出：
- 中文名 / 英文名 / 拉丁学名
- 分类学信息：界、门、纲、目、科、属、种（务必包含「科 family」与「属 genus」）
- 判断依据（识别特征）、易混淆物种及区别、栖息环境、分布范围、保护级别
"""


def build_system_prompt() -> str:
    """渲染系统提示词（把输出模板写进提示词，保证字段名一致）。"""
    schema = json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT.format(schema=schema)


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
