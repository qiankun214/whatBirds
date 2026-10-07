"""core 层的数据模型：鸟类识别结果与其分类学信息。

这些对象是「core -> 上层」的唯一数据契约，CLI 与 Web 共享同一定义，
因此新增字段时上层可以自动获得（``to_dict`` 就是对外 JSON 结构）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: 分类学层级顺序（从高到低）与中文名
TAXONOMY_LEVELS: tuple[tuple[str, str], ...] = (
    ("kingdom", "界"),
    ("phylum", "门"),
    ("class", "纲"),
    ("order", "目"),
    ("family", "科"),
    ("genus", "属"),
    ("species", "种"),
)


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        return [text for item in value if (text := _as_str(item))]
    return [_as_str(value)]


def _as_confidence(value: Any) -> float:
    """把 0.92 / "0.92" / "92%" / 92 统一成 0~1 的浮点数。"""
    number: float | None = None
    if isinstance(value, bool):
        number = float(value)
    elif isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip().rstrip("%")
        try:
            number = float(text)
        except ValueError:
            return 0.0
        if value.strip().endswith("%") or number > 1.0:
            number /= 100.0
    if number is None:
        return 0.0
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    return min(max(number, 0.0), 1.0)


@dataclass(slots=True)
class Taxonomy:
    """七级分类学信息（界 / 门 / 纲 / 目 / 科 / 属 / 种）。"""

    kingdom: str = ""
    phylum: str = ""
    class_: str = ""
    order: str = ""
    family: str = ""
    genus: str = ""
    species: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Taxonomy":
        data = data or {}
        return cls(
            kingdom=_as_str(data.get("kingdom")),
            phylum=_as_str(data.get("phylum")),
            class_=_as_str(data.get("class") or data.get("class_")),
            order=_as_str(data.get("order")),
            family=_as_str(data.get("family")),
            genus=_as_str(data.get("genus")),
            species=_as_str(data.get("species")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kingdom": self.kingdom,
            "phylum": self.phylum,
            "class": self.class_,
            "order": self.order,
            "family": self.family,
            "genus": self.genus,
            "species": self.species,
        }

    @property
    def is_empty(self) -> bool:
        return not any(self.to_dict().values())

    def as_rows(self) -> list[tuple[str, str, str]]:
        """返回 ``[(层级 key, 中文层级名, 取值)]``，便于 CLI / 前端渲染表格。"""
        values = self.to_dict()
        return [(key, label, values.get(key, "")) for key, label in TAXONOMY_LEVELS]


@dataclass(slots=True)
class SimilarSpecies:
    """易混淆的相似物种及区分要点。"""

    name: str = ""
    difference: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> "SimilarSpecies":
        if isinstance(data, str):
            return cls(name=data.strip())
        return cls(
            name=_as_str(data.get("name") or data.get("species")),
            difference=_as_str(data.get("difference") or data.get("distinguish")),
        )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "difference": self.difference}


@dataclass(slots=True)
class BirdIdentification:
    """一次鸟类识别的完整结果：命名信息 + 分类学 + 说明 + 调用元数据。"""

    is_bird: bool = False
    confidence: float = 0.0
    chinese_name: str = ""
    english_name: str = ""
    scientific_name: str = ""
    taxonomy: Taxonomy = field(default_factory=Taxonomy)
    identify_basis: list[str] = field(default_factory=list)
    similar_species: list[SimilarSpecies] = field(default_factory=list)
    habitat: str = ""
    distribution: str = ""
    protection_status: str = ""
    notes: str = ""
    # 元数据
    source_image: str = ""
    model: str = ""
    elapsed_seconds: float = 0.0
    raw_response: str = ""

    # --------------------------------------------------------------- 构造
    @classmethod
    def from_model_output(
        cls,
        data: dict[str, Any],
        *,
        source_image: str = "",
        model: str = "",
        elapsed_seconds: float = 0.0,
        raw_response: str = "",
    ) -> "BirdIdentification":
        """把模型返回的 JSON 字典规整成强类型对象（容忍字段缺失/类型偏差）。"""
        similar_raw = data.get("similar_species") or data.get("similar") or []
        if isinstance(similar_raw, dict):
            similar_raw = [similar_raw]
        return cls(
            is_bird=bool(data.get("is_bird", True)),
            confidence=_as_confidence(data.get("confidence")),
            chinese_name=_as_str(data.get("chinese_name") or data.get("name_zh")),
            english_name=_as_str(data.get("english_name") or data.get("name_en")),
            scientific_name=_as_str(data.get("scientific_name") or data.get("latin_name")),
            taxonomy=Taxonomy.from_dict(data.get("taxonomy")),
            identify_basis=_as_str_list(data.get("identify_basis") or data.get("basis")),
            similar_species=[
                SimilarSpecies.from_dict(item)
                for item in similar_raw
                if isinstance(item, (dict, str))
            ],
            habitat=_as_str(data.get("habitat")),
            distribution=_as_str(data.get("distribution")),
            protection_status=_as_str(data.get("protection_status") or data.get("protection")),
            notes=_as_str(data.get("notes") or data.get("comment")),
            source_image=source_image,
            model=model,
            elapsed_seconds=elapsed_seconds,
            raw_response=raw_response,
        )

    # --------------------------------------------------------------- 输出
    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """对外 JSON 结构（Web API 与 CLI --json 共用）。"""
        payload: dict[str, Any] = {
            "is_bird": self.is_bird,
            "confidence": round(self.confidence, 4),
            "names": {
                "chinese": self.chinese_name,
                "english": self.english_name,
                "scientific": self.scientific_name,
            },
            "taxonomy": self.taxonomy.to_dict(),
            "identify_basis": list(self.identify_basis),
            "similar_species": [item.to_dict() for item in self.similar_species],
            "habitat": self.habitat,
            "distribution": self.distribution,
            "protection_status": self.protection_status,
            "notes": self.notes,
            "meta": {
                "model": self.model,
                "source_image": self.source_image,
                "elapsed_seconds": round(self.elapsed_seconds, 3),
            },
        }
        if include_raw:
            payload["meta"]["raw_response"] = self.raw_response
        return payload

    # --------------------------------------------------------------- 展示
    def summary_line(self) -> str:
        """一行摘要，CLI 控制台用。"""
        name = self.chinese_name or self.scientific_name or "未识别"
        extra = f"（{self.scientific_name}）" if self.scientific_name and self.chinese_name else ""
        return f"{name}{extra}  置信度 {self.confidence:.0%}"


# --------------------------------------------------------------------- 批处理
@dataclass(slots=True)
class BatchItem:
    """批处理里的一张图的识别结果（成功或失败）。"""

    index: int  # 从 1 开始
    total: int
    source: str
    status: str  # ok / error
    identification: BirdIdentification | None = None
    error_message: str = ""
    error_hint: str = ""
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def display_name(self) -> str:
        if self.identification is None:
            return "—"
        return self.identification.chinese_name or self.identification.scientific_name or "未识别"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "total": self.total,
            "source": self.source,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "result": self.identification.to_dict() if self.identification else None,
            "error": None,
        }
        if self.error_message:
            payload["error"] = {"message": self.error_message, "hint": self.error_hint}
        return payload


@dataclass(slots=True)
class BatchReport:
    """一次批处理的整体结果（CLI 与 Web 共用同一份结构）。"""

    total: int
    items: list[BatchItem] = field(default_factory=list)
    stopped: bool = False
    elapsed_seconds: float = 0.0

    @property
    def completed(self) -> int:
        return len(self.items)

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.ok)

    @property
    def failed(self) -> int:
        return self.completed - self.succeeded

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "completed": self.completed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "stopped": self.stopped,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "items": [item.to_dict() for item in self.items],
        }
