"""core 识别流程测试：不联网，用假客户端替换方舟调用。"""

from __future__ import annotations

import json

import pytest

from whatbird.core import ArkConfig, ModelCallError, ResponseParseError
from whatbird.core.ark_client import _extract_text, _translate_error
from whatbird.core.prompt import parse_json_object
from whatbird.core.recognizer import BirdRecognizer

MODEL_JSON = {
    "is_bird": True,
    "confidence": 0.93,
    "chinese_name": "白鹡鸰",
    "english_name": "White Wagtail",
    "scientific_name": "Motacilla alba",
    "taxonomy": {
        "kingdom": "动物界 Animalia",
        "phylum": "脊索动物门 Chordata",
        "class": "鸟纲 Aves",
        "order": "雀形目 Passeriformes",
        "family": "鹡鸰科 Motacillidae",
        "genus": "鹡鸰属 Motacilla",
        "species": "白鹡鸰 Motacilla alba",
    },
    "identify_basis": ["黑白灰配色", "尾羽细长且不停摆动"],
    "similar_species": [{"name": "灰鹡鸰", "difference": "灰鹡鸰背部偏灰、下体黄色"}],
    "habitat": "水边、农田、城市空地",
    "distribution": "欧亚大陆广布",
    "protection_status": "中国三有保护动物",
    "notes": "亚种较多",
}


class FakeClient:
    """假的方舟客户端：把预设文本原样返回，并记录调用参数。"""

    def __init__(self, reply: str = "") -> None:
        self.reply = reply or json.dumps(MODEL_JSON, ensure_ascii=False)
        self.image_calls: list[dict] = []
        self.pinged = False

    def complete_with_image(self, *, system_prompt, user_prompt, image_data_url) -> str:
        self.image_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "image_data_url": image_data_url,
            }
        )
        return self.reply

    def ping(self) -> str:
        self.pinged = True
        return "正常"


def _recognizer(reply: str = "") -> tuple[BirdRecognizer, FakeClient]:
    client = FakeClient(reply)
    return BirdRecognizer(ArkConfig(api_key="sk-test"), client=client), client


def test_recognize_returns_taxonomy(jpeg_factory):
    recognizer, client = _recognizer()
    result = recognizer.recognize(str(jpeg_factory()))

    assert result.chinese_name == "白鹡鸰"
    assert result.scientific_name == "Motacilla alba"
    assert result.taxonomy.family == "鹡鸰科 Motacillidae"
    assert result.taxonomy.genus == "鹡鸰属 Motacilla"
    assert result.confidence == pytest.approx(0.93)
    assert result.model == "doubao-seed-2-1-lite"
    assert result.elapsed_seconds >= 0

    # 图片确实以 data URL 形式传给了模型
    assert client.image_calls[0]["image_data_url"].startswith("data:image/jpeg;base64,")


def test_result_to_dict_contract(jpeg_factory):
    recognizer, _ = _recognizer()
    payload = recognizer.recognize(str(jpeg_factory())).to_dict()

    assert payload["names"]["chinese"] == "白鹡鸰"
    assert payload["taxonomy"]["family"] == "鹡鸰科 Motacillidae"
    assert payload["taxonomy"]["genus"] == "鹡鸰属 Motacilla"
    assert payload["similar_species"][0]["name"] == "灰鹡鸰"
    assert payload["meta"]["model"] == "doubao-seed-2-1-lite"
    assert "raw_response" not in payload["meta"]


def test_markdown_fenced_reply_is_parsed(jpeg_factory):
    fenced = "```json\n" + json.dumps(MODEL_JSON, ensure_ascii=False) + "\n```"
    recognizer, _ = _recognizer(fenced)
    assert recognizer.recognize(str(jpeg_factory())).chinese_name == "白鹡鸰"


def test_reply_with_prose_around_json_is_parsed(jpeg_factory):
    noisy = "好的，识别结果如下：\n" + json.dumps(MODEL_JSON, ensure_ascii=False) + "\n希望有帮助！"
    recognizer, _ = _recognizer(noisy)
    assert recognizer.recognize(str(jpeg_factory())).chinese_name == "白鹡鸰"


def test_non_json_reply_raises(jpeg_factory):
    recognizer, _ = _recognizer("抱歉，我无法识别这张图片。")
    with pytest.raises(ResponseParseError):
        recognizer.recognize(str(jpeg_factory()))


def test_missing_fields_are_tolerated(jpeg_factory):
    recognizer, _ = _recognizer(json.dumps({"chinese_name": "麻雀"}))
    result = recognizer.recognize(str(jpeg_factory()))
    assert result.chinese_name == "麻雀"
    assert result.taxonomy.is_empty
    assert result.is_bird is True  # 缺省按「是鸟」处理


def test_confidence_string_and_not_bird(jpeg_factory):
    reply = json.dumps({"is_bird": False, "confidence": "85%", "notes": "图中是石头"})
    recognizer, _ = _recognizer(reply)
    result = recognizer.recognize(str(jpeg_factory()))
    assert result.is_bird is False
    assert result.confidence == pytest.approx(0.85)


def test_missing_config_rejected_before_network(jpeg_factory):
    """"未配置 Key 时必须立刻报错，不能发起请求。"""
    client = FakeClient()
    recognizer = BirdRecognizer(ArkConfig(api_key=""), client=client)
    from whatbird.core import ConfigError

    with pytest.raises(ConfigError):
        recognizer.recognize(str(jpeg_factory()))
    assert client.image_calls == []


def test_verify_connection_uses_ping():
    recognizer, client = _recognizer()
    assert recognizer.verify_connection() == "正常"
    assert client.pinged is True


# ------------------------------------------------------------------ 提示词解析
@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '前言 {"a": 1} 后记',
    ],
)
def test_parse_json_object_variants(text):
    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_empty_raises():
    with pytest.raises(ResponseParseError):
        parse_json_object("   ")


def test_parse_json_object_array_raises():
    with pytest.raises(ResponseParseError):
        parse_json_object("[1, 2, 3]")


def test_system_prompt_contains_schema():
    from whatbird.core.prompt import build_system_prompt

    prompt = build_system_prompt()
    assert "family" in prompt and "genus" in prompt and "taxonomy" in prompt


# ------------------------------------------------------------------ 异常翻译
class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]


def test_extract_text_from_string_content():
    assert _extract_text(_FakeResponse("你好")) == "你好"


def test_extract_text_from_list_content():
    content = [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]
    assert _extract_text(_FakeResponse(content)) == "第一段\n第二段"


def test_extract_empty_content_raises():
    with pytest.raises(ModelCallError):
        _extract_text(_FakeResponse(""))


@pytest.mark.parametrize(
    "status,expected",
    [(401, "鉴权失败"), (403, "无权访问"), (404, "不存在"), (429, "限流"), (400, "被方舟拒绝")],
)
def test_error_translation_mentions_status(status, expected):
    error = Exception("boom")
    error.status_code = status
    assert expected in _translate_error(error, model="doubao-seed-2-1-lite").message


def test_timeout_error_translation():
    class APITimeoutError(Exception):
        pass

    assert "超时" in _translate_error(APITimeoutError("t"), model="m").message
