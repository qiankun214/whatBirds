"""前端契约测试：静态校验 HTML / CSS / JS 三者一致。

这一类 bug（JS 里 `$("xxx")` 写的 id 在页面上不存在）不会有任何编译期报错，
却会让整页脚本在初始化时抛异常、向导完全点不动，所以用测试把它钉死。
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[1] / "whatbird" / "web" / "static"
HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")


def html_ids() -> list[str]:
    return re.findall(r'\bid="([^"]+)"', HTML)


def js_element_ids() -> set[str]:
    return set(re.findall(r'\$\("([^"]+)"\)', JS))


def js_class_selectors() -> set[str]:
    """取出 querySelectorAll 里的简单类选择器（.foo 形式）。"""
    selectors = re.findall(r'querySelectorAll\("([^"]+)"\)', JS)
    return {item[1:] for item in selectors if item.startswith(".") and " " not in item}


def has_class(html: str, name: str) -> bool:
    return re.search(rf'class="[^"]*\b{re.escape(name)}\b[^"]*"', html) is not None


# ------------------------------------------------------------------ 元素契约
def test_html_ids_are_unique():
    ids = html_ids()
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    assert not duplicates, f"index.html 存在重复 id（$() 只会取到第一个）：{duplicates}"


def test_every_referenced_element_id_exists():
    missing = sorted(js_element_ids() - set(html_ids()))
    assert not missing, f"app.js 引用了页面上不存在的元素 id：{missing}"


def test_class_selectors_used_by_js_exist():
    missing = sorted(name for name in js_class_selectors() if not has_class(HTML, name))
    assert not missing, f"app.js 用到的类名在页面上不存在：{missing}"


def test_structural_selectors_used_by_js_exist():
    # renderStep 依赖这两个结构
    assert 'id="wz-steps"' in HTML
    assert has_class(HTML, "wizard-body")
    assert 'class="tab active"' in HTML
    for step in (1, 2, 3, 4):
        assert f'data-step="{step}"' in HTML, f"缺少向导步骤 {step}"
        assert f'data-step-dot="{step}"' in HTML, f"缺少步骤指示点 {step}"


def test_css_styles_the_wizard():
    for selector in (".modal-card", ".steps", ".wizard-body", ".wizard-foot", ".summary", ".alert-ok"):
        assert selector in CSS, selector


def test_css_guarantees_hidden_actually_hides():
    """回归防护：`.loading`/`.empty` 这类元素带 display 样式，
    会盖过浏览器默认的 [hidden]{display:none}，必须由全局兜底规则压住。"""
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none", CSS), "style.css 缺少全局 [hidden] 兜底规则"


def test_elements_toggled_by_hidden_are_covered_by_the_global_guard():
    """找出「带 hidden 属性、但其 id/class 被 CSS 设了 display」的元素。

    这类元素正是必须依赖全局 [hidden] 兜底的场景 —— `.loading` 和 `.empty`
    曾经因此一直显示（加载遮罩不消失、识别完成后空状态不隐藏）。
    """
    guarded: list[str] = []
    for tag in re.findall(r"<[^>]*\bhidden\b[^>]*>", HTML):
        selectors: list[str] = []
        element_id = re.search(r'\bid="([^"]+)"', tag)
        classes = re.search(r'\bclass="([^"]*)"', tag)
        if element_id:
            selectors.append("#" + element_id.group(1))
        if classes:
            selectors.extend("." + name for name in classes.group(1).split())

        for selector in selectors:
            if re.search(rf"{re.escape(selector)}\s*\{{[^}}]*display:", CSS):
                guarded.append(selector)

    assert guarded, "预期至少有一个 hidden 元素被 CSS 设了 display，否则这条兜底规则无从验证"


# ------------------------------------------------------------------ 接口契约
def js_api_paths() -> set[str]:
    return set(re.findall(r'api\(\s*"(/api/[^"]+)"', JS))


def test_frontend_only_calls_existing_endpoints():
    """前端调用的每个 /api 路径都必须真的注册过后端路由。"""
    web_app = importlib.import_module("whatbird.web.app")
    registered = set(web_app.app.openapi()["paths"])

    used = js_api_paths()
    assert used, "没有解析到前端调用的接口，正则可能失效了"

    missing = sorted(path for path in used if path not in registered)
    assert not missing, f"前端调用了不存在的接口：{missing}（已注册：{sorted(registered)}）"


def test_wizard_flows_are_wired_to_endpoints():
    """配置向导的四个动作必须都接了对应接口。"""
    used = js_api_paths()
    assert "/api/config" in used          # 读取 + 保存
    assert "/api/config/verify" in used   # 测试连接
    assert "/api/recognize" in used       # 上传识别
    assert "/api/recognize-path" in used  # 路径识别


def test_client_header_is_sent_with_every_request():
    """防跨站请求头必须统一注入，不能漏。"""
    assert "X-WhatBird-Client" in JS
    assert re.search(r'var CLIENT_HEADER = \{[^}]*X-WhatBird-Client', JS)
