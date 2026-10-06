/**
 * WhatBird 前端：只与 REST API 通信，不含任何识别逻辑（前后端分离）。
 * 全部原生 JS，不依赖任何外部 CDN。
 */
(function () {
  "use strict";

  var CLIENT_HEADER = { "X-WhatBird-Client": "1" };
  var TAXONOMY_LABELS = [
    ["kingdom", "界"], ["phylum", "门"], ["class", "纲"], ["order", "目"],
    ["family", "科"], ["genus", "属"], ["species", "种"]
  ];
  var FACT_LABELS = [
    ["habitat", "栖息环境"], ["distribution", "分布范围"],
    ["protection_status", "保护级别"], ["notes", "备注"]
  ];

  var $ = function (id) { return document.getElementById(id); };

  // ------------------------------------------------------------ API 封装
  async function api(path, options) {
    options = options || {};
    var headers = Object.assign({}, CLIENT_HEADER, options.headers || {});
    var response;
    try {
      response = await fetch(path, Object.assign({}, options, { headers: headers }));
    } catch (err) {
      throw new Error("无法连接后端服务，请确认 `whatbird serve` 仍在运行。");
    }

    var payload = null;
    try { payload = await response.json(); } catch (err) { /* 非 JSON 响应 */ }

    if (!response.ok) {
      var detail = payload && (payload.error || payload.detail);
      if (detail && typeof detail === "object") {
        throw new Error(detail.message + (detail.hint ? "\n提示：" + detail.hint : ""));
      }
      throw new Error(typeof detail === "string" ? detail : "请求失败（HTTP " + response.status + "）");
    }
    return payload;
  }

  function setLoading(visible) { $("loading").hidden = !visible; }

  function showError(message) {
    var box = $("error-box");
    if (!message) { box.hidden = true; box.textContent = ""; return; }
    box.hidden = false;
    box.textContent = message;
  }

  // ------------------------------------------------------------ 配置
  async function loadConfig() {
    var pill = $("status-pill");
    try {
      var data = await api("/api/config");
      var config = data.config || {};
      $("config-file").textContent = "配置文件：" + (data.config_file || "—");
      $("cfg-base-url").value = config.base_url || "";
      $("cfg-model").value = config.model || "";
      $("cfg-timeout").value = config.timeout != null ? config.timeout : "";
      $("cfg-max-side").value = config.max_image_side != null ? config.max_image_side : "";
      $("cfg-api-key").value = "";
      $("cfg-api-key").placeholder = config.configured
        ? "已配置（" + config.api_key + "），留空则不修改"
        : "粘贴火山方舟 API Key";

      if (config.configured) {
        pill.textContent = "已配置 · " + (config.model || "");
        pill.className = "pill pill-ok";
      } else {
        pill.textContent = "尚未配置 API Key";
        pill.className = "pill pill-warn";
        $("config-panel").hidden = false;
      }
      return config;
    } catch (err) {
      pill.textContent = "后端不可用";
      pill.className = "pill pill-warn";
      showError(err.message);
      return null;
    }
  }

  async function saveConfig(event) {
    event.preventDefault();
    var msg = $("config-msg");
    var body = {
      api_key: $("cfg-api-key").value.trim(),
      base_url: $("cfg-base-url").value.trim(),
      model: $("cfg-model").value.trim()
    };
    var timeout = parseFloat($("cfg-timeout").value);
    var maxSide = parseInt($("cfg-max-side").value, 10);
    if (!isNaN(timeout)) { body.timeout = timeout; }
    if (!isNaN(maxSide)) { body.max_image_side = maxSide; }

    try {
      var data = await api("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      msg.hidden = false;
      msg.className = "msg";
      msg.textContent = data.message || "已保存";
      await loadConfig();
    } catch (err) {
      msg.hidden = false;
      msg.className = "msg error";
      msg.textContent = err.message;
    }
  }

  async function verifyConfig() {
    var msg = $("config-msg");
    msg.hidden = false;
    msg.className = "msg";
    msg.textContent = "正在测试连接…";
    try {
      var data = await api("/api/config/verify", { method: "POST" });
      msg.className = "msg";
      msg.textContent = "✓ 连接正常，模型回复：" + (data.model_reply || "");
    } catch (err) {
      msg.className = "msg error";
      msg.textContent = err.message;
    }
  }

  // ------------------------------------------------------------ 结果渲染
  function renderTaxonomy(taxonomy) {
    var tbody = $("r-taxonomy");
    tbody.innerHTML = "";
    var any = false;
    TAXONOMY_LABELS.forEach(function (pair) {
      var value = (taxonomy || {})[pair[0]] || "";
      if (!value) { return; }
      any = true;
      var tr = document.createElement("tr");
      var th = document.createElement("th");
      th.textContent = pair[1];
      var td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(th);
      tr.appendChild(td);
      tbody.appendChild(tr);
    });
    if (!any) {
      var tr = document.createElement("tr");
      var td = document.createElement("td");
      td.colSpan = 2;
      td.className = "muted";
      td.textContent = "模型未能给出分类学信息";
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
  }

  function renderList(container, items, render) {
    container.innerHTML = "";
    (items || []).forEach(function (item) {
      var li = document.createElement("li");
      render(li, item);
      container.appendChild(li);
    });
  }

  function renderResult(result, extraMeta) {
    $("result-empty").hidden = true;
    $("result-body").hidden = false;

    var names = result.names || {};
    $("r-chinese").textContent = names.chinese || "(未能确定中文名)";
    $("r-scientific").textContent = names.scientific || "(未能确定学名)";
    $("r-english").textContent = names.english || "";

    var confidence = typeof result.confidence === "number" ? result.confidence : 0;
    $("r-confidence").textContent = Math.round(confidence * 100) + "%";
    $("r-confidence-bar").style.width = Math.max(2, confidence * 100) + "%";
    $("r-not-bird").hidden = result.is_bird !== false;

    renderTaxonomy(result.taxonomy);

    var basis = result.identify_basis || [];
    $("r-basis-wrap").hidden = basis.length === 0;
    renderList($("r-basis"), basis, function (li, item) { li.textContent = item; });

    var similar = result.similar_species || [];
    $("r-similar-wrap").hidden = similar.length === 0;
    renderList($("r-similar"), similar, function (li, item) {
      var name = document.createElement("span");
      name.className = "similar-name";
      name.textContent = item.name || "—";
      li.appendChild(name);
      if (item.difference) { li.appendChild(document.createTextNode("：" + item.difference)); }
    });

    var facts = $("r-facts");
    facts.innerHTML = "";
    FACT_LABELS.forEach(function (pair) {
      var value = result[pair[0]];
      if (!value) { return; }
      var row = document.createElement("div");
      var dt = document.createElement("dt");
      dt.textContent = pair[1];
      var dd = document.createElement("dd");
      dd.textContent = value;
      row.appendChild(dt);
      row.appendChild(dd);
      facts.appendChild(row);
    });

    var meta = result.meta || {};
    var parts = [];
    if (meta.model) { parts.push("模型 " + meta.model); }
    if (extraMeta) { parts.push(extraMeta); }
    if (typeof meta.elapsed_seconds === "number") { parts.push("耗时 " + meta.elapsed_seconds.toFixed(2) + "s"); }
    $("result-meta").textContent = parts.join(" ｜ ");
  }

  // ------------------------------------------------------------ 识别
  async function recognizeUpload() {
    var input = $("file-input");
    if (!input.files || !input.files.length) { return; }

    var form = new FormData();
    form.append("file", input.files[0]);

    showError("");
    setLoading(true);
    try {
      var data = await api("/api/recognize", { method: "POST", body: form });
      renderResult(data.result, input.files[0].name);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function recognizePath() {
    var path = $("path-input").value.trim();
    if (!path) { showError("请先填写 jpg 文件路径。"); return; }

    showError("");
    setLoading(true);
    try {
      var data = await api("/api/recognize-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path })
      });
      renderResult(data.result, path);
    } catch (err) {
      showError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ------------------------------------------------------------ 事件绑定
  function bindTabs() {
    var tabs = document.querySelectorAll(".tab");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (other) { other.classList.remove("active"); });
        tab.classList.add("active");
        var mode = tab.dataset.mode;
        $("mode-upload").hidden = mode !== "upload";
        $("mode-path").hidden = mode !== "path";
        showError("");
      });
    });
  }

  function bindFilePicker() {
    var input = $("file-input");
    var zone = $("dropzone");
    var button = $("btn-recognize-upload");

    function showFile(file) {
      if (!file) { return; }
      var ok = /\.jpe?g$/i.test(file.name);
      $("preview-wrap").hidden = false;
      $("preview-name").textContent = file.name;
      $("preview-size").textContent = (file.size / 1024).toFixed(0) + " KB";
      button.disabled = !ok;
      if (ok) {
        showError("");
        $("preview").src = URL.createObjectURL(file);
      } else {
        showError("仅支持 .jpg / .jpeg 图片，请重新选择。");
        $("preview").removeAttribute("src");
      }
    }

    input.addEventListener("change", function () { showFile(input.files[0]); });

    ["dragenter", "dragover"].forEach(function (type) {
      zone.addEventListener(type, function (event) {
        event.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (type) {
      zone.addEventListener(type, function (event) {
        event.preventDefault();
        zone.classList.remove("dragover");
      });
    });
    zone.addEventListener("drop", function (event) {
      var files = event.dataTransfer && event.dataTransfer.files;
      if (files && files.length) {
        // 通过 DataTransfer 把文件塞回 input，保证后续上传逻辑统一
        var transfer = new DataTransfer();
        transfer.items.add(files[0]);
        input.files = transfer.files;
        showFile(files[0]);
      }
    });

    button.addEventListener("click", recognizeUpload);
  }

  function init() {
    bindTabs();
    bindFilePicker();
    $("toggle-config").addEventListener("click", function () {
      var panel = $("config-panel");
      panel.hidden = !panel.hidden;
      if (!panel.hidden) { panel.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
    });
    $("save-config").addEventListener("click", saveConfig);
    $("verify-config").addEventListener("click", verifyConfig);
    $("btn-recognize-path").addEventListener("click", recognizePath);
    $("path-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { recognizePath(); }
    });
    loadConfig();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
