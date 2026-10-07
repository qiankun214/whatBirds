/**
 * WhatBird 前端：只与 REST API 通信，不含任何识别逻辑（前后端分离）。
 * 全部原生 JS，不依赖任何外部 CDN。
 *
 * 两块功能：
 *   1) 配置向导（4 步）：获取 Key -> 接入信息 -> 高级参数 -> 测试并保存
 *   2) 鸟类识别：上传图片 / 服务器文件路径 -> 渲染分类学结果
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
  var TOTAL_STEPS = 4;
  var NEXT_LABELS = { 1: "下一步", 2: "下一步", 3: "下一步", 4: "完成" };

  var state = {
    configured: false,
    maskedKey: "",
    configFile: "",
    configFileExists: false,
    loadedFrom: null,
    configWritable: true,
    needsConsent: false,
    projectCandidate: null,
    busy: false,
    location: null,
    config: null,
    defaults: null,
    step: 1,
    saved: false
  };

  function $(id) { return document.getElementById(id); }

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

  function setMessage(element, text, kind) {
    if (!text) { element.hidden = true; element.textContent = ""; element.className = "alert"; return; }
    element.hidden = false;
    element.className = "alert " + (kind === "ok" ? "alert-ok" : kind === "warn" ? "alert-warn" : "alert-error");
    element.textContent = text;
  }

  function maskKey(value) {
    if (!value) { return ""; }
    if (value.length <= 8) { return "*".repeat(value.length); }
    return value.slice(0, 4) + "****" + value.slice(-4);
  }

  // ------------------------------------------------------------ 配置状态
  async function loadConfig() {
    var pill = $("status-pill");
    try {
      var data = await api("/api/config");
      var config = data.config || {};

      state.configured = !!config.configured;
      state.maskedKey = config.api_key || "";
      state.config = config;
      state.configFile = data.config_file || "";
      state.configFileExists = !!data.config_file_exists;
      state.loadedFrom = data.config_loaded_from || null;
      state.configWritable = data.config_writable !== false;
      state.location = data.config_location || null;
      state.needsConsent = !!(state.location && state.location.needs_consent);
      state.projectCandidate = (state.location && state.location.project_candidate) || null;
      state.defaults = data.defaults || null;

      $("wz-config-file").textContent = state.configFile || "—";
      $("footer-config").textContent =
        (state.configFileExists ? "配置文件：" : "配置文件（尚未创建）：") + state.configFile +
        (state.location && state.location.label ? "｜" + state.location.label : "");

      if (state.configured) {
        pill.textContent = "已配置 · " + (config.model || "");
        pill.className = "pill pill-ok";
      } else {
        pill.textContent = "尚未配置 API Key";
        pill.className = "pill pill-warn";
      }
      return config;
    } catch (err) {
      pill.textContent = "后端不可用";
      pill.className = "pill pill-warn";
      showError(err.message);
      return null;
    }
  }

  // ------------------------------------------------------------ 向导：表单
  function setIfPresent(id, value) {
    $(id).value = (value === null || value === undefined) ? "" : value;
  }

  function fillIdentity(config) {
    setIfPresent("cfg-base-url", config.base_url);
    setIfPresent("cfg-model", config.model);
    $("cfg-api-key").value = "";
    $("cfg-api-key").type = "password";
    $("toggle-key").textContent = "显示";
  }

  function fillAdvanced(config) {
    setIfPresent("cfg-timeout", config.timeout);
    setIfPresent("cfg-max-tokens", config.max_tokens);
    setIfPresent("cfg-temperature", config.temperature);
    setIfPresent("cfg-max-side", config.max_image_side);
    setIfPresent("cfg-jpeg-quality", config.jpeg_quality);
  }

  function numberOrNull(id, integer) {
    var raw = $(id).value.trim();
    if (!raw) { return null; }
    var value = integer ? parseInt(raw, 10) : parseFloat(raw);
    return isNaN(value) ? null : value;
  }

  function collectDraft() {
    return {
      api_key: $("cfg-api-key").value.trim(),
      base_url: $("cfg-base-url").value.trim(),
      model: $("cfg-model").value.trim(),
      timeout: numberOrNull("cfg-timeout", false),
      max_tokens: numberOrNull("cfg-max-tokens", true),
      temperature: numberOrNull("cfg-temperature", false),
      max_image_side: numberOrNull("cfg-max-side", true),
      jpeg_quality: numberOrNull("cfg-jpeg-quality", true),
      allow_project_fallback: $("wz-allow-project").checked
    };
  }

  function keyHint() {
    $("key-hint").textContent = state.configured
      ? "当前已保存：" + state.maskedKey + "（留空表示不修改，只在需要更换 Key 时才填写）"
      : "还没有配置过 Key，请把控制台里复制的 Key 粘贴进来。";
  }

  // ------------------------------------------------------------ 向导：流程
  function openWizard(startStep) {
    state.saved = false;
    setMessage($("wz-msg"), "");
    setMessage($("wz-result"), "");
    // 每次打开都按已保存的配置重新填充，避免上次未保存的输入残留
    if (state.config) {
      fillIdentity(state.config);
      fillAdvanced(state.config);
    } else if (state.defaults) {
      fillAdvanced(state.defaults);
    }
    goStep(startStep || (state.configured ? 2 : 1), true);
    $("wizard").hidden = false;
    document.body.classList.add("modal-open");
    $("wz-next").focus();
  }

  function closeWizard() {
    $("wizard").hidden = true;
    document.body.classList.remove("modal-open");
  }

  function goStep(step, silent) {
    state.step = Math.min(Math.max(step, 1), TOTAL_STEPS);

    var sections = document.querySelectorAll(".wizard-body section[data-step]");
    sections.forEach(function (section) {
      section.hidden = Number(section.dataset.step) !== state.step;
    });

    document.querySelectorAll("#wz-steps li").forEach(function (item) {
      var index = Number(item.dataset.stepDot);
      item.classList.toggle("active", index === state.step);
      item.classList.toggle("done", index < state.step);
    });

    $("wz-prev").hidden = state.step === 1;
    $("wz-next").textContent = NEXT_LABELS[state.step];
    $("wz-progress").textContent = "第 " + state.step + " / " + TOTAL_STEPS + " 步";
    if (!silent) { setMessage($("wz-msg"), ""); setMessage($("wz-result"), ""); }

    if (state.step === 2) { keyHint(); }
    if (state.step === 4) { renderSummary(); renderLocationWarning(); renderConsent(); }
  }

  function renderConsent() {
    var project = state.projectCandidate || {};
    $("wz-consent").hidden = !state.needsConsent;
    $("wz-project-path").textContent = project.path || "（项目内 .whatbird/config.json）";
    if (!state.needsConsent) { $("wz-allow-project").checked = false; }
    updateSaveState();
  }

  function renderLocationWarning() {
    var box = $("wz-location-warning");
    var migrated = state.loadedFrom && state.loadedFrom !== state.configFile
      ? "\n现有配置读取自 " + state.loadedFrom + "；下次保存会写入上面的新位置。"
      : "";

    if (state.needsConsent) {
      var project = state.projectCandidate || {};
      var reason = state.location && state.location.preferred_error
        ? "（" + state.location.preferred_error + "）"
        : "";
      setMessage(
        box,
        "用户级配置位置都不可写" + reason + "。\n" +
        "程序不会自动把密钥放进项目里。若你同意，可以保存到：" + (project.path || "-") + "\n" +
        "该文件已被 .gitignore 忽略，不会被提交到 git；勾选下方复选框后才能保存。" + migrated,
        "warn"
      );
      return;
    }

    if (!state.configWritable) {
      var interp = (state.location && state.location.interpreter) || {};
      setMessage(
        box,
        "当前所有候选配置位置都不可写，直接保存会失败。\n" +
        "处理办法：在普通（非受限）终端里重新运行 whatbird serve；" +
        "或用环境变量指定一个可写位置：" +
        "set WHATBIRD_CONFIG_FILE=D:\\whatbird\\config.json\n" +
        "也可以在命令行运行 whatbird config doctor 查看每个位置的具体原因。" +
        (interp.hint ? "\n\n" + interp.hint : ""),
        "error"
      );
      return;
    }

    if (state.location && state.location.is_fallback) {
      var why = state.location.preferred_error
        ? "\n首选位置 " + state.location.preferred_path + " 写入被系统拒绝：" + state.location.preferred_error
        : "";
      setMessage(
        box,
        "当前使用：" + state.location.label + "\n" +
        (state.location.warning || "") + why + migrated +
        "\n若想让密钥完全放在项目之外，请在普通终端里运行 whatbird serve" +
        "（每次启动都会重新检测，可写时会自动切回用户目录），" +
        "或用环境变量指定位置：set WHATBIRD_CONFIG_FILE=D:\\whatbird\\config.json",
        "warn"
      );
      return;
    }

    setMessage(box, migrated ? migrated.replace(/^\n/, "") : "");
  }

  function validateIdentity() {
    var key = $("cfg-api-key").value.trim();
    if (!key && !state.configured) {
      setMessage($("wz-msg"), "请先填写 API Key。", "error");
      return false;
    }
    var baseUrl = $("cfg-base-url").value.trim();
    if (!baseUrl || !/^https?:\/\//i.test(baseUrl)) {
      setMessage($("wz-msg"), "Base URL 需要以 http:// 或 https:// 开头。", "error");
      return false;
    }
    if (!$("cfg-model").value.trim()) {
      setMessage($("wz-msg"), "请填写模型 ID 或推理接入点 ID。", "error");
      return false;
    }
    setMessage($("wz-msg"), "");
    return true;
  }

  function renderSummary() {
    var draft = collectDraft();
    var rows = [
      ["API Key", draft.api_key ? maskKey(draft.api_key) + "（保存后生效）"
        : (state.configured ? "沿用已保存的 " + state.maskedKey : "未填写")],
      ["Base URL", draft.base_url || "（默认）"],
      ["模型 / 接入点", draft.model || "（默认）"],
      ["超时", draft.timeout !== null ? draft.timeout + " 秒" : "（默认）"],
      ["max_tokens", draft.max_tokens !== null ? draft.max_tokens : "（默认）"],
      ["temperature", draft.temperature !== null ? draft.temperature : "（默认）"],
      ["图片最长边", draft.max_image_side !== null ? draft.max_image_side + " px" : "（默认）"],
      ["JPEG 质量", draft.jpeg_quality !== null ? draft.jpeg_quality : "（默认）"],
      ["位置类型", state.needsConsent
        ? "用户级位置不可写（需勾选同意后才用项目内位置）"
        : (state.location
          ? state.location.label + (state.location.is_fallback ? "（降级）" : "")
          : "（默认）")],
      ["保存位置", state.configFile || "（默认用户目录）"]
    ];

    var box = $("wz-summary");
    box.innerHTML = "";
    rows.forEach(function (pair) {
      var row = document.createElement("div");
      var dt = document.createElement("dt");
      dt.textContent = pair[0];
      var dd = document.createElement("dd");
      dd.textContent = pair[1];
      row.appendChild(dt);
      row.appendChild(dd);
      box.appendChild(row);
    });
  }

  function setBusy(busy) {
    state.busy = busy;
    ["wz-verify", "wz-recheck", "wz-next", "wz-prev"].forEach(function (id) {
      $(id).disabled = busy;
    });
    updateSaveState();
  }

  function updateSaveState() {
    // 需要同意才能保存时，勾选框没打勾就不让点保存（避免点了才报错）
    var blocked = state.needsConsent && !$("wz-allow-project").checked;
    $("wz-save").disabled = state.busy || blocked;
    $("wz-save").title = blocked ? "请先勾选同意，或改用可写的用户目录" : "";
  }

  async function recheckLocation() {
    setBusy(true);
    setMessage($("wz-result"), "正在重新检测配置位置…");
    try {
      await loadConfig();
      renderSummary();
      renderLocationWarning();
      setMessage(
        $("wz-result"),
        "已重新检测：" + (state.configFile || "-") + "\n位置：" +
        (state.location ? state.location.label : "-") +
        (state.configWritable ? "（可写）" : "（不可写）"),
        state.configWritable ? "ok" : "error"
      );
    } catch (err) {
      setMessage($("wz-result"), "✗ 重新检测失败：" + err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  async function verifyDraft() {
    if (!validateIdentity()) { return; }
    setBusy(true);
    setMessage($("wz-result"), "正在测试连接…");
    try {
      var data = await api("/api/config/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectDraft())
      });
      var tested = data.tested || {};
      setMessage(
        $("wz-result"),
        "✓ 连接正常（模型 " + (tested.model || "-") + "，Key " + (tested.api_key || "-") + "）\n" +
        "模型回复：" + (data.model_reply || "") + "\n注意：这次测试没有写盘，确认无误后请点「保存配置」。",
        "ok"
      );
    } catch (err) {
      setMessage($("wz-result"), "✗ 测试失败：" + err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  async function saveFromWizard() {
    if (!validateIdentity()) { goStep(2, true); return; }
    setBusy(true);
    setMessage($("wz-result"), "正在保存…");
    try {
      var data = await api("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectDraft())
      });
      state.saved = true;
      await loadConfig();
      var extra = state.location && state.location.warning
        ? "\n说明：" + state.location.warning
        : "";
      setMessage(
        $("wz-result"),
        "✓ " + (data.message || "配置已保存") + "\n文件位置：" + (data.config_file || state.configFile) +
        extra + "\n现在就可以回到主界面开始识别了。",
        "ok"
      );
      $("wz-next").textContent = "完成";
      renderSummary();
      renderLocationWarning();
    } catch (err) {
      setMessage($("wz-result"), "✗ 保存失败：" + err.message, "error");
    } finally {
      setBusy(false);
    }
  }

  function bindWizard() {
    $("open-wizard").addEventListener("click", function () {
      openWizard(state.configured ? 2 : 1);
    });

    $("wizard").addEventListener("click", function (event) {
      if (event.target.dataset && event.target.dataset.close !== undefined) { closeWizard(); }
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !$("wizard").hidden) { closeWizard(); }
    });

    $("wz-prev").addEventListener("click", function () { goStep(state.step - 1); });

    $("wz-next").addEventListener("click", function () {
      if (state.step < TOTAL_STEPS) {
        if (state.step === 2 && !validateIdentity()) { return; }
        goStep(state.step + 1);
        return;
      }
      closeWizard();
      if (state.saved) { showError(""); }
    });

    $("toggle-key").addEventListener("click", function () {
      var input = $("cfg-api-key");
      var show = input.type === "password";
      input.type = show ? "text" : "password";
      $("toggle-key").textContent = show ? "隐藏" : "显示";
    });

    $("wz-reset-advanced").addEventListener("click", function () {
      if (state.defaults) { fillAdvanced(state.defaults); }
    });

    $("wz-verify").addEventListener("click", verifyDraft);
    $("wz-save").addEventListener("click", saveFromWizard);
    $("wz-recheck").addEventListener("click", recheckLocation);
    $("wz-allow-project").addEventListener("change", function () {
      updateSaveState();
      renderLocationWarning();
    });
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
  function requireConfigured() {
    if (state.configured) { return true; }
    showError("还没有配置 API Key，请先在「API 配置」里完成配置。");
    openWizard(1);
    return false;
  }

  async function recognizeUpload() {
    var input = $("file-input");
    if (!input.files || !input.files.length) { return; }
    if (!requireConfigured()) { return; }

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
    if (!requireConfigured()) { return; }

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

  async function init() {
    bindTabs();
    bindFilePicker();
    bindWizard();
    $("btn-recognize-path").addEventListener("click", recognizePath);
    $("path-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter") { recognizePath(); }
    });

    var config = await loadConfig();
    if (config) {
      fillIdentity(config);
      fillAdvanced(config);
      // 首次使用：直接打开配置向导
      if (!state.configured) { openWizard(1); }
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
