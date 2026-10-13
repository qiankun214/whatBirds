/**
 * WhatBird 前端：只与 REST API 通信，不含任何识别逻辑（前后端分离）。
 * 全部原生 JS，不依赖任何外部 CDN。
 *
 * 三块功能：
 *   1) 配置向导（4 步）：获取 Key -> 接入信息 -> 高级参数 -> 测试并保存
 *   2) 识别：单张与批量走同一套「任务 + 轮询」流程，因此都能显示进度并随时终止
 *   3) 结果：单张显示完整详情；多张显示结果表 + 点击行查看详情
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
  var POLL_INTERVAL_MS = 700;

  var state = {
    // 配置
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
    saved: false,
    // 识别
    files: [],
    previewUrl: null,
    job: null,
    lastBatch: null,
    selectedIndex: null
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
      var message = "请求失败（HTTP " + response.status + "）";
      if (detail && typeof detail === "object") {
        message = detail.message + (detail.hint ? "\n提示：" + detail.hint : "");
      } else if (typeof detail === "string") {
        message = detail;
      }
      throw new Error(message);
    }
    return payload;
  }

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

  // ------------------------------------------------------------ 进度面板
  function beginProgress(total, title) {
    var job = {
      id: null,
      total: total || 0,
      timer: null,
      elapsedTimer: null,
      startedAt: Date.now()
    };
    state.job = job;

    $("loading-title").textContent = title;
    $("loading-bar").style.width = "2%";
    $("loading-progress").textContent = total > 1
      ? "正在准备，共 " + total + " 张…"
      : "正在准备…";
    $("loading-stats").textContent = "成功 0 · 失败 0";
    $("loading-elapsed").textContent = "0.0";
    $("loading-status").hidden = true;
    $("btn-stop").disabled = false;
    $("btn-stop").textContent = "终止识别";
    $("loading").hidden = false;

    job.elapsedTimer = setInterval(function () {
      if (state.job === job) {
        $("loading-elapsed").textContent = ((Date.now() - job.startedAt) / 1000).toFixed(1);
      }
    }, 200);
  }

  function updateProgress(snapshot) {
    var total = snapshot.total || 0;
    var done = snapshot.completed || 0;
    var percent = total ? Math.round((done / total) * 100) : 0;
    $("loading-bar").style.width = Math.max(2, percent) + "%";

    var currentIndex = snapshot.current_index || 0;
    var currentName = snapshot.current_source || "";
    var active = snapshot.active || [];

    if (active.length > 1) {
      // 并发时同时有好几张在跑，如实列出，而不是只显示其中一张
      var names = active.map(function (item) { return item.source; }).join("、");
      $("loading-progress").textContent =
        "已完成 " + done + " / " + total + " 张 · 进行中：" + names;
    } else if (currentIndex >= 1 && currentIndex <= total) {
      $("loading-progress").textContent = "正在识别第 " + currentIndex + " / " + total + " 张"
        + (currentName ? "：" + currentName : "");
    } else {
      $("loading-progress").textContent = "已完成 " + done + " / " + total + " 张";
    }
    $("loading-stats").textContent = "成功 " + (snapshot.succeeded || 0) + " · 失败 " + (snapshot.failed || 0);
    if (snapshot.elapsed_seconds != null) {
      $("loading-elapsed").textContent = snapshot.elapsed_seconds.toFixed(1);
    }
  }

  function endProgress() {
    var job = state.job;
    if (job) {
      clearTimeout(job.timer);
      clearInterval(job.elapsedTimer);
      state.job = null;
    }
    $("loading").hidden = true;
  }

  async function stopBatch() {
    if (!state.job || !state.job.id) { return; }
    $("btn-stop").disabled = true;
    $("btn-stop").textContent = "正在终止…";
    $("loading-status").hidden = false;
    $("loading-status").textContent = "已发送终止请求，正在等待当前请求中断…";
    try {
      await api("/api/batch/" + state.job.id + "/stop", { method: "POST" });
    } catch (err) {
      $("loading-status").textContent = "终止请求失败：" + err.message;
      $("btn-stop").disabled = false;
      $("btn-stop").textContent = "终止识别";
    }
  }

  // ------------------------------------------------------------ 批量任务
  async function startBatch(url, options, expectedTotal, title) {
    showError("");
    beginProgress(expectedTotal, title);
    try {
      var data = await api(url, options);
      if (!state.job) { return; }  // 期间被取消
      state.job.id = data.job_id;
      updateProgress({
        total: data.total, completed: 0, succeeded: 0, failed: 0, current_index: 0
      });
      pollBatch();
    } catch (err) {
      endProgress();
      showError(err.message);
    }
  }

  async function pollBatch() {
    var job = state.job;
    if (!job || !job.id) { return; }

    var snapshot;
    try {
      snapshot = await api("/api/batch/" + job.id);
    } catch (err) {
      if (state.job === job) { finishBatch(null, err.message); }
      return;
    }
    if (state.job !== job) { return; }  // 已经开始了新任务

    updateProgress(snapshot);
    if (snapshot.status === "done" || snapshot.status === "stopped" || snapshot.status === "failed") {
      finishBatch(snapshot, null);
      return;
    }
    job.timer = setTimeout(pollBatch, POLL_INTERVAL_MS);
  }

  function finishBatch(snapshot, errorMessage) {
    endProgress();

    if (snapshot) {
      renderResults(snapshot);
      if (snapshot.status === "stopped") {
        showError("已终止识别：完成 " + snapshot.completed + " / " + snapshot.total + " 张。");
      } else if (snapshot.status === "failed") {
        var detail = snapshot.error || {};
        showError("识别失败：" + (detail.message || "未知错误") + (detail.hint ? "\n提示：" + detail.hint : ""));
      } else {
        showError("");
      }
    } else if (errorMessage) {
      showError("无法获取识别进度：" + errorMessage);
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
    $("detail-error").hidden = true;
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
    if (meta.mode === "fast") { parts.push("快速模式"); }
    if (extraMeta) { parts.push(extraMeta); }
    if (typeof meta.elapsed_seconds === "number") { parts.push("耗时 " + meta.elapsed_seconds.toFixed(2) + "s"); }
    if (meta.usage && meta.usage.total_tokens) {
      parts.push("token 输入 " + meta.usage.prompt_tokens + " / 输出 " + meta.usage.completion_tokens);
    }
    $("result-meta").textContent = parts.join(" ｜ ");
  }

  function renderDetailError(item) {
    $("result-body").hidden = true;
    $("result-empty").hidden = true;
    var detail = item.error || {};
    var box = $("detail-error");
    box.hidden = false;
    box.textContent = "这张图片未能识别：" + (detail.message || "未知错误")
      + (detail.hint ? "\n提示：" + detail.hint : "");
    $("result-meta").textContent = item.source || "";
  }

  function renderResults(snapshot) {
    state.lastBatch = snapshot;
    var items = snapshot.items || [];

    if (snapshot.total <= 1) {
      $("batch-panel").hidden = true;
      if (!items.length) { return; }
      var only = items[0];
      if (only.status === "ok" && only.result) { renderResult(only.result, only.source); }
      else { renderDetailError(only); }
      return;
    }

    $("batch-panel").hidden = false;
    renderBatchTable(snapshot);
    $("batch-summary-text").textContent =
      "共 " + snapshot.total + " 张：成功 " + snapshot.succeeded + " · 失败 " + snapshot.failed
      + (snapshot.mode === "fast" ? " · 快速模式" : "")
      + (snapshot.concurrency > 1 ? " · 并发 " + snapshot.concurrency : "")
      + (snapshot.stopped ? " · 已终止" : "")
      + " · 用时 " + (snapshot.elapsed_seconds || 0).toFixed(1) + "s";

    var firstOk = null;
    for (var i = 0; i < items.length; i++) {
      if (items[i].status === "ok") { firstOk = items[i]; break; }
    }
    var target = firstOk || items[0];
    if (target) { selectBatchRow(target.index); }
    else {
      $("result-body").hidden = true;
      $("detail-error").hidden = true;
      $("result-empty").hidden = false;
    }
  }

  function renderBatchTable(snapshot) {
    var tbody = $("batch-rows");
    tbody.innerHTML = "";

    (snapshot.items || []).forEach(function (item) {
      var result = item.result || {};
      var names = result.names || {};
      var taxonomy = result.taxonomy || {};
      var tr = document.createElement("tr");
      tr.dataset.index = String(item.index);
      if (item.status !== "ok") { tr.className = "row-error"; }

      appendCell(tr, String(item.index), "col-index");
      appendCell(tr, item.source || "—", "col-source");
      appendCell(tr, item.status === "ok"
        ? (names.chinese || names.scientific || "未识别")
        : (item.error && item.error.message) || "失败", "col-name");
      appendCell(tr, item.status === "ok" ? (taxonomy.family || "—") : "—");
      appendCell(tr, item.status === "ok" ? (taxonomy.genus || "—") : "—");
      appendCell(tr, item.status === "ok"
        ? Math.round((result.confidence || 0) * 100) + "%"
        : "—", "col-confidence");
      appendCell(tr, item.status === "ok" ? "成功" : "失败", "col-status");

      tr.addEventListener("click", function () { selectBatchRow(item.index); });
      tbody.appendChild(tr);
    });
  }

  function appendCell(row, text, className) {
    var td = document.createElement("td");
    td.textContent = text;
    if (className) { td.className = className; }
    row.appendChild(td);
  }

  function selectBatchRow(index) {
    var snapshot = state.lastBatch;
    if (!snapshot) { return; }
    var item = null;
    (snapshot.items || []).forEach(function (candidate) {
      if (candidate.index === index) { item = candidate; }
    });
    if (!item) { return; }

    state.selectedIndex = index;
    Array.prototype.forEach.call($("batch-rows").children, function (row) {
      row.classList.toggle("selected", Number(row.dataset.index) === index);
    });

    if (item.status === "ok" && item.result) { renderResult(item.result, item.source); }
    else { renderDetailError(item); }
  }

  function exportBatchJson() {
    if (!state.lastBatch) { return; }
    var blob = new Blob([JSON.stringify(state.lastBatch, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = "whatbird-batch-" + new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19) + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
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
    setIfPresent("cfg-concurrency", config.concurrency);
    $("cfg-fast-mode").checked = !!config.fast_mode;
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
      concurrency: numberOrNull("cfg-concurrency", true),
      fast_mode: $("cfg-fast-mode").checked,
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
      ["识别模式", draft.fast_mode ? "快速模式（只要名称 + 分类学）" : "完整模式"],
      ["批量并发数", draft.concurrency !== null ? draft.concurrency : "（默认）"],
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

  function setBusy(busy) {
    state.busy = busy;
    ["wz-verify", "wz-recheck", "wz-next", "wz-prev"].forEach(function (id) {
      $(id).disabled = busy;
    });
    updateSaveState();
  }

  function updateSaveState() {
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
      renderConsent();
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
      renderConsent();
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

  // ------------------------------------------------------------ 选择图片
  function showFiles(fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    state.files = files;

    var button = $("btn-recognize-upload");
    var list = $("file-list");

    if (!files.length) {
      $("preview-wrap").hidden = true;
      list.hidden = true;
      list.innerHTML = "";
      button.disabled = true;
      button.textContent = "开始识别";
      return;
    }

    var invalid = files.filter(function (file) { return !/\.jpe?g$/i.test(file.name); });
    if (invalid.length) {
      showError("这些文件不是 .jpg / .jpeg：" + invalid.map(function (f) { return f.name; }).join("、"));
    } else {
      showError("");
    }

    button.disabled = invalid.length > 0;
    button.textContent = files.length > 1 ? "开始识别（" + files.length + " 张）" : "开始识别";

    if (state.previewUrl) { URL.revokeObjectURL(state.previewUrl); }
    state.previewUrl = URL.createObjectURL(files[0]);
    $("preview").src = state.previewUrl;
    $("preview-wrap").hidden = false;
    $("preview-name").textContent = files.length > 1
      ? files[0].name + " 等 " + files.length + " 个文件"
      : files[0].name;
    var totalKb = files.reduce(function (sum, file) { return sum + file.size; }, 0) / 1024;
    $("preview-size").textContent = Math.round(totalKb) + " KB";

    list.innerHTML = "";
    files.forEach(function (file, index) {
      var li = document.createElement("li");
      li.textContent = (index + 1) + ". " + file.name + "（" + Math.round(file.size / 1024) + " KB）";
      list.appendChild(li);
    });
    list.hidden = files.length <= 1;
    setFocusBusy(false);
    setDetectBusy(false);
  }

  function parsePaths() {
    return $("path-input").value
      .split(/\r?\n/)
      .map(function (line) { return line.trim(); })
      .filter(function (line) { return line.length > 0; });
  }

  // ------------------------------------------------------------ 识别入口
  function requireConfigured() {
    if (state.configured) { return true; }
    showError("还没有配置 API Key，请先在「API 配置」里完成配置。");
    openWizard(1);
    return false;
  }

  async function recognizeUpload() {
    if (!requireConfigured()) { return; }
    var files = state.files || [];
    if (!files.length) { showError("请先选择至少一张 jpg 图片。"); return; }

    var form = new FormData();
    files.forEach(function (file) { form.append("files", file); });

    await startBatch(
      "/api/batch/start-upload",
      { method: "POST", body: form },
      files.length,
      files.length > 1 ? "正在依次识别 " + files.length + " 张图片…" : "正在调用 Doubao-Seed-2.1 识别…"
    );
  }

  async function recognizePath() {
    if (!requireConfigured()) { return; }
    var paths = parsePaths();
    if (!paths.length) { showError("请至少填写一个 jpg 文件路径。"); return; }

    await startBatch(
      "/api/batch/start-paths",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: paths })
      },
      paths.length,
      paths.length > 1 ? "正在依次识别 " + paths.length + " 张图片…" : "正在调用 Doubao-Seed-2.1 识别…"
    );
  }

  // ------------------------------------------------------------ 对焦点分析
  // 纯本地算法：不调用方舟、不消耗额度，所以这里刻意**不做** requireConfigured 检查。
  function showFocusError(message) {
    setMessage($("focus-error"), message || "", "error");
  }

  function percent(value) {
    var number = typeof value === "number" && isFinite(value) ? value : 0;
    return Math.round(number * 100) + "%";
  }

  function setFocusBusy(busy) {
    var upload = $("btn-focus-upload");
    var byPath = $("btn-focus-path");
    upload.disabled = !!busy || !(state.files || []).length;
    byPath.disabled = !!busy;
    upload.textContent = busy ? "分析中…" : "分析已选图片的第一张";
  }

  function focusRouteRow(tbody, route, hit) {
    var isExif = route === "exif";
    var tr = document.createElement("tr");

    var nameCell = document.createElement("td");
    nameCell.className = isExif ? "route-exif" : "route-content";
    nameCell.textContent = isExif ? "EXIF 元数据" : "内容分析";
    tr.appendChild(nameCell);

    if (!hit.found) {
      var missing = document.createElement("td");
      missing.className = "missing";
      missing.colSpan = 3;
      missing.textContent = "没有结论：" + (hit.detail || "—");
      tr.appendChild(missing);
      tbody.appendChild(tr);
      return;
    }

    var pointCell = document.createElement("td");
    pointCell.textContent = "(" + hit.x + ", " + hit.y + ")";
    var detail = document.createElement("span");
    detail.className = "route-detail";
    detail.textContent = hit.detail || "";
    pointCell.appendChild(detail);
    tr.appendChild(pointCell);

    appendCell(tr, percent(hit.x_ratio) + " / " + percent(hit.y_ratio));
    appendCell(tr, percent(hit.confidence));
    tbody.appendChild(tr);
  }

  function renderFocus(payload) {
    var result = payload.result || {};
    var exif = result.exif || {};
    var content = result.content || {};
    var image = result.image || {};

    $("focus-empty").hidden = true;
    $("focus-body").hidden = false;

    var rows = $("focus-rows");
    rows.innerHTML = "";
    focusRouteRow(rows, "exif", exif);
    focusRouteRow(rows, "content", content);

    var verdict = $("focus-verdict");
    if (exif.found && content.found) {
      if (result.agree) {
        verdict.className = "focus-verdict";
        verdict.textContent = "两条路线一致：相机记录的对焦点与画面实际最清晰处互相印证。";
      } else {
        verdict.className = "focus-verdict is-disagree";
        verdict.textContent = "两条路线不一致：相机声称的对焦点与画面实际最清晰处不同，值得看一眼。";
      }
    } else if (exif.found || content.found) {
      verdict.className = "focus-verdict is-disagree";
      verdict.textContent = "只有一条路线给出了结论（另一条的原因见上表）。";
    } else {
      verdict.className = "focus-verdict is-none";
      verdict.textContent = "两条路线都没能定位对焦点。";
    }

    var region = result.region;
    $("focus-region").textContent = region
      ? "合焦区域：x " + region.x + "、y " + region.y + "，" + region.width + "×" + region.height +
        "，占画面 " + percent(region.area_ratio)
      : "";

    var notes = result.notes || [];
    $("focus-notes-wrap").hidden = notes.length === 0;
    renderList($("focus-notes"), notes, function (li, note) { li.textContent = note; });

    var others = (result.candidates || []).slice(1);
    $("focus-candidates-wrap").hidden = others.length === 0;
    renderList($("focus-candidates"), others, function (li, item) {
      li.textContent = "(" + item.x + ", " + item.y + ")　相对锐度 " + percent(item.score);
    });

    var elapsed = typeof result.elapsed_seconds === "number" ? result.elapsed_seconds : 0;
    $("focus-meta").textContent =
      image.width + "×" + image.height +
      " ｜ 分析尺度 " + image.analyzed_width + "×" + image.analyzed_height +
      " ｜ " + elapsed.toFixed(2) + "s";
    $("focus-overlay").src = payload.overlay || "";
  }

  async function runFocus(url, options) {
    showFocusError("");
    setFocusBusy(true);
    try {
      renderFocus(await api(url, options));
    } catch (err) {
      showFocusError(err.message || String(err));
    } finally {
      setFocusBusy(false);
    }
  }

  async function analyzeFocusUpload() {
    var files = state.files || [];
    if (!files.length) { showFocusError("请先在上方选择至少一张 jpg 图片。"); return; }
    var form = new FormData();
    form.append("file", files[0]);
    await runFocus("/api/focus", { method: "POST", body: form });
  }

  async function analyzeFocusPath() {
    var paths = parsePaths();
    if (!paths.length) { showFocusError("请先在上方填写一个 jpg 文件路径。"); return; }
    await runFocus("/api/focus-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: paths[0] })
    });
  }

  // ------------------------------------------------------------ 鸟类位置检测
  // 与对焦点分析同样是纯本地能力：不调用方舟、不消耗额度，所以也不做 requireConfigured 检查。
  function showDetectError(message) {
    setMessage($("detect-error"), message || "", "error");
  }

  function setDetectBusy(busy) {
    var upload = $("btn-detect-upload");
    var byPath = $("btn-detect-path");
    upload.disabled = !!busy || !(state.files || []).length;
    byPath.disabled = !!busy;
    upload.textContent = busy ? "检测中…" : "检测已选图片的第一张";
  }

  function renderDetect(payload) {
    var result = payload.result || {};
    var image = result.image || {};
    var items = result.birds || [];

    $("detect-empty").hidden = true;
    $("detect-body").hidden = false;

    var rows = $("detect-rows");
    rows.innerHTML = "";
    items.forEach(function (item, index) {
      var name = item.class_name && item.class_name !== "bird" ? " " + item.class_name : "";
      var tr = document.createElement("tr");
      appendCell(tr, "#" + (index + 1) + name);
      appendCell(tr, "x=" + item.x + " y=" + item.y + " w=" + item.width + " h=" + item.height);
      appendCell(tr, percent(item.x_ratio) + " / " + percent(item.y_ratio));
      appendCell(tr, percent(item.confidence));
      rows.appendChild(tr);
    });

    var verdict = $("detect-verdict");
    if (items.length) {
      verdict.className = "focus-verdict";
      verdict.textContent = "检测到 " + items.length + " 个目标"
        + (result.found ? "（其中有鸟）" : "") + "。框已画在下图的原分辨率图上。";
    } else {
      verdict.className = "focus-verdict is-none";
      verdict.textContent = "没有检测到鸟：可能画面里确实没有鸟，也可能是鸟太小、太远或被遮挡。";
    }

    var others = result.other_classes || [];
    $("detect-other-wrap").hidden = others.length === 0;
    $("detect-other").textContent = others.length
      ? "按参数过滤、不计入结果的目标类别：" + others.join("、")
      : "";

    var notes = result.notes || [];
    $("detect-notes-wrap").hidden = notes.length === 0;
    renderList($("detect-notes"), notes, function (li, note) { li.textContent = note; });

    var elapsed = typeof result.elapsed_seconds === "number" ? result.elapsed_seconds : 0;
    $("detect-meta").textContent =
      image.width + "×" + image.height +
      " ｜ 设备 " + (result.device || "-") +
      " ｜ " + (result.model_name || "-") +
      " ｜ " + elapsed.toFixed(2) + "s";
    $("detect-overlay").src = payload.overlay || "";
  }

  async function runDetect(url, options) {
    showDetectError("");
    setDetectBusy(true);
    try {
      renderDetect(await api(url, options));
    } catch (err) {
      showDetectError(err.message || String(err));
    } finally {
      setDetectBusy(false);
    }
  }

  async function detectUpload() {
    var files = state.files || [];
    if (!files.length) { showDetectError("请先在上方选择至少一张 jpg 图片。"); return; }
    var form = new FormData();
    form.append("file", files[0]);
    await runDetect("/api/detect", { method: "POST", body: form });
  }

  async function detectPath() {
    var paths = parsePaths();
    if (!paths.length) { showDetectError("请先在上方填写一个 jpg 文件路径。"); return; }
    await runDetect("/api/detect-path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: paths[0] })
    });
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

    input.addEventListener("change", function () { showFiles(input.files); });

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
        Array.prototype.forEach.call(files, function (file) { transfer.items.add(file); });
        input.files = transfer.files;
        showFiles(files);
      }
    });

    $("btn-recognize-upload").addEventListener("click", recognizeUpload);
  }

  async function init() {
    bindTabs();
    bindFilePicker();
    bindWizard();

    $("btn-recognize-path").addEventListener("click", recognizePath);
    $("btn-stop").addEventListener("click", stopBatch);
    $("btn-export").addEventListener("click", exportBatchJson);
    $("btn-focus-upload").addEventListener("click", analyzeFocusUpload);
    $("btn-focus-path").addEventListener("click", analyzeFocusPath);
    $("btn-detect-upload").addEventListener("click", detectUpload);
    $("btn-detect-path").addEventListener("click", detectPath);
    $("path-input").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { recognizePath(); }
    });

    setFocusBusy(false);
    setDetectBusy(false);

    var config = await loadConfig();
    if (config) {
      fillIdentity(config);
      fillAdvanced(config);
      if (!state.configured) { openWizard(1); }
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
