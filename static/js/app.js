(() => {
  const pollIntervalMs = Math.max((window.APP_CONFIG?.pollInterval ?? 1) * 1000, 800);

  const PRESETS = {
    fast: {
      cell_size: 64,
      top_candidates: 16,
      max_repeated_tiles: 4,
      repetition_penalty: 0.18,
      neighbor_radius: 0,
      blend_ratio: 0.18,
      enable_enhancement: false,
      distance_metric: "hybrid",
    },
    balanced: {
      cell_size: 40,
      top_candidates: 48,
      max_repeated_tiles: 6,
      repetition_penalty: 0.3,
      neighbor_radius: 1,
      blend_ratio: 0.26,
      enable_enhancement: true,
      distance_metric: "hybrid",
    },
    high: {
      cell_size: 24,
      top_candidates: 96,
      max_repeated_tiles: 3,
      repetition_penalty: 0.42,
      neighbor_radius: 2,
      blend_ratio: 0.32,
      enable_enhancement: true,
      distance_metric: "hybrid",
    },
    unam: {
      cell_size: 32,
      top_candidates: 72,
      max_repeated_tiles: 4,
      repetition_penalty: 0.38,
      neighbor_radius: 2,
      blend_ratio: 0.3,
      enable_enhancement: true,
      distance_metric: "riemersma",
    },
  };

  const metricDescriptions = {
    "Celdas procesadas": "Total de bloques evaluados.",
    "Teselas unicas": "Teselas diferentes usadas en el mosaico.",
    "Repeticion promedio": "Veces promedio que se repite una tesela.",
    "Entropia mosaico": "Diversidad cromatica global obtenida.",
    "Blend configurado": "Mezcla indicada por el filtro Blend.",
    "Realce tonal": "Ajuste de contraste y nitidez al blend final.",
    "Tiempo total": "Duración completa de la ejecución.",
    "Seleccion de teselas": "Tiempo dedicado a elegir candidatos por celda.",
    "Artefactos": "Tiempo en blending y escala de grises.",
  };

  const fallbackStatusMessage = {
    idle: "Sube una imagen para comenzar.",
    pending: "Esperando a que inicie la generacion...",
    running: "Generando fotomosaico...",
    completed: "Mosaico finalizado.",
    failed: "El proceso reporto un error.",
  };

  const MAX_PROCESS_EVENTS = 10;

  const form = document.getElementById("mosaic-form");
  const fileInput = document.getElementById("source-image");
  const submitBtn = document.getElementById("submit-btn");
  const statusPill = document.querySelector(".status-pill");
  const progressBar = document.querySelector(".progress-bar");
  const statusMsg = document.querySelector(".status-message");
  const statusMeta = document.querySelector(".status-meta");
  const statusMetaElapsed = document.querySelector(".status-meta .elapsed");
  const statusMetaEta = document.querySelector(".status-meta .eta");
  const statusEvents = document.querySelector(".status-events");
  const statusEventsList = document.querySelector(".status-events-list");
  const resultImg = document.getElementById("result-image");
  const resultType = document.getElementById("result-type");
  const resultEmpty = document.querySelector(".result-empty");
  const downloadResult = document.getElementById("download-result");
  const metricsTableBody = document.querySelector(".metrics-body");
  const chartStatus = document.getElementById("chart-status");
  const progressCanvas = document.getElementById("progress-chart");
  const historyList = document.querySelector(".history-list");
  const historyEmpty = document.querySelector(".history-empty");
  const blendSlider = document.getElementById("blend-ratio");
  const enhancementToggle = document.getElementById("enable-enhancement");
  const metricSelect = document.getElementById("distance-metric");
  const sourcePreviewWrapper = document.getElementById("source-preview-wrapper");
  const grayscalePreviewToggle = document.getElementById("grayscale-preview-toggle");
  const sourcePreview = document.getElementById("source-preview");
  const presetButtons = document.querySelectorAll(".preset-button");
  const livePreviewImg = document.getElementById("live-preview-image");
  const livePreviewEmpty = document.getElementById("live-preview-empty");
  const resultFiltersWrapper = document.querySelector(".result-filters");
  const resultFilterOptions = document.getElementById("result-filter-options");
  const grayscalePreviewCanvas = document.getElementById("grayscale-preview-canvas");

  const sliderBindings = [
    ["cell-size", "cell-size-value", (value) => `${value}px`],
    ["top-candidates", "top-candidates-value", (value) => value],
    ["max-repeats", "max-repeats-value", (value) => (value === "0" ? "Sin limite" : value)],
    ["repetition-penalty", "penalty-value", (value) => Number(value).toFixed(2)],
    ["neighbor-radius", "radius-value", (value) => value],
    ["blend-ratio", "blend-value", (value) => `${Math.round(Number(value) * 100)}%`],
  ];

  let activeJobId = null;
  let pollHandle = null;
  let progressEventLog = [];
  let lastProgressSnapshot = null;
  let previewUrlRef = null;
  let latestResultPaths = {};
  let latestParameters = null;
  let availableResultFilters = [];
  let activeResultFilter = null;
  let lastSummaryJobId = null;
  let activeHistoryId = null;
  const progressSeries = [];
  let jobTimings = { startedAt: null, etaSeconds: null, totalSeconds: null, elapsedSeconds: null };
  let sampleFilePromise = null;

  document.addEventListener("DOMContentLoaded", () => {
    bindSliders();
    setupPresetButtons();
    setupForm();
    ensureSampleFile()
      .then(setFileInputFromFile)
      .catch((error) => console.warn("No se pudo precargar la muestra:", error));
    fetchDefaults()
      .catch(() => {})
      .finally(() => applyPreset("balanced", { loadSample: false }));
    fetchHistory();
  });
  if (grayscalePreviewToggle) {
    grayscalePreviewToggle.addEventListener('change', () => {
      if (fileInput?.files?.length) {
        renderGrayscalePreview(fileInput.files[0]);
      } else if (grayscalePreviewCanvas) {
        grayscalePreviewCanvas.hidden = true;
      }
    });
  }


  function bindSliders() {
    sliderBindings.forEach(([sliderId, outputId, formatter]) => {
      const slider = document.getElementById(sliderId);
      const output = document.getElementById(outputId);
      if (!slider || !output) return;
      const update = () => {
        output.textContent = formatter(slider.value);
      };
      slider.addEventListener("input", update);
      update();
    });
  }

  function setupPresetButtons() {
    presetButtons.forEach((button) => {
      button.addEventListener("click", () => applyPreset(button.dataset.preset));
    });
  }

  function setupForm() {
    if (!form) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!fileInput?.files?.length) {
        window.alert("Selecciona una imagen antes de continuar.");
        return;
      }

      resetStatusEvents();
      submitBtn.disabled = true;
      setStatus("pending", "Preparando solicitud...");
      resetStatusEvents();
      resetProgressChart();
      updateProgress(
        {
          percent: 0,
          message: "Cargando imagen...",
          completed: 0,
          total: null,
        },
        "pending",
      );
      resetLivePreview();

      try {
        const payload = buildPayload();
        const response = await fetch("/api/mosaics", {
          method: "POST",
          body: payload,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || "No se pudo crear el trabajo.");
        }

        const job = await response.json();
        activeJobId = job.id;
        activeHistoryId = job.id;
        setStatus("pending", "Solicitud aceptada. Encolando trabajo...");
        beginPolling();
      } catch (error) {
        console.error(error);
        setStatus("failed", error.message || "Error inesperado.");
        submitBtn.disabled = false;
      }
    });

    if (fileInput) {
      fileInput.addEventListener("change", handleFilePreview);
    }
  }

  function buildPayload() {
    const params = {
      parameters: {
        cell_size: Number(document.getElementById("cell-size").value),
        top_candidates: Number(document.getElementById("top-candidates").value),
        max_repeated_tiles: Number(document.getElementById("max-repeats").value),
        repetition_penalty: Number(document.getElementById("repetition-penalty").value),
        neighbor_radius: Number(document.getElementById("neighbor-radius").value),
        blend_ratio: Number(document.getElementById("blend-ratio").value),
        enable_enhancement: enhancementToggle ? enhancementToggle.checked : true,
        distance_metric: metricSelect?.value || "hybrid",
      },
    };

    const data = new FormData();
    data.append("request_payload", JSON.stringify(params));
    data.append("source_image", fileInput.files[0]);
    return data;
  }

  function beginPolling() {
    jobTimings.startedAt = Date.now();
    jobTimings.elapsedSeconds = null;
    jobTimings.etaSeconds = null;
    jobTimings.totalSeconds = null;
    resetProgressChart();
    updateStatusMeta();
    if (!activeJobId) return;
    clearInterval(pollHandle);
    pollHandle = window.setInterval(async () => {
      try {
        const response = await fetch(`/api/mosaics/${activeJobId}`);
        if (!response.ok) {
          throw new Error(`No se encontro el trabajo ${activeJobId}.`);
        }
        const job = await response.json();
        renderJob(job);
        updateTimings(job);
        if (job.status === "completed" || job.status === "failed") {
          clearInterval(pollHandle);
          submitBtn.disabled = false;
          fetchHistory();
        }
      } catch (error) {
        console.error(error);
        clearInterval(pollHandle);
        submitBtn.disabled = false;
        setStatus("failed", error.message || "Error consultando el estado.");
      }
    }, pollIntervalMs);
  }

  function renderJob(job) {
    const message = job.message || fallbackStatusMessage[job.status] || "Actualizando estado...";
    setStatus(job.status, message);
    if (job.id && !activeHistoryId) {
      activeHistoryId = job.id;
    }
    if (job.id) {
      highlightHistorySelection(job.id);
    }
    updateTimings(job);
    updateStatusMeta();
    if (job.progress) {
      updateProgress(job.progress, job.status);
    }

    if (job.preview_url) {
      updateLivePreview(job.preview_url);
    }

    renderResults(job.result || {}, job.parameters);
    renderMetrics(job.metrics, job.parameters);
    surfaceRunSummary(job);
  }

  function renderResults(result, parameters) {
    latestResultPaths = result || {};
    latestParameters = parameters || null;

    const hasMosaic = Boolean(latestResultPaths.mosaic);
    const hasBlend = Boolean(latestResultPaths.blended);
    const hasGrayscale = Boolean(latestResultPaths.grayscale);

    availableResultFilters = [];
  if (hasMosaic) {
    availableResultFilters.push({
      id: "mosaic",
      label: hasBlend ? "Mosaico base" : "Mosaico",
      url: latestResultPaths.mosaic,
      downloadName: "mosaico.jpg",
      downloadLabel: "Descargar mosaico",
    });
  }
  if (hasBlend) {
    availableResultFilters.push({
      id: "blended",
      label: "Blend con original",
      url: latestResultPaths.blended,
      downloadName: "mosaico_mezcla.jpg",
      downloadLabel: "Descargar mezcla",
    });
  }
  if (hasGrayscale) {
    availableResultFilters.push({
      id: "grayscale",
      label: "Mosaico en escala de grises",
      url: latestResultPaths.grayscale,
      downloadName: "mosaico_grises.jpg",
      downloadLabel: "Descargar versión gris",
    });
  }

    if (!availableResultFilters.length) {
      activeResultFilter = null;
      if (resultImg) resultImg.hidden = true;
      if (resultEmpty) resultEmpty.hidden = false;
      if (downloadResult) downloadResult.hidden = true;
      if (resultFiltersWrapper) resultFiltersWrapper.hidden = true;
      if (resultFilterOptions) resultFilterOptions.innerHTML = "";
      resultType.textContent = "Sin generar";
      return;
    }

    if (!availableResultFilters.some((filter) => filter.id === activeResultFilter)) {
      activeResultFilter = availableResultFilters[0]?.id ?? null;
    }

    renderResultFilterButtons();
    updateResultView();
    if (resultEmpty) resultEmpty.hidden = true;
  }

  function surfaceRunSummary(job) {
    if (!job || job.status !== "completed") return;
    const summary = job.metrics?.run_summary;
    if (!summary || lastSummaryJobId === job.id) return;
    const summaryLine = summary.split?.("\n")?.[0] || summary;
    if (!summaryLine || !summaryLine.trim()) {
      lastSummaryJobId = job.id;
      return;
    }
    logProcessingEvent({
      message: `${summaryLine}`,
      percent: 100,
      completed: job.progress?.completed ?? null,
      total: job.progress?.total ?? null,
    });
    lastSummaryJobId = job.id;
  }

  function renderResultFilterButtons() {
    if (!resultFilterOptions || !resultFiltersWrapper) return;
    resultFilterOptions.innerHTML = "";
    availableResultFilters.forEach((filter) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "result-filter-button";
      button.dataset.filter = filter.id;
      button.textContent = filter.label;
      button.setAttribute("aria-pressed", filter.id === activeResultFilter ? "true" : "false");
      button.addEventListener("click", () => {
        if (activeResultFilter === filter.id) return;
        activeResultFilter = filter.id;
        updateResultView();
      });
      resultFilterOptions.appendChild(button);
    });
    resultFiltersWrapper.hidden = availableResultFilters.length === 0;
  }

  function updateFilterButtons() {
    if (!resultFilterOptions) return;
    const buttons = resultFilterOptions.querySelectorAll(".result-filter-button");
    buttons.forEach((button) => {
      const { filter } = button.dataset;
      button.setAttribute("aria-pressed", filter === activeResultFilter ? "true" : "false");
    });
  }

  function updateResultView() {
    if (!resultImg) return;
    if (!availableResultFilters.length || !activeResultFilter) {
      resultImg.hidden = true;
      if (downloadResult) downloadResult.hidden = true;
      resultType.textContent = "Sin generar";
      return;
    }

    const selected =
      availableResultFilters.find((filter) => filter.id === activeResultFilter) ||
      availableResultFilters[0];
    if (!selected) {
      resultImg.hidden = true;
      if (downloadResult) downloadResult.hidden = true;
      resultType.textContent = "Sin generar";
      return;
    }

    activeResultFilter = selected.id;
    resultImg.src = appendCacheBuster(selected.url);
    resultImg.hidden = false;

    if (downloadResult) {
      downloadResult.href = selected.url;
      downloadResult.hidden = false;
      downloadResult.textContent = selected.downloadLabel;
      downloadResult.download = selected.downloadName;
    }

    resultType.textContent = selected.label;
    updateFilterButtons();
  }

  function renderMetrics(metrics, parameters) {
    if (!metricsTableBody) return;
    metricsTableBody.innerHTML = "";
    if (!metrics) {
      metricsTableBody.insertAdjacentHTML(
        "beforeend",
        `<tr><td colspan="3">Metricas pendientes</td></tr>`
      );
      return;
    }

    const rows = [
      ["Celdas procesadas", metrics.total_cells],
      ["Teselas unicas", metrics.unique_images],
      ["Repeticion promedio", safeFixed(metrics.repetition_index)],
      ["Entropia mosaico", safeFixed(metrics.entropy)],
    ];

    if (parameters) {
      const ratioPercent = Math.round(Number(parameters.blend_ratio ?? 0) * 100);
      rows.push(["Blend configurado", ratioPercent > 0 ? `${ratioPercent}%` : "Sin mezcla"]);
      rows.push(["Realce tonal", parameters.enable_enhancement ? "Activado" : "Desactivado"]);
    }

    if (metrics.timing) {
      const { timing } = metrics;
      rows.push(["Tiempo total", formatSeconds(timing.total_seconds)]);
      if (timing.selection_seconds !== undefined) {
        rows.push(["Seleccion de teselas", formatSeconds(timing.selection_seconds)]);
      }
      if (timing.artifact_seconds !== undefined) {
        rows.push(["Artefactos", formatSeconds(timing.artifact_seconds)]);
      }
    }

    rows.forEach(([label, value]) => {
      if (value === undefined || value === null || value === "") return;
      const note = metricDescriptions[label] ?? "";
      metricsTableBody.insertAdjacentHTML(
        "beforeend",
        `<tr>
            <td>${label}</td>
            <td class="value">${value}</td>
            <td class="note">${note}</td>
         </tr>`
      );
    });
  }

  function fetchDefaults() {
    return fetch("/api/mosaics/defaults")
      .then((response) => response.json())
      .then((defaults) => {
        const mapping = {
          "cell-size": defaults.cell_size,
          "top-candidates": defaults.top_candidates,
          "max-repeats": defaults.max_repeated_tiles,
          "repetition-penalty": defaults.repetition_penalty,
          "neighbor-radius": defaults.neighbor_radius,
          "blend-ratio": defaults.blend_ratio,
        };
        Object.entries(mapping).forEach(([id, value]) => {
          const input = document.getElementById(id);
          if (input) {
            input.value = value;
            input.dispatchEvent(new Event("input"));
          }
        });

        if (blendSlider) {
          blendSlider.value = defaults.blend_ratio;
          blendSlider.dispatchEvent(new Event("input"));
        }
        if (enhancementToggle) {
          enhancementToggle.checked = defaults.enable_enhancement;
        }
      })
      .catch((error) => {
        console.warn("No se pudieron cargar los valores por defecto:", error);
        throw error;
      });
  }

  function fetchHistory() {
    fetch("/api/mosaics")
      .then((response) => response.json())
      .then((jobs) => {
        historyList.innerHTML = "";
        if (!jobs.length) {
          historyEmpty.hidden = false;
          return;
        }

        historyEmpty.hidden = true;
        jobs.slice(0, 8).forEach((job) => {
          const item = document.createElement("li");
          item.className = "history-item";
          item.dataset.jobId = job.id;
          if (job.id === activeHistoryId) {
            item.classList.add("is-active");
          }
          const header = document.createElement("div");
          const idLabel = document.createElement("strong");
          idLabel.textContent = job.id.slice(0, 8);
          const pill = document.createElement("div");
          pill.className = "status-pill";
          pill.dataset.state = job.status;
          pill.textContent = job.status.toUpperCase();
          header.append(idLabel, pill);
          const time = document.createElement("time");
          time.textContent = new Date(job.created_at).toLocaleString();
          item.append(header, time);
          item.addEventListener("click", () => {
            if (activeHistoryId === job.id && !activeJobId) {
              highlightHistorySelection(job.id);
              return;
            }
            activeHistoryId = job.id;
            highlightHistorySelection(job.id);
            loadJobById(job.id);
          });
          historyList.appendChild(item);
        });
        highlightHistorySelection();
      })
      .catch((error) => console.warn("No se pudo recuperar el historial:", error));
  }

  function highlightHistorySelection(selectedId = activeHistoryId) {
    if (!historyList) return;
    historyList.querySelectorAll(".history-item").forEach((item) => {
      item.classList.toggle("is-active", item.dataset.jobId === selectedId);
    });
  }

  function loadJobById(jobId) {
    if (!jobId) return;
    clearInterval(pollHandle);
    resetStatusEvents();
    resetProgressChart();
    fetch(`/api/mosaics/${jobId}`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`No se encontró el trabajo ${jobId}.`);
        }
        return response.json();
      })
      .then((job) => {
        activeHistoryId = job.id;
        highlightHistorySelection(job.id);
        if (job.status === "running" || job.status === "pending") {
          activeJobId = job.id;
        } else {
          activeJobId = null;
        }
        renderJob(job);
        if (job.status === "running" || job.status === "pending") {
          beginPolling();
        }
      })
      .catch((error) => {
        console.error(error);
        setStatus("failed", error.message || "No se pudo cargar el trabajo.");
      });
  }

  function ensureSampleFile() {
    if (!sampleFilePromise) {
      sampleFilePromise = fetch("/static/morsa.jpg")
        .then((response) => {
          if (!response.ok) throw new Error("No se pudo descargar la muestra");
          return response.blob();
        })
        .then((blob) => new File([blob], "morsa.jpg", { type: blob.type || "image/jpeg" }));
    }
    return sampleFilePromise;
  }

  function setFileInputFromFile(file) {
    if (!fileInput) return;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
    handleFilePreview();
  }

  function handleFilePreview() {
    if (!sourcePreviewWrapper || !sourcePreview || !fileInput) return;
    if (!fileInput.files?.length) {
      if (previewUrlRef) {
        URL.revokeObjectURL(previewUrlRef);
        previewUrlRef = null;
      }
      sourcePreviewWrapper.hidden = true;
      sourcePreview.src = "";
      return;
    }

    if (previewUrlRef) {
      URL.revokeObjectURL(previewUrlRef);
    }
    const file = fileInput.files[0];
    previewUrlRef = URL.createObjectURL(file);
    sourcePreview.src = previewUrlRef;
    sourcePreviewWrapper.hidden = false;
    renderGrayscalePreview(file);
  }

    function renderGrayscalePreview(file) {
    if (!grayscalePreviewCanvas) return;
    if (!grayscalePreviewToggle?.checked) {
      grayscalePreviewCanvas.hidden = true;
      return;
    }

    const ctx = grayscalePreviewCanvas.getContext("2d");
    if (!ctx) return;

    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        grayscalePreviewCanvas.width = img.width;
        grayscalePreviewCanvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const data = imageData.data;
        for (let i = 0; i < data.length; i += 4) {
          const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
          data[i] = data[i + 1] = data[i + 2] = gray;
        }
        ctx.putImageData(imageData, 0, 0);
        grayscalePreviewCanvas.hidden = false;
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  }

function applyPreset(name, options = {}) {
    const preset = PRESETS[name];
    if (!preset) return;

    presetButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.preset === name);
    });

    setControlValue("cell-size", preset.cell_size);
    setControlValue("top-candidates", preset.top_candidates);
    setControlValue("max-repeats", preset.max_repeated_tiles);
    setControlValue("repetition-penalty", preset.repetition_penalty);
    setControlValue("neighbor-radius", preset.neighbor_radius);
    setControlValue("blend-ratio", preset.blend_ratio);

    if (metricSelect && preset.distance_metric) {
      metricSelect.value = preset.distance_metric;
    } else if (metricSelect && !preset.distance_metric) {
      metricSelect.value = "hybrid";
    }
    if (enhancementToggle) {
      enhancementToggle.checked = preset.enable_enhancement;
    }

    if (options.loadSample !== false) {
      ensureSampleFile()
        .then(setFileInputFromFile)
        .catch((error) => console.warn("No se pudo asignar la muestra:", error));
    }
  }

  function setControlValue(id, value) {
    const control = document.getElementById(id);
    if (!control) return;
    control.value = value;
    control.dispatchEvent(new Event("input"));
  }

  function updateLivePreview(url) {
    if (!livePreviewImg) return;
    livePreviewImg.src = appendCacheBuster(url);
    livePreviewImg.hidden = false;
    if (livePreviewEmpty) livePreviewEmpty.hidden = true;
  }

  function resetLivePreview() {
    if (!livePreviewImg) return;
    livePreviewImg.hidden = true;
    if (livePreviewEmpty) livePreviewEmpty.hidden = false;
  }

  function resetStatusEvents() {
    progressEventLog = [];
    lastProgressSnapshot = null;
    lastSummaryJobId = null;
    if (statusEventsList) {
      statusEventsList.innerHTML = "";
    }
    if (statusEvents) {
      statusEvents.hidden = true;
    }
  }

  function resetProgressChart() {
    progressSeries.length = 0;
    if (chartStatus) chartStatus.textContent = "Sin datos";
    drawProgressChart();
  }

  function logProcessingEvent(progress) {
    if (!progress) return;

    const message = typeof progress.message === "string" ? progress.message.trim() : "";
    const percent = Number(progress.percent);
    const completed = Number.isFinite(Number(progress.completed)) ? Number(progress.completed) : null;
    const total = Number.isFinite(Number(progress.total)) ? Number(progress.total) : null;

    const snapshot = {
      message,
      percent: Number.isFinite(percent) ? Number(percent) : null,
      completed,
      total,
    };

    if (
      lastProgressSnapshot &&
      snapshot.message === lastProgressSnapshot.message &&
      snapshot.percent === lastProgressSnapshot.percent &&
      snapshot.completed === lastProgressSnapshot.completed &&
      snapshot.total === lastProgressSnapshot.total
    ) {
      return;
    }

    const detailParts = [];
    if (Number.isFinite(snapshot.percent)) {
      const decimals = snapshot.percent >= 100 || snapshot.percent <= 0 ? 0 : 1;
      detailParts.push(`${snapshot.percent.toFixed(decimals)}%`);
    }
    if (total !== null && total > 0) {
      const completedText = completed !== null ? completed : "?";
      detailParts.push(`${completedText}/${total} celdas`);
    } else if (completed !== null && completed > 0) {
      detailParts.push(`${completed} celdas`);
    }

    if (!message && detailParts.length === 0) {
      return;
    }

    lastProgressSnapshot = snapshot;

    const label = message || "Avance del pipeline";
    const text = detailParts.length ? `${label} (${detailParts.join(" | ")})` : label;

    progressEventLog.push({ text, time: new Date() });
    if (progressEventLog.length > MAX_PROCESS_EVENTS) {
      progressEventLog = progressEventLog.slice(-MAX_PROCESS_EVENTS);
    }

    if (statusEventsList) {
      statusEventsList.innerHTML = "";
      progressEventLog.forEach((entry) => {
        const item = document.createElement("li");
        const timeSpan = document.createElement("span");
        timeSpan.className = "event-time";
        timeSpan.textContent = formatEventTime(entry.time);
        const textSpan = document.createElement("span");
        textSpan.className = "event-text";
        textSpan.textContent = entry.text;
        item.append(timeSpan, textSpan);
        statusEventsList.appendChild(item);
      });
    }
    if (statusEvents) {
      statusEvents.hidden = progressEventLog.length === 0;
    }
  }

  function recordProgressPoint(progress, status) {
    if (!progressCanvas) return;
    const total = Number.isFinite(Number(progress.total)) ? Number(progress.total) : null;
    let completed = Number(progress.completed);
    if (!Number.isFinite(completed)) {
      if (total !== null && Number.isFinite(Number(progress.percent))) {
        completed = Math.round((Number(progress.percent) / 100) * total);
      } else {
        return;
      }
    }
    const timestampSeconds =
      jobTimings.startedAt !== null ? (Date.now() - jobTimings.startedAt) / 1000 : progress.elapsed_seconds ?? 0;
    if (!Number.isFinite(timestampSeconds)) {
      return;
    }

    const lastPoint = progressSeries[progressSeries.length - 1];
    if (lastPoint && Math.abs(timestampSeconds - lastPoint.t) < 0.25 && completed === lastPoint.v) {
      return;
    }

    if (!progressSeries.length) {
      const baselineTotal = total ?? completed ?? 0;
      progressSeries.push({ t: 0, v: 0, total: baselineTotal });
    }

    progressSeries.push({ t: Math.max(timestampSeconds, 0), v: Math.max(completed, 0), total });
    if (progressSeries.length > 360) {
      progressSeries.shift();
    }

    if (chartStatus) {
      if (total !== null) {
        chartStatus.textContent = `${completed}/${total} celdas`;
      } else {
        chartStatus.textContent = `${completed} celdas`;
      }
    }

    drawProgressChart(status);
  }

  function drawProgressChart(status) {
    if (!progressCanvas) return;
    const ctx = progressCanvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const displayWidth = progressCanvas.clientWidth || progressCanvas.width || 600;
    const displayHeight = progressCanvas.clientHeight || progressCanvas.height || 220;
    if (progressCanvas.width !== displayWidth * dpr || progressCanvas.height !== displayHeight * dpr) {
      progressCanvas.width = displayWidth * dpr;
      progressCanvas.height = displayHeight * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayWidth, displayHeight);
    ctx.fillStyle = "rgba(15, 23, 42, 0.65)";
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    if (progressSeries.length < 2) {
      ctx.restore();
      return;
    }

    const padding = 28;
    const innerW = Math.max(displayWidth - padding * 2, 10);
    const innerH = Math.max(displayHeight - padding * 2, 10);
    const minT = progressSeries[0].t;
    const maxT = progressSeries[progressSeries.length - 1].t || 1;
    const maxV = Math.max(...progressSeries.map((p) => p.v), 1);

    const xScale = innerW / Math.max(maxT - minT, 0.1);
    const yScale = innerH / Math.max(maxV, 1);

    // Axes
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding, displayHeight - padding);
    ctx.lineTo(displayWidth - padding, displayHeight - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, displayHeight - padding);
    ctx.stroke();
    ctx.setLineDash([]);

    const firstPoint = progressSeries[0];
    const lastPoint = progressSeries[progressSeries.length - 1];
    const firstX = padding + (firstPoint.t - minT) * xScale;
    const firstY = displayHeight - padding - firstPoint.v * yScale;
    const lastX = padding + (lastPoint.t - minT) * xScale;
    const lastY = displayHeight - padding - lastPoint.v * yScale;

    ctx.strokeStyle = status === "completed" ? "rgba(34, 197, 94, 0.85)" : "rgba(56, 189, 248, 0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    progressSeries.forEach((point, index) => {
      const x = padding + (point.t - minT) * xScale;
      const y = displayHeight - padding - point.v * yScale;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    ctx.fillStyle = status === "completed" ? "rgba(34, 197, 94, 0.15)" : "rgba(56, 189, 248, 0.12)";
    ctx.beginPath();
    ctx.moveTo(firstX, firstY);
    progressSeries.forEach((point) => {
      const x = padding + (point.t - minT) * xScale;
      const y = displayHeight - padding - point.v * yScale;
      ctx.lineTo(x, y);
    });
    ctx.lineTo(lastX, displayHeight - padding);
    ctx.lineTo(firstX, displayHeight - padding);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }

  function formatEventTime(date) {
    const reference = date instanceof Date ? date : new Date(date);
    const hours = String(reference.getHours()).padStart(2, "0");
    const minutes = String(reference.getMinutes()).padStart(2, "0");
    const seconds = String(reference.getSeconds()).padStart(2, "0");
    return `${hours}:${minutes}:${seconds}`;
  }

  function setStatus(state, message) {
    if (statusPill && state) {
      statusPill.dataset.state = state;
      statusPill.textContent = state.toUpperCase();
    }
    const resolvedMessage = message || fallbackStatusMessage[state] || (statusMsg ? statusMsg.textContent : "");
    if (statusMsg) {
      statusMsg.textContent = resolvedMessage;
    }

    if (state !== "running" && statusEvents && progressEventLog.length === 0) {
      statusEvents.hidden = true;
    }
  }


  function updateTimings(job) {
    if (!job) return;
    const createdAt = job.created_at ? new Date(job.created_at) : null;
    const finishedAt = job.finished_at ? new Date(job.finished_at) : null;
    if (createdAt) jobTimings.startedAt = createdAt.getTime();

    const progress = job.progress || {};
    if (progress.elapsed_seconds !== undefined) {
      jobTimings.elapsedSeconds = Number(progress.elapsed_seconds);
    }
    if (progress.estimated_total_seconds !== undefined) {
      jobTimings.totalSeconds = Number(progress.estimated_total_seconds);
    }
    if (progress.eta_seconds !== undefined) {
      jobTimings.etaSeconds = Number(progress.eta_seconds);
    }

    const timing = job.timing || job.metrics?.timing;
    if (timing?.total_seconds !== undefined) jobTimings.totalSeconds = Number(timing.total_seconds);
    if (timing?.eta_seconds !== undefined) {
      jobTimings.etaSeconds = Number(timing.eta_seconds);
    } else if (job.status === 'running' && job.progress?.percent) {
      const elapsed = jobTimings.startedAt ? (Date.now() - jobTimings.startedAt) / 1000 : progress.elapsed_seconds ?? 0;
      const percent = Math.max(job.progress.percent, 1e-3);
      const totalEstimate = percent > 0 ? elapsed / (percent / 100) : null;
      if (totalEstimate !== null && Number.isFinite(totalEstimate)) {
        jobTimings.totalSeconds = totalEstimate;
        jobTimings.etaSeconds = Math.max(totalEstimate - elapsed, 0);
      }
    } else if (job.status === 'completed' && finishedAt && jobTimings.startedAt) {
      jobTimings.totalSeconds = (finishedAt.getTime() - jobTimings.startedAt) / 1000;
      jobTimings.etaSeconds = 0;
    }
  }

  function updateStatusMeta() {
    if (!statusMeta || !statusMetaElapsed || !statusMetaEta) return;
    let elapsedSeconds = null;
    if (jobTimings.elapsedSeconds !== undefined) {
      elapsedSeconds = jobTimings.elapsedSeconds;
    } else if (jobTimings.startedAt) {
      elapsedSeconds = (Date.now() - jobTimings.startedAt) / 1000;
    }

    statusMetaElapsed.textContent = elapsedSeconds !== null ? `Tiempo transcurrido: ${formatSeconds(elapsedSeconds)}` : 'Tiempo transcurrido: --';

    if (jobTimings.etaSeconds !== null && jobTimings.etaSeconds >= 0) {
      statusMetaEta.textContent = `ETA: ${formatSeconds(jobTimings.etaSeconds)}`;
    } else if (jobTimings.totalSeconds !== null && elapsedSeconds !== null && jobTimings.totalSeconds >= elapsedSeconds) {
      statusMetaEta.textContent = `ETA: ${formatSeconds(Math.max(jobTimings.totalSeconds - elapsedSeconds, 0))}`;
    } else {
      statusMetaEta.textContent = 'ETA: --';
    }

    statusMeta.hidden = !jobTimings.startedAt;
  }

  function updateProgress(progress, status) {
    if (!progress) return;
    const percentValue = Number(progress.percent);
    if (progressBar) {
      const value = Number.isFinite(percentValue) ? Math.min(100, Math.max(0, percentValue)) : 0;
      progressBar.style.width = `${value}%`;
    }
    if (progress.message && statusMsg) {
      statusMsg.textContent = progress.message;
    }
    if (status === "running" || status === "pending" || status === "completed") {
      recordProgressPoint(progress, status);
    }
    if (status === "running" || status === "pending") {
      logProcessingEvent(progress);
    } else if (statusEvents && progressEventLog.length === 0) {
      statusEvents.hidden = true;
    }
  }

  function appendCacheBuster(url) {
    const separator = url.includes("?") ? "&" : "?";
    return `${url}${separator}v=${Date.now()}`;
  }

  function safeFixed(value) {
    if (value === undefined || value === null) return value;
    const num = Number(value);
    if (!Number.isFinite(num)) return value;
    return num % 1 === 0 ? num : num.toFixed(2);
  }

  function formatSeconds(seconds) {
    if (seconds === undefined || seconds === null) return seconds;
    const num = Number(seconds);
    if (!Number.isFinite(num)) return seconds;
    if (num >= 60) {
      const minutes = Math.floor(num / 60);
      const remaining = (num % 60).toFixed(1);
      return `${minutes}m ${remaining}s`;
    }
    return `${num.toFixed(2)} s`;
  }
})();
