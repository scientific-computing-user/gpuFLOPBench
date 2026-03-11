(function () {
  const {
    COLORS,
    basePlotlyLayout,
    emptyState,
    formatNumber,
    inlineMetric,
    metricTile,
  } = window.GFBUtils;

  const data = window.GPU_FLOWBENCH_DATA;
  if (!data) {
    console.error("gpuFLOPBench data payload is missing.");
    return;
  }

  const meta = data.meta;
  const kernelRows = data.kernelRows;
  const sourceRows = data.sourceRows;
  const hasPlotly = Boolean(window.Plotly);

  const heroMetricsNode = document.getElementById("heroMetrics");
  const benchmarkSurfaceGridNode = document.getElementById("benchmarkSurfaceGrid");
  const deviceCardsNode = document.getElementById("deviceCards");
  const downloadsGridNode = document.getElementById("downloadsGrid");
  const peakPerfListNode = document.getElementById("peakPerfList");
  const aiDenseListNode = document.getElementById("aiDenseList");
  const readingGuideMetricsNode = document.getElementById("readingGuideMetrics");
  const citationMetaNode = document.getElementById("citationMeta");
  const citationBibtexNode = document.getElementById("citationBibtex");
  const teamGridNode = document.getElementById("teamGrid");
  const lastUpdatedNode = document.getElementById("lastUpdated");

  const modelCoverageNode = document.getElementById("modelCoverageChart");
  const categoryCoverageNode = document.getElementById("categoryCoverageChart");
  const devicePerfNode = document.getElementById("devicePerfChart");
  const rooflineNode = document.getElementById("rooflineChart");
  const rooflineSummaryNode = document.getElementById("rooflineSummary");
  const rooflineReferenceSummaryNode = document.getElementById("rooflineReferenceSummary");
  const rooflineReferencePreviewNode = document.getElementById("rooflineReferencePreview");
  const rooflineReferenceCountNode = document.getElementById("rooflineReferenceCount");
  const devicePerfPanelNode = document.getElementById("devicePerfPanel");
  const rooflineReferencePanelNode = document.getElementById("rooflineReferencePanel");
  const rooflineReferenceBodyNode = document.getElementById("rooflineReferenceBody");
  const rooflineAtlasPanelNode = document.getElementById("rooflineAtlasPanel");
  const rooflineSpecGridNode = document.getElementById("rooflineSpecGrid");
  const rooflineDetailSummaryNode = document.getElementById("rooflineDetailSummary");
  const rooflineDetailBody = document.getElementById("rooflineDetailBody");
  const explorerNode = document.getElementById("explorerChart");
  const sourceTableBody = document.getElementById("sourceTableBody");
  const sourceCountSummary = document.getElementById("sourceCountSummary");

  const rooflineDevice = document.getElementById("rooflineDevice");
  const rooflineModel = document.getElementById("rooflineModel");
  const rooflineProgram = document.getElementById("rooflineProgram");
  const rooflineCategory = document.getElementById("rooflineCategory");
  const rooflineKernel = document.getElementById("rooflineKernel");
  const rooflinePrecision = document.getElementById("rooflinePrecision");
  const explorerDevice = document.getElementById("explorerDevice");
  const explorerModel = document.getElementById("explorerModel");
  const explorerCategory = document.getElementById("explorerCategory");
  const explorerSearch = document.getElementById("explorerSearch");

  function uniqueSorted(values) {
    return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
  }

  function refillSelect(node, values, allLabel) {
    const current = node.value;
    node.innerHTML = "";

    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = allLabel;
    node.appendChild(allOption);

    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      node.appendChild(option);
    });

    node.value = values.includes(current) ? current : "all";
  }

  function buildRooflineRange(rows) {
    const aiValues = rows.map((row) => Number(row.arithmetic_intensity)).filter((value) => Number.isFinite(value) && value > 0);
    const minAI = aiValues.length ? Math.min(...aiValues) : 1e-3;
    const maxAI = aiValues.length ? Math.max(...aiValues) : 1e3;
    return {
      min: Math.pow(10, Math.floor(Math.log10(minAI)) - 0.2),
      max: Math.pow(10, Math.ceil(Math.log10(maxAI)) + 0.2),
    };
  }

  function buildLogSeries(minValue, maxValue, points) {
    const minLog = Math.log10(minValue);
    const maxLog = Math.log10(maxValue);
    return Array.from({ length: points }, function (_, index) {
      const ratio = index / (points - 1);
      return Math.pow(10, minLog + (maxLog - minLog) * ratio);
    });
  }

  function renderHeroMetrics(metrics) {
    metrics.forEach((metric) => heroMetricsNode.appendChild(metricTile(metric)));
  }

  function renderBenchmarkSurfaces() {
    const cards = [
      {
        label: "inventory",
        title: "Benchmark inventory",
        text: `${meta.inventory.totals.benchmarks_yaml} benchmark entries define the source footprint visible in gpuFLOPBench.`,
      },
      {
        label: "profiling",
        title: "Kernel performance corpus",
        text: `${meta.inventory.totals.profiled_sources} profiled source binaries expand into ${kernelRows.length} exact kernel-device rows.`,
      },
      {
        label: "roofline",
        title: "Measured floating-point rooflines",
        text: `Kernel performance is recomputed from floating-point work over execution time and plotted against arithmetic intensity.`,
      },
      {
        label: "exploration",
        title: "Source and kernel drill-down",
        text: `The site keeps category coverage, exact kernel rows, and source-level best-observed performance in one place.`,
      },
    ];

    cards.forEach((item) => {
      const card = document.createElement("article");
      card.className = "paper-card";
      card.innerHTML = `
        <em>${item.label}</em>
        <h3>${item.title}</h3>
        <p>${item.text}</p>
      `;
      benchmarkSurfaceGridNode.appendChild(card);
    });
  }

  function renderDeviceCards(devices) {
    devices.forEach((device) => {
      const card = document.createElement("article");
      card.className = "device-card";
      card.innerHTML = `
        <span class="tag">${device.device}</span>
        <h3>${device.label}</h3>
        <p>${device.architecture}, compute capability ${device.compute_capability}.</p>
      `;
      const metrics = document.createElement("div");
      metrics.className = "inline-metrics";
      metrics.append(
        inlineMetric("sources", device.sources, 0),
        inlineMetric("kernels", device.kernels, 0),
        inlineMetric("bandwidth (GB/s)", device.memory_bandwidth_gbps, 0),
        inlineMetric("peak fp32", device.peak_fp32_tflops, 2)
      );
      card.appendChild(metrics);
      deviceCardsNode.appendChild(card);
    });
  }

  function renderDownloads(downloads) {
    downloads.forEach((item) => {
      const card = document.createElement("article");
      card.className = "download-card";
      card.innerHTML = `
        <span class="tag">download</span>
        <h3>${item.label}</h3>
        <p>${item.path}</p>
        <div class="inline-metrics"></div>
        <div style="margin-top:16px;">
          <a class="button secondary" href="${item.href}" download>Download artifact</a>
        </div>
      `;
      card.querySelector(".inline-metrics").append(inlineMetric("size", item.size_bytes, 0));
      downloadsGridNode.appendChild(card);
    });
  }

  function renderTopList(node, title, items) {
    const card = document.createElement("details");
    card.className = "top-card top-detail";
    const preview = items[0] ? `${items[0].source} on ${items[0].device}` : "No highlighted sources yet";
    card.innerHTML = `
      <summary>
        <div class="top-summary-copy">
          <span class="tag">${title}</span>
          <h3>${title}</h3>
          <p>${preview}. Expand to inspect the highlighted source-device rows.</p>
        </div>
        <div class="top-summary-meta">
          <strong>${Math.min(items.length, 8)}</strong>
          <span>entries</span>
        </div>
      </summary>
    `;
    const list = document.createElement("div");
    list.className = "note-list";

    items.slice(0, 8).forEach((item) => {
      const row = document.createElement("div");
      row.className = "inline-metrics";
      row.style.marginTop = "10px";
      row.innerHTML = `
        <div class="inline-metric" style="flex:1 1 100%;">
          <span>${item.device} / ${item.model_type}</span>
          <strong>${item.source}</strong>
          <div class="metric-note">${item.category}</div>
        </div>
      `;
      row.append(
        inlineMetric("best perf", item.peak_performance_tflops, 4),
        inlineMetric("median AI", item.median_arithmetic_intensity, 4),
        inlineMetric("kernels", item.kernel_count, 0)
      );
      list.appendChild(row);
    });

    card.appendChild(list);
    node.appendChild(card);
  }

  function renderRooflineReference() {
    const selectedPrecision = rooflinePrecision.value;
    const secondaryPrecision = selectedPrecision === "fp32" ? "fp16" : "fp32";
    const specs = meta.roofline_specs.filter((spec) => rooflineDevice.value === "all" || spec.device === rooflineDevice.value);
    rooflineSpecGridNode.innerHTML = "";
    rooflineReferencePreviewNode.innerHTML = "";
    rooflineReferenceCountNode.textContent = String(specs.length);

    if (!specs.length) {
      rooflineReferenceSummaryNode.textContent = "No roofline reference cards match the current device filter.";
      return;
    }

    const scopeLabel =
      rooflineDevice.value === "all"
        ? "All devices"
        : `${specs[0].label}`;
    rooflineReferenceSummaryNode.textContent =
      `${scopeLabel}. ${selectedPrecision.toUpperCase()} roofs at default clocks. Scroll for the hardware cards.`;

    specs.slice(0, 3).forEach((spec) => {
      const pill = document.createElement("div");
      pill.className = "roofline-preview-pill";
      pill.innerHTML = `
        <span>${spec.device}</span>
        <strong>${formatNumber(spec[`peak_${selectedPrecision}_tflops`], 2)} TFLOP/s</strong>
      `;
      rooflineReferencePreviewNode.appendChild(pill);
    });

    if (specs.length > 3) {
      const extra = document.createElement("div");
      extra.className = "roofline-preview-pill roofline-preview-pill-faint";
      extra.innerHTML = `<span>more</span><strong>+${specs.length - 3} GPUs</strong>`;
      rooflineReferencePreviewNode.appendChild(extra);
    }

    specs.forEach((spec) => {
      const card = document.createElement("article");
      card.className = "roofline-spec-card";
      card.innerHTML = `
        <div class="roofline-spec-head">
          <span class="tag">${spec.device}</span>
          <strong>${spec.label}</strong>
        </div>
        <p>${spec.architecture}, compute capability ${spec.compute_capability}.</p>
        <div class="inline-metrics roofline-inline-metrics"></div>
      `;
      const metrics = card.querySelector(".roofline-inline-metrics");
      metrics.append(
        inlineMetric("bandwidth (GB/s)", spec.memory_bandwidth_gbps, 0),
        inlineMetric(`${selectedPrecision.toUpperCase()} roof`, spec[`peak_${selectedPrecision}_tflops`], 2),
        inlineMetric(`${secondaryPrecision.toUpperCase()} roof`, spec[`peak_${secondaryPrecision}_tflops`], 2)
      );
      rooflineSpecGridNode.appendChild(card);
    });
  }

  function syncPerformancePanelHeights() {
    if (!devicePerfPanelNode || !rooflineReferencePanelNode || !rooflineReferenceBodyNode || !rooflineAtlasPanelNode) {
      return;
    }

    if (window.innerWidth <= 1120) {
      rooflineReferencePanelNode.style.height = "";
      rooflineReferenceBodyNode.style.maxHeight = "";
      return;
    }

    const stackNode = devicePerfPanelNode.parentElement;
    const gapValue = stackNode ? window.getComputedStyle(stackNode).rowGap || window.getComputedStyle(stackNode).gap : "20px";
    const gap = Number.parseFloat(gapValue) || 20;
    const atlasHeight = rooflineAtlasPanelNode.getBoundingClientRect().height;
    const overviewHeight = devicePerfPanelNode.getBoundingClientRect().height;
    const targetHeight = Math.max(280, Math.floor(atlasHeight - overviewHeight - gap));

    rooflineReferencePanelNode.style.height = `${targetHeight}px`;
    rooflineReferenceBodyNode.style.maxHeight = `${Math.max(180, targetHeight - 142)}px`;
  }

  function queuePerformancePanelSync() {
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(syncPerformancePanelHeights);
    });
  }

  function renderReadingGuide() {
    const uniqueCategories = new Set(meta.category_profiled.map((entry) => entry.category)).size;
    readingGuideMetricsNode.append(
      inlineMetric("GPUs", meta.device_summary.length, 0),
      inlineMetric("profiled binaries", meta.inventory.totals.profiled_sources, 0),
      inlineMetric("kernel rows", kernelRows.length, 0),
      inlineMetric("categories", uniqueCategories, 0)
    );
  }

  function renderCitation() {
    if (!citationMetaNode || !citationBibtexNode || !meta.paper) {
      return;
    }
    citationMetaNode.innerHTML = `
      <div class="citation-actions">
        <div>
          <span class="tag">${meta.paper.venue}</span>
          <h3>${meta.paper.title}</h3>
          <p>Reference link and PDF for the paper that launched this project direction.</p>
        </div>
        <div class="hero-actions" style="margin-top:0;">
          <a class="button secondary" href="${meta.paper.doi_url}" target="_blank" rel="noreferrer">Conference entry</a>
          <a class="button primary" href="${meta.paper.pdf_url}" target="_blank" rel="noreferrer">${meta.paper.pdf_label}</a>
        </div>
      </div>
    `;
    citationBibtexNode.textContent = meta.paper.bibtex;
  }

  function renderTeam(team) {
    if (!teamGridNode || !Array.isArray(team) || !team.length) {
      return;
    }
    team.forEach((member) => {
      const card = document.createElement("article");
      card.className = "team-card";
      card.innerHTML = `
        <img class="team-photo" src="${member.image_path}" alt="${member.name}">
        <div class="team-copy">
          <h3>${member.name}</h3>
          <p>${member.affiliation}</p>
          <a class="button secondary" href="${member.profile_url}" target="_blank" rel="noreferrer">Profile</a>
        </div>
      `;
      teamGridNode.appendChild(card);
    });
  }

  function renderPlot(node, traces, layout, emptyMessage) {
    if (!hasPlotly) {
      emptyState(node, emptyMessage || "Interactive charts require Plotly to load.");
      return;
    }
    window.Plotly.react(node, traces, layout, { responsive: true, displayModeBar: false });
  }

  function renderModelCoverage(modelMatrix) {
    renderPlot(
      modelCoverageNode,
      [
        {
          type: "bar",
          name: "declared sources",
          x: modelMatrix.map((entry) => entry.model.toUpperCase()),
          y: modelMatrix.map((entry) => entry.available),
          marker: { color: modelMatrix.map((entry) => COLORS[entry.model] || "#90b7ff") },
        },
        {
          type: "bar",
          name: "profiled sources",
          x: modelMatrix.map((entry) => entry.model.toUpperCase()),
          y: modelMatrix.map((entry) => entry.profiled),
          marker: { color: "rgba(255, 156, 91, 0.84)" },
        },
      ],
      basePlotlyLayout({ barmode: "group", yaxis: { title: "source binaries" } })
    );
  }

  function renderCategoryCoverage(categoryProfiled) {
    const categories = [...new Set(categoryProfiled.map((entry) => entry.category))];
    const models = [...new Set(categoryProfiled.map((entry) => entry.model_type))];
    renderPlot(
      categoryCoverageNode,
      models.map((model) => ({
        type: "bar",
        orientation: "h",
        name: model.toUpperCase(),
        y: categories,
        x: categories.map((category) => {
          const entry = categoryProfiled.find((row) => row.category === category && row.model_type === model);
          return entry ? entry.profiled_sources : 0;
        }),
        marker: { color: COLORS[model] || "#90b7ff" },
      })),
      basePlotlyLayout({
        barmode: "stack",
        xaxis: { title: "profiled source binaries" },
        yaxis: { automargin: true },
        margin: { l: 180, r: 24, t: 26, b: 48 },
      })
    );
  }

  function renderDevicePerf(devices) {
    renderPlot(
      devicePerfNode,
      [
        {
          type: "bar",
          name: "sources",
          x: devices.map((device) => device.device),
          y: devices.map((device) => device.sources),
          marker: { color: "rgba(144, 183, 255, 0.76)" },
          yaxis: "y",
        },
        {
          type: "scatter",
          mode: "lines+markers",
          name: "median performance",
          x: devices.map((device) => device.device),
          y: devices.map((device) => device.median_performance_tflops),
          marker: { color: "#ff9c5b", size: 10 },
          line: { color: "#ff9c5b", width: 3 },
          yaxis: "y2",
        },
      ],
      basePlotlyLayout({
        yaxis: { title: "source count" },
        yaxis2: {
          title: "median performance (TFLOP/s)",
          overlaying: "y",
          side: "right",
          gridcolor: "rgba(0,0,0,0)",
          color: "#ff9c5b",
        },
      })
    );
    queuePerformancePanelSync();
  }

  function matchesRooflineFilters(row) {
    const matchesDevice = rooflineDevice.value === "all" || row.device === rooflineDevice.value;
    const matchesModel = rooflineModel.value === "all" || row.model_type === rooflineModel.value;
    const matchesProgram = rooflineProgram.value === "all" || row.benchmark === rooflineProgram.value;
    const matchesCategory = rooflineCategory.value === "all" || row.category === rooflineCategory.value;
    const matchesKernel = rooflineKernel.value === "all" || row.kernel === rooflineKernel.value;
    return matchesDevice && matchesModel && matchesProgram && matchesCategory && matchesKernel;
  }

  function filteredKernelRows(rows) {
    return rows.filter(matchesRooflineFilters);
  }

  function syncRooflineFilters() {
    const baseRows = kernelRows.filter((row) => {
      const matchesDevice = rooflineDevice.value === "all" || row.device === rooflineDevice.value;
      const matchesModel = rooflineModel.value === "all" || row.model_type === rooflineModel.value;
      const matchesCategory = rooflineCategory.value === "all" || row.category === rooflineCategory.value;
      return matchesDevice && matchesModel && matchesCategory;
    });

    refillSelect(rooflineProgram, uniqueSorted(baseRows.map((row) => row.benchmark)), "all programs");

    const programRows = baseRows.filter((row) => rooflineProgram.value === "all" || row.benchmark === rooflineProgram.value);
    refillSelect(rooflineKernel, uniqueSorted(programRows.map((row) => row.kernel)), "all kernels");
  }

  function renderRooflineDetails(rows) {
    const subset = filteredKernelRows(rows).sort((left, right) => Number(right.performance_tflops) - Number(left.performance_tflops));
    rooflineDetailBody.innerHTML = "";

    if (!subset.length) {
      rooflineDetailSummaryNode.textContent = "No exact kernel rows match the current filters.";
      return;
    }

    rooflineDetailSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> exact kernel rows match the current filters.
      Narrow to a single program and kernel to inspect one row directly.
    `;

    subset.slice(0, 32).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <strong>${row.source}</strong>
          <span>${row.category}</span>
        </td>
        <td>
          <strong>${row.kernel}</strong>
          <span>block ${row.block_size || "n/a"} | grid ${row.grid_size || "n/a"}</span>
        </td>
        <td><span class="tag">${row.device}</span></td>
        <td><span class="tag">${row.model_type}</span></td>
        <td class="mono">${formatNumber(row.performance_tflops, 4)}</td>
        <td class="mono">${formatNumber(row.arithmetic_intensity, 4)}</td>
        <td class="mono">${formatNumber(row.float_flops, 0)}</td>
        <td class="mono">${formatNumber(row.bytes_total, 0)}</td>
        <td class="mono">${formatNumber(row.xtime_ns, 2)}</td>
      `;
      rooflineDetailBody.appendChild(tr);
    });
  }

  function renderRoofline(rows) {
    const exactRows = filteredKernelRows(rows);
    const subset = exactRows.filter((row) => Number(row.arithmetic_intensity) > 0 && Number(row.performance_tflops) > 0);
    renderRooflineDetails(rows);
    renderRooflineReference();

    if (!subset.length) {
      emptyState(rooflineNode, "No floating-point kernel rows match the current filters.");
      rooflineSummaryNode.textContent = "";
      queuePerformancePanelSync();
      return;
    }

    const selectedPrecision = rooflinePrecision.value;
    const range = buildRooflineRange(subset);
    const roofSpecs = meta.roofline_specs.filter((spec) => rooflineDevice.value === "all" || spec.device === rooflineDevice.value);
    const xSeries = buildLogSeries(range.min, range.max, 60);

    const roofTraces = roofSpecs.map((spec) => ({
      type: "scatter",
      mode: "lines",
      name: `${spec.device} ${selectedPrecision.toUpperCase()} roof`,
      x: xSeries,
      y: xSeries.map((ai) => Math.min(ai * spec.memory_bandwidth_gbps / 1000.0, spec[`peak_${selectedPrecision}_tflops`])),
      hovertemplate:
        `<b>${spec.label}</b><br>` +
        `precision=${selectedPrecision.toUpperCase()}<br>` +
        `bandwidth=${formatNumber(spec.memory_bandwidth_gbps, 0)} GB/s<extra></extra>`,
      line: {
        color: COLORS[spec.device] || "#90b7ff",
        width: 2,
        dash: "dash",
      },
      opacity: 0.8,
    }));

    const pointTraces = uniqueSorted(subset.map((row) => row.device)).map((device) => {
      const deviceRows = subset.filter((row) => row.device === device);
      return {
        type: "scattergl",
        mode: "markers",
        name: device,
        x: deviceRows.map((row) => row.arithmetic_intensity),
        y: deviceRows.map((row) => row.performance_tflops),
        text: deviceRows.map((row) => `${row.source}<br>${row.kernel}`),
        customdata: deviceRows.map((row) => [row.category, row.model_type, row.dominant_precision, row.xtime_ns]),
        hovertemplate:
          "<b>%{text}</b><br>" +
          "category=%{customdata[0]}<br>" +
          "model=%{customdata[1]}<br>" +
          "dominant precision=%{customdata[2]}<br>" +
          "AI=%{x:.4f}<br>" +
          "performance=%{y:.4f} TFLOP/s<br>" +
          "time=%{customdata[3]:.2f} ns<extra></extra>",
        marker: {
          size: 8,
          opacity: 0.72,
          color: COLORS[device] || "#90b7ff",
        },
      };
    });

    renderPlot(
      rooflineNode,
      roofTraces.concat(pointTraces),
      basePlotlyLayout({
        xaxis: { title: "arithmetic intensity (FLOPs / byte)", type: "log" },
        yaxis: { title: "performance (TFLOP/s)", type: "log" },
        margin: { l: 64, r: 28, t: 30, b: 58 },
      })
    );

    const perfValues = subset.map((row) => Number(row.performance_tflops)).sort((left, right) => left - right);
    const aiValues = subset.map((row) => Number(row.arithmetic_intensity)).sort((left, right) => left - right);
    rooflineSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> floating-point kernel rows in view.
      Median AI <strong>${formatNumber(aiValues[Math.floor(aiValues.length / 2)], 4)}</strong>,
      median performance <strong>${formatNumber(perfValues[Math.floor(perfValues.length / 2)], 4)} TFLOP/s</strong>.
      Dashed lines show the ${selectedPrecision.toUpperCase()} theoretical roofline at default device clocks.
    `;
    queuePerformancePanelSync();
  }

  function filteredSourceRows(rows) {
    const search = explorerSearch.value.trim().toLowerCase();
    return rows.filter((row) => {
      const matchesDevice = explorerDevice.value === "all" || row.device === explorerDevice.value;
      const matchesModel = explorerModel.value === "all" || row.model_type === explorerModel.value;
      const matchesCategory = explorerCategory.value === "all" || row.category === explorerCategory.value;
      const matchesSearch =
        !search ||
        row.source.toLowerCase().includes(search) ||
        row.benchmark.toLowerCase().includes(search);
      return matchesDevice && matchesModel && matchesCategory && matchesSearch;
    });
  }

  function renderSourceTable(rows) {
    sourceTableBody.innerHTML = "";
    rows.slice(0, 120).forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <strong>${row.source}</strong>
          <span>${row.category}</span>
        </td>
        <td><span class="tag">${row.device}</span></td>
        <td><span class="tag">${row.model_type}</span></td>
        <td class="mono">${formatNumber(row.kernel_count, 0)}</td>
        <td class="mono">${formatNumber(row.peak_performance_tflops, 4)}</td>
        <td class="mono">${formatNumber(row.median_arithmetic_intensity, 4)}</td>
        <td class="mono">${formatNumber(row.median_xtime_ns, 2)}</td>
        <td class="mono">${formatNumber(row.coverage_rank, 0)}</td>
      `;
      sourceTableBody.appendChild(tr);
    });
  }

  function renderSourceChart(rows) {
    renderPlot(
      explorerNode,
      uniqueSorted(rows.map((row) => row.device)).map((device) => {
        const subset = rows.filter((row) => row.device === device && Number(row.peak_performance_tflops) > 0 && Number(row.median_arithmetic_intensity) > 0);
        return {
          type: "scattergl",
          mode: "markers",
          name: device,
          x: subset.map((row) => row.median_arithmetic_intensity),
          y: subset.map((row) => row.peak_performance_tflops),
          text: subset.map((row) => row.source),
          customdata: subset.map((row) => [row.model_type, row.kernel_count, row.category]),
          hovertemplate:
            "<b>%{text}</b><br>" +
            "model=%{customdata[0]}<br>" +
            "kernels=%{customdata[1]}<br>" +
            "category=%{customdata[2]}<br>" +
            "median AI=%{x:.4f}<br>" +
            "best performance=%{y:.4f} TFLOP/s<extra></extra>",
          marker: {
            color: COLORS[device] || "#90b7ff",
            size: subset.map((row) => Math.max(8, Math.min(28, Number(row.kernel_count) || 8))),
            opacity: 0.66,
          },
        };
      }),
      basePlotlyLayout({
        xaxis: { title: "median source AI (FLOPs / byte)", type: "log" },
        yaxis: { title: "best observed source performance (TFLOP/s)", type: "log" },
      })
    );
  }

  function renderExplorer(rows) {
    const subset = filteredSourceRows(rows).sort((left, right) => Number(right.peak_performance_tflops) - Number(left.peak_performance_tflops));
    sourceCountSummary.innerHTML = `
      <strong>${subset.length}</strong> source-device rows match the current filters.
      The table below shows the top 120 by best observed floating-point performance.
    `;
    renderSourceTable(subset);
    renderSourceChart(subset);
  }

  function init() {
    renderHeroMetrics(meta.hero.headline_metrics);
    renderBenchmarkSurfaces();
    renderDeviceCards(meta.device_summary);
    renderDownloads(meta.downloads);
    renderTopList(peakPerfListNode, "Performance leaders", meta.top_lists.performance_sources);
    renderTopList(aiDenseListNode, "AI-dense leaders", meta.top_lists.ai_dense_sources);
    renderReadingGuide();
    renderCitation();
    renderTeam(meta.team);
    renderModelCoverage(meta.model_matrix);
    renderCategoryCoverage(meta.category_profiled);
    renderDevicePerf(meta.device_summary);

    refillSelect(rooflineDevice, uniqueSorted(kernelRows.map((row) => row.device)), "all devices");
    refillSelect(rooflineModel, uniqueSorted(kernelRows.map((row) => row.model_type)), "all models");
    refillSelect(rooflineCategory, uniqueSorted(kernelRows.map((row) => row.category)), "all categories");
    syncRooflineFilters();

    refillSelect(explorerDevice, uniqueSorted(sourceRows.map((row) => row.device)), "all devices");
    refillSelect(explorerModel, uniqueSorted(sourceRows.map((row) => row.model_type)), "all models");
    refillSelect(explorerCategory, uniqueSorted(sourceRows.map((row) => row.category)), "all categories");

    [rooflineDevice, rooflineModel, rooflineProgram, rooflineCategory, rooflineKernel, rooflinePrecision].forEach((node) => {
      node.addEventListener("change", function () {
        syncRooflineFilters();
        renderRoofline(kernelRows);
      });
    });

    [explorerDevice, explorerModel, explorerCategory].forEach((node) => {
      node.addEventListener("change", function () {
        renderExplorer(sourceRows);
      });
    });
    explorerSearch.addEventListener("input", function () {
      renderExplorer(sourceRows);
    });

    renderRoofline(kernelRows);
    renderExplorer(sourceRows);
    lastUpdatedNode.textContent = new Date(meta.audit.generated_at).toLocaleString();
    window.addEventListener("resize", queuePerformancePanelSync);
  }

  init();
})();
