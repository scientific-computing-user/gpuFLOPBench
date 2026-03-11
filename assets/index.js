(function () {
  const {
    COLORS,
    CLUSTER_COLORS,
    basePlotlyLayout,
    emptyState,
    formatNumber,
    githubDataUrl,
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
  const featureRows = data.featureRows;
  const sourceRows = data.sourceRows;
  const hasPlotly = Boolean(window.Plotly);

  const heroMetricsNode = document.getElementById("heroMetrics");
  const benchmarkSurfaceGridNode = document.getElementById("benchmarkSurfaceGrid");
  const deviceCardsNode = document.getElementById("deviceCards");
  const downloadsGridNode = document.getElementById("downloadsGrid");
  const peakPerfListNode = document.getElementById("peakPerfList");
  const aiDenseListNode = document.getElementById("aiDenseList");
  const clusterGridNode = document.getElementById("clusterGrid");
  const readingGuideMetricsNode = document.getElementById("readingGuideMetrics");
  const lastUpdatedNode = document.getElementById("lastUpdated");

  const modelCoverageNode = document.getElementById("modelCoverageChart");
  const categoryCoverageNode = document.getElementById("categoryCoverageChart");
  const devicePerfNode = document.getElementById("devicePerfChart");
  const rooflineNode = document.getElementById("rooflineChart");
  const rooflineSummaryNode = document.getElementById("rooflineSummary");
  const featureNode = document.getElementById("featureChart");
  const featureSummaryNode = document.getElementById("featureSummary");
  const explorerNode = document.getElementById("explorerChart");
  const sourceTableBody = document.getElementById("sourceTableBody");
  const sourceCountSummary = document.getElementById("sourceCountSummary");

  const rooflineDevice = document.getElementById("rooflineDevice");
  const rooflineModel = document.getElementById("rooflineModel");
  const rooflineCategory = document.getElementById("rooflineCategory");
  const featureDevice = document.getElementById("featureDevice");
  const featureModel = document.getElementById("featureModel");
  const explorerDevice = document.getElementById("explorerDevice");
  const explorerModel = document.getElementById("explorerModel");
  const explorerCategory = document.getElementById("explorerCategory");
  const explorerSearch = document.getElementById("explorerSearch");

  function fillSelect(node, values) {
    values.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      node.appendChild(option);
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
        text: `${meta.inventory.totals.benchmarks_yaml} benchmark entries define the visible footprint of gpuFLOPBench across the source space tracked in this site.`,
      },
      {
        label: "profiling",
        title: "Kernel performance corpus",
        text: `${meta.inventory.totals.profiled_sources} profiled source binaries expand into ${kernelRows.length} aggregated kernel-device rows.`,
      },
      {
        label: "feature space",
        title: "Instruction-mix layer",
        text: `${featureRows.length} feature-space rows expose repeatable structure in operation-type behavior.`,
      },
      {
        label: "hardware",
        title: "Cross-device view",
        text: `${meta.device_summary.map((device) => device.device).join(", ")} are represented directly in the benchmark atlas.`,
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
        <p>${device.family} generation, ${device.compute_capability} compute capability.</p>
      `;
      const metrics = document.createElement("div");
      metrics.className = "inline-metrics";
      metrics.append(
        inlineMetric("sources", device.sources, 0),
        inlineMetric("kernels", device.kernels, 0),
        inlineMetric("median throughput", device.median_total_perf),
        inlineMetric("median AI", device.median_total_ai, 4)
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
          <a class="button secondary" href="${githubDataUrl(item.path)}" target="_blank" rel="noreferrer">Open artifact</a>
        </div>
      `;
      card.querySelector(".inline-metrics").append(inlineMetric("size", item.size_bytes, 0));
      downloadsGridNode.appendChild(card);
    });
  }

  function renderTopList(node, title, items) {
    const card = document.createElement("article");
    card.className = "top-card";
    card.innerHTML = `<span class="tag">${title}</span><h3>${title}</h3>`;
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
        inlineMetric("peak throughput", item.peak_total_perf),
        inlineMetric("median AI", item.median_total_ai, 4),
        inlineMetric("kernels", item.kernel_count, 0)
      );
      list.appendChild(row);
    });

    card.appendChild(list);
    node.appendChild(card);
  }

  function renderReadingGuide() {
    const uniqueCategories = new Set(meta.category_profiled.map((entry) => entry.category)).size;
    readingGuideMetricsNode.append(
      inlineMetric("GPUs", meta.device_summary.length, 0),
      inlineMetric("source rows", meta.inventory.totals.profiled_sources, 0),
      inlineMetric("feature rows", featureRows.length, 0),
      inlineMetric("categories", uniqueCategories, 0)
    );
  }

  function renderClusterCards(clusters) {
    clusters.forEach((cluster, index) => {
      const detail = document.createElement("details");
      detail.className = "cluster-detail";
      detail.open = index === 0;
      detail.innerHTML = `
        <summary>
          <div class="cluster-detail-title">
            <span class="tag">cluster ${cluster.cluster}</span>
            <h3>${cluster.top_op_types.map((entry) => entry.name).join(" + ")}</h3>
            <div class="cluster-detail-summary">
              <span>${formatNumber(cluster.size, 0)} feature rows</span>
              <span>median IPC ${formatNumber(cluster.median_ipc, 2)}</span>
              <span>median AI ${formatNumber(cluster.median_total_ai, 4)}</span>
            </div>
          </div>
          <span class="cluster-detail-caret" aria-hidden="true">+</span>
        </summary>
        <div class="cluster-detail-body">
          <p>Dominant operation blend across the cluster median profile.</p>
          <div class="inline-metrics"></div>
        </div>
      `;
      const metrics = detail.querySelector(".inline-metrics");
      cluster.top_op_types.forEach((entry) => metrics.append(inlineMetric(entry.name, entry.value, 3)));
      clusterGridNode.appendChild(detail);
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
          marker: { color: "rgba(144, 183, 255, 0.8)" },
        },
        {
          type: "bar",
          name: "profiled sources",
          x: modelMatrix.map((entry) => entry.model.toUpperCase()),
          y: modelMatrix.map((entry) => entry.profiled),
          marker: { color: "rgba(106, 224, 191, 0.9)" },
        },
        {
          type: "bar",
          name: "feature-space sources",
          x: modelMatrix.map((entry) => entry.model.toUpperCase()),
          y: modelMatrix.map((entry) => entry.feature_enriched),
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
        margin: { l: 110, r: 24, t: 26, b: 48 },
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
          name: "median throughput",
          x: devices.map((device) => device.device),
          y: devices.map((device) => device.median_total_perf),
          marker: { color: "#ff9c5b", size: 10 },
          line: { color: "#ff9c5b", width: 3 },
          yaxis: "y2",
        },
      ],
      basePlotlyLayout({
        yaxis: { title: "source count" },
        yaxis2: {
          title: "median throughput",
          overlaying: "y",
          side: "right",
          gridcolor: "rgba(0,0,0,0)",
          color: "#ff9c5b",
        },
      })
    );
  }

  function filteredKernelRows(rows) {
    return rows.filter((row) => {
      const matchesDevice = rooflineDevice.value === "all" || row.device === rooflineDevice.value;
      const matchesModel = rooflineModel.value === "all" || row.model_type === rooflineModel.value;
      const matchesCategory = rooflineCategory.value === "all" || row.category === rooflineCategory.value;
      return matchesDevice && matchesModel && matchesCategory && row.total_ai > 0 && row.total_perf > 0;
    });
  }

  function renderRoofline(rows) {
    const subset = filteredKernelRows(rows);
    if (!subset.length) {
      emptyState(rooflineNode, "No kernel rows match the current filters.");
      rooflineSummaryNode.textContent = "";
      return;
    }

    renderPlot(
      rooflineNode,
      [...new Set(subset.map((row) => row.device))].map((device) => {
        const deviceRows = subset.filter((row) => row.device === device);
        return {
          type: "scattergl",
          mode: "markers",
          name: device,
          x: deviceRows.map((row) => row.total_ai),
          y: deviceRows.map((row) => row.total_perf),
          text: deviceRows.map((row) => `${row.source}<br>${row.kernel}`),
          customdata: deviceRows.map((row) => [row.category, row.model_type, row.device]),
          hovertemplate:
            "<b>%{text}</b><br>" +
            "category=%{customdata[0]}<br>" +
            "model=%{customdata[1]}<br>" +
            "device=%{customdata[2]}<br>" +
            "AI=%{x:.4f}<br>" +
            "throughput=%{y:.3e}<extra></extra>",
          marker: {
            size: 8,
            opacity: 0.72,
            color: COLORS[device] || "#90b7ff",
          },
        };
      }),
      basePlotlyLayout({
        xaxis: { title: "total arithmetic intensity", type: "log" },
        yaxis: { title: "total throughput", type: "log" },
        margin: { l: 58, r: 26, t: 30, b: 58 },
      })
    );

    const perfValues = subset.map((row) => row.total_perf).sort((left, right) => left - right);
    const aiValues = subset.map((row) => row.total_ai).sort((left, right) => left - right);
    rooflineSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> kernel-device rows in view.
      Median AI <strong>${formatNumber(aiValues[Math.floor(aiValues.length / 2)], 4)}</strong>,
      median throughput <strong>${formatNumber(perfValues[Math.floor(perfValues.length / 2)])}</strong>.
    `;
  }

  function filteredFeatureRows(rows) {
    return rows.filter((row) => {
      const matchesDevice = featureDevice.value === "all" || row.device === featureDevice.value;
      const matchesModel = featureModel.value === "all" || row.model_type === featureModel.value;
      return matchesDevice && matchesModel;
    });
  }

  function renderFeature(rows) {
    const subset = filteredFeatureRows(rows);
    if (!subset.length) {
      emptyState(featureNode, "No feature rows match the current filters.");
      featureSummaryNode.textContent = "";
      return;
    }

    renderPlot(
      featureNode,
      [...new Set(subset.map((row) => row.cluster))]
        .sort((left, right) => left - right)
        .map((cluster) => {
          const clusterRows = subset.filter((row) => row.cluster === cluster);
          return {
            type: "scattergl",
            mode: "markers",
            name: `cluster ${cluster}`,
            x: clusterRows.map((row) => row.pc1),
            y: clusterRows.map((row) => row.pc2),
            text: clusterRows.map((row) => `${row.source}<br>${row.kernel}`),
            customdata: clusterRows.map((row) => [row.dominant_op_type, row.total_ai, row.ipc_active, row.device]),
            hovertemplate:
              "<b>%{text}</b><br>" +
              "dominant op=%{customdata[0]}<br>" +
              "AI=%{customdata[1]:.4f}<br>" +
              "IPC=%{customdata[2]:.4f}<br>" +
              "device=%{customdata[3]}<extra></extra>",
            marker: {
              size: 8,
              opacity: 0.72,
              color: CLUSTER_COLORS[cluster % CLUSTER_COLORS.length],
            },
          };
        }),
      basePlotlyLayout({
        xaxis: { title: "feature PC1" },
        yaxis: { title: "feature PC2" },
        margin: { l: 56, r: 22, t: 28, b: 50 },
      })
    );

    featureSummaryNode.innerHTML = `
      <strong>${subset.length}</strong> feature-space rows in view.
      ${new Set(subset.map((row) => row.dominant_op_type)).size} dominant op families remain visible after filtering.
    `;
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
        <td class="mono">${formatNumber(row.peak_total_perf)}</td>
        <td class="mono">${formatNumber(row.median_total_ai, 4)}</td>
        <td class="mono">${formatNumber(row.median_xtime)}</td>
        <td class="mono">${formatNumber(row.coverage_rank, 0)}</td>
      `;
      sourceTableBody.appendChild(tr);
    });
  }

  function renderSourceChart(rows) {
    renderPlot(
      explorerNode,
      [...new Set(rows.map((row) => row.device))].map((device) => {
        const subset = rows.filter((row) => row.device === device && row.peak_total_perf > 0 && row.median_total_ai > 0);
        return {
          type: "scattergl",
          mode: "markers",
          name: device,
          x: subset.map((row) => row.median_total_ai),
          y: subset.map((row) => row.peak_total_perf),
          text: subset.map((row) => row.source),
          customdata: subset.map((row) => [row.model_type, row.kernel_count, row.category]),
          hovertemplate:
            "<b>%{text}</b><br>" +
            "model=%{customdata[0]}<br>" +
            "kernels=%{customdata[1]}<br>" +
            "category=%{customdata[2]}<br>" +
            "median AI=%{x:.4f}<br>" +
            "peak throughput=%{y:.3e}<extra></extra>",
          marker: {
            color: COLORS[device] || "#90b7ff",
            size: subset.map((row) => Math.max(8, Math.min(28, Number(row.kernel_count) || 8))),
            opacity: 0.66,
          },
        };
      }),
      basePlotlyLayout({
        xaxis: { title: "median source AI", type: "log" },
        yaxis: { title: "peak source throughput", type: "log" },
      })
    );
  }

  function renderExplorer(rows) {
    const subset = filteredSourceRows(rows).sort((left, right) => Number(right.peak_total_perf) - Number(left.peak_total_perf));
    sourceCountSummary.innerHTML = `
      <strong>${subset.length}</strong> source-device rows match the current filters.
      The table below shows the top 120 by peak throughput.
    `;
    renderSourceTable(subset);
    renderSourceChart(subset);
  }

  function init() {
    renderHeroMetrics(meta.hero.headline_metrics);
    renderBenchmarkSurfaces();
    renderDeviceCards(meta.device_summary);
    renderDownloads(meta.downloads);
    renderTopList(peakPerfListNode, "Throughput leaders", meta.top_lists.peak_perf_sources);
    renderTopList(aiDenseListNode, "AI-dense leaders", meta.top_lists.ai_dense_sources);
    renderReadingGuide();
    renderClusterCards(meta.cluster_summary.clusters);
    renderModelCoverage(meta.model_matrix);
    renderCategoryCoverage(meta.category_profiled);
    renderDevicePerf(meta.device_summary);

    fillSelect(rooflineDevice, [...new Set(kernelRows.map((row) => row.device))].sort());
    fillSelect(rooflineModel, [...new Set(kernelRows.map((row) => row.model_type))].sort());
    fillSelect(rooflineCategory, [...new Set(kernelRows.map((row) => row.category))].sort());
    fillSelect(featureDevice, [...new Set(featureRows.map((row) => row.device))].sort());
    fillSelect(featureModel, [...new Set(featureRows.map((row) => row.model_type))].sort());
    fillSelect(explorerDevice, [...new Set(sourceRows.map((row) => row.device))].sort());
    fillSelect(explorerModel, [...new Set(sourceRows.map((row) => row.model_type))].sort());
    fillSelect(explorerCategory, [...new Set(sourceRows.map((row) => row.category))].sort());

    [rooflineDevice, rooflineModel, rooflineCategory].forEach((node) => {
      node.addEventListener("change", function () {
        renderRoofline(kernelRows);
      });
    });
    [featureDevice, featureModel].forEach((node) => {
      node.addEventListener("change", function () {
        renderFeature(featureRows);
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
    renderFeature(featureRows);
    renderExplorer(sourceRows);
    lastUpdatedNode.textContent = new Date(meta.audit.generated_at).toLocaleString();
  }

  init();
})();
