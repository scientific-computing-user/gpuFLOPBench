(function () {
  const PLOTLY_THEME = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Space Grotesk, sans-serif", color: "#eef4ff" },
    xaxis: {
      gridcolor: "rgba(145,183,255,0.10)",
      zerolinecolor: "rgba(145,183,255,0.08)",
      color: "#9bb1d2",
    },
    yaxis: {
      gridcolor: "rgba(145,183,255,0.10)",
      zerolinecolor: "rgba(145,183,255,0.08)",
      color: "#9bb1d2",
    },
    margin: { l: 58, r: 24, t: 30, b: 52 },
    legend: {
      orientation: "h",
      yanchor: "bottom",
      y: 1.04,
      xanchor: "left",
      x: 0,
    },
  };

  const COLORS = {
    3080: "#6ae0bf",
    A10: "#90b7ff",
    A100: "#ffd277",
    H100: "#ff9c5b",
    cuda: "#6ae0bf",
    omp: "#ff9c5b",
  };

  const CLUSTER_COLORS = [
    "#6ae0bf",
    "#90b7ff",
    "#ff9c5b",
    "#ffd277",
    "#7ac7ff",
    "#f7a9c4",
    "#c9a9ff",
    "#8bf0a8",
  ];

  function formatNumber(value, digits) {
    if (digits === undefined) digits = 2;
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }

    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "n/a";
    }

    if (Math.abs(numeric) >= 1e12) return `${(numeric / 1e12).toFixed(2)}T`;
    if (Math.abs(numeric) >= 1e9) return `${(numeric / 1e9).toFixed(2)}B`;
    if (Math.abs(numeric) >= 1e6) return `${(numeric / 1e6).toFixed(2)}M`;
    if (Math.abs(numeric) >= 1e3) return `${(numeric / 1e3).toFixed(1)}K`;
    if (Math.abs(numeric) >= 100) return numeric.toFixed(0);
    return numeric.toFixed(digits);
  }

  function metricTile(metric) {
    const tile = document.createElement("article");
    tile.className = "metric-tile";
    tile.innerHTML = `
      <div class="metric-label">${metric.label}</div>
      <div class="metric-value">${formatNumber(metric.value, 0)}</div>
      <div class="metric-note">${metric.note || ""}</div>
    `;
    return tile;
  }

  function inlineMetric(label, value, digits) {
    const wrapper = document.createElement("div");
    wrapper.className = "inline-metric";
    wrapper.innerHTML = `<span>${label}</span><strong>${formatNumber(value, digits === undefined ? 2 : digits)}</strong>`;
    return wrapper;
  }

  function emptyState(node, message) {
    node.innerHTML = `<div class="panel"><p class="muted">${message}</p></div>`;
  }

  function basePlotlyLayout(overrides) {
    overrides = overrides || {};
    return {
      ...PLOTLY_THEME,
      ...overrides,
      xaxis: { ...PLOTLY_THEME.xaxis, ...(overrides.xaxis || {}) },
      yaxis: { ...PLOTLY_THEME.yaxis, ...(overrides.yaxis || {}) },
      legend: { ...PLOTLY_THEME.legend, ...(overrides.legend || {}) },
      margin: { ...PLOTLY_THEME.margin, ...(overrides.margin || {}) },
    };
  }

  function githubDataUrl(path) {
    return `https://media.githubusercontent.com/media/gregbolet/gpuFLOPBench-updated/main/${path}`;
  }

  window.GFBUtils = {
    COLORS,
    CLUSTER_COLORS,
    formatNumber,
    metricTile,
    inlineMetric,
    emptyState,
    basePlotlyLayout,
    githubDataUrl,
  };
})();
