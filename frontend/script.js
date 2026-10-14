
const API = "";

function showResultModal(id) {
  document.getElementById(id)?.classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function closeResultModal(modal) {
  modal.classList.add("hidden");
  if (!document.querySelector(".result-modal:not(.hidden)")) {
    document.body.classList.remove("modal-open");
  }
}

document.querySelectorAll(".result-modal").forEach(modal => {
  modal.querySelectorAll("[data-close-modal]").forEach(closeButton => {
    closeButton.addEventListener("click", () => closeResultModal(modal));
  });
});

// ============================================================
// Tabs
// ============================================================

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });

    document.querySelectorAll(".panel-view").forEach(panel => {
      panel.classList.remove("active");
    });

    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    document.getElementById(tab.dataset.tab)?.classList.add("active");
  });
});

// ============================================================
// Dashboard
// ============================================================



async function loadDashboard() {
  const loading = document.getElementById("dashboard-loading");
  const error = document.getElementById("dashboard-error");
  const content = document.getElementById("dashboard-content");

  try {
    const [healthResponse, modelResponse, statsResponse] =
      await Promise.all([
        fetch(`${API}/api/health`),
        fetch(`${API}/api/model`),
        fetch(`${API}/api/dashboard/stats`)
      ]);

    if (!healthResponse.ok || !modelResponse.ok || !statsResponse.ok) {
      throw new Error("Could not load dashboard information.");
    }

    const health = await healthResponse.json();
    const model = await modelResponse.json();
    const stats = await statsResponse.json();

    updateLiveAnalytics(stats);
    loading.classList.add("hidden");
    error.classList.add("hidden");
    content.classList.remove("hidden");

    // Model information
    document.getElementById("best-model-name").textContent =
      model.model || "LightGBM";

    document.getElementById("best-model-sub").textContent =
      "Experimental behavioural fraud detection model";

    // Live prediction statistics
    document.getElementById("stat-rows").textContent =
      stats.total_predictions.toLocaleString();

    document.getElementById("stat-fraud-rate").textContent =
      stats.total_predictions > 0
        ? `${((stats.suspicious_clicks / stats.total_predictions) * 100).toFixed(2)}%`
        : "0.00%";

    const featureCount = model.features?.length || 0;
    const featureCard = document.getElementById("model-feature-count");

    if (featureCard) {
      featureCard.textContent = featureCount;
    }

    // Update dataset/model warning
    const note = document.getElementById("model-warning");

    if (note) {
      note.textContent = model.warning ||
        "Experimental model. Results require independent validation.";
    }

  } catch (err) {
    console.error("Dashboard loading error:", err);

    loading.classList.add("hidden");
    content.classList.add("hidden");

    error.textContent = err.message;
    error.classList.remove("hidden");
  }
}


// ============================================================
// Live Prediction
// ============================================================

const predictForm = document.getElementById("predict-form");

if (predictForm) {
  predictForm.addEventListener("submit", async event => {
    event.preventDefault();

    const submitButton = predictForm.querySelector(
      'button[type="submit"]'
    );

    const errorElement = document.getElementById("result-error");
    const emptyElement = document.getElementById("result-empty");
    const contentElement = document.getElementById("result-content");

    errorElement.classList.add("hidden");
    contentElement.classList.add("hidden");
    emptyElement.classList.add("hidden");

    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";

    const formData = Object.fromEntries(
      new FormData(predictForm).entries()
    );

    const numericFields = [
      "click_duration",
      "scroll_depth",
      "mouse_movement",
      "keystrokes_detected",
      "click_frequency",
      "time_since_last_click",
      "VPN_usage",
      "proxy_usage"
    ];

    const payload = { ...formData };

    numericFields.forEach(field => {
      payload[field] = Number(formData[field]);
    });

    try {
      const response = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.detail || "Prediction request failed."
        );
      }

      renderResult(result, payload);
      showResultModal("lgbm-result-modal");
      await loadDashboard();
      await loadPredictionHistory();

    } catch (err) {
      console.error("Prediction error:", err);

      errorElement.textContent = err.message;
      errorElement.classList.remove("hidden");

    } finally {
      submitButton.disabled = false;
      submitButton.textContent = "Analyze this click";
    }
  });
}

// ============================================================
// Render Prediction
// ============================================================

function renderResult(result, payload) {
  const content = document.getElementById("result-content");
  const empty = document.getElementById("result-empty");
  const error = document.getElementById("result-error");

  empty.classList.add("hidden");
  error.classList.add("hidden");
  content.classList.remove("hidden");
  // Remove metadata from the previous prediction.
  content.querySelectorAll(".result-meta").forEach(el => el.remove());

  const probability = Number(result.probability);
  const percentage = Number.isFinite(
    Number(result.probability_percent)
  )
    ? Number(result.probability_percent)
    : probability * 100;

  const risk = String(result.risk || "minimal").toLowerCase();

  const badge = document.getElementById("verdict-badge");

  badge.textContent =
    `${result.label || "Prediction"} · ${percentage.toFixed(2)}%`;

  badge.className = `verdict risk-${risk}`;

  const meter = document.getElementById("meter-fill");

  meter.style.width = `${Math.max(0, Math.min(100, percentage))}%`;

  const riskColors = {
    minimal: "#16A34A",
    low: "#EAB308",
    medium: "#F97316",
    high: "#DC2626"
  };

  meter.style.background = riskColors[risk] || "#64748B";

  document.getElementById("meter-label").textContent =
    `${percentage.toFixed(2)}% fraud score`;
  // Display risk interpretation
  const riskDescription = {
    minimal: "The model assigns this click a minimal risk score. No strong suspicious pattern was detected.",
    low: "The model assigns this click a low risk score. Continue normal monitoring.",
    medium: "The model assigns this click a medium risk score. Further investigation may be appropriate.",
    high: "The model assigns this click a high risk score. Review this click for potentially suspicious activity."
  };

  document.getElementById("risk-description").textContent =
    riskDescription[risk] || "Risk interpretation is unavailable.";

  // Display the submitted attributes, not fabricated explanations.
  const heading = content.querySelector("h3");

  if (heading) {
    heading.textContent = "Submitted click details";
  }

  const list = document.getElementById("factors-list");
  list.replaceChildren();

  const displayFields = [
    ["Device", payload.device_type],
    ["Browser", payload.browser],
    ["Operating system", payload.operating_system],
    ["Ad position", payload.ad_position],
    ["IP reputation", payload.device_ip_reputation],
    ["Click duration", `${payload.click_duration} seconds`],
    ["Scroll depth", payload.scroll_depth],
    ["Mouse movement", payload.mouse_movement],
    ["Keystrokes", payload.keystrokes_detected],
    ["Click frequency", payload.click_frequency],
    ["Time since last click", `${payload.time_since_last_click} seconds`],
    ["VPN usage", payload.VPN_usage ? "Yes" : "No"],
    ["Proxy usage", payload.proxy_usage ? "Yes" : "No"]
  ];

  displayFields.forEach(([name, value]) => {
    const item = document.createElement("li");

    const label = document.createElement("span");
    label.className = "factor-name";
    label.textContent = name;

    const fieldValue = document.createElement("span");
    fieldValue.className = "factor-value";
    fieldValue.textContent = String(value);

    item.append(label, fieldValue);
    list.appendChild(item);
  });

  if (result.warning) {
    const warning = document.createElement("p");
    warning.className = "card-note result-meta";
    warning.textContent = result.warning;
    list.after(warning);
  }

  if (result.model) {
    const modelInfo = document.createElement("p");
    modelInfo.className = "card-note result-meta";
    modelInfo.textContent = `Model: ${result.model}`;
    list.after(modelInfo);
  }
}

// ============================================================
// Prediction History
// ============================================================

async function loadPredictionHistory() {
  const tbody = document.getElementById(
    "prediction-history-body"
  );

  if (!tbody) return;

  try {
    const response = await fetch(`${API}/api/history/recent`);

    if (!response.ok) return;

    const data = await response.json();
    historyPredictions = data.predictions || [];
    renderPredictionHistory(historyPredictions);

  } catch (error) {
    console.warn("Prediction history unavailable:", error);
  }
}

let riskDistributionChart = null;
let predictionDistributionChart = null;



function createCenterTextPlugin(total, label) {
  return {
    id: `centerText-${label.replace(/\s/g, "")}`,

    afterDraw(chart) {
      const { ctx, chartArea } = chart;

      if (!chartArea) return;

      const centerX = (chartArea.left + chartArea.right) / 2;
      const centerY = (chartArea.top + chartArea.bottom) / 2;

      ctx.save();
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      ctx.font = "bold 24px sans-serif";
      ctx.fillStyle = "#111827";
      ctx.fillText(total.toLocaleString(), centerX, centerY - 9);

      ctx.font = "12px sans-serif";
      ctx.fillStyle = "#64748b";
      ctx.fillText(label, centerX, centerY + 14);

      ctx.restore();
    }
  };
}

function updateLiveAnalytics(stats) {
  const total = Number(stats.total_predictions) || 0;
  const suspicious = Number(stats.suspicious_clicks) || 0;
  const legitimate = Number(stats.legitimate_clicks) || 0;

  document.getElementById("live-total").textContent = total;
  document.getElementById("live-suspicious").textContent = suspicious;
  document.getElementById("live-legitimate").textContent = legitimate;

  const riskCounts = stats.risk_distribution || {};

  if (riskDistributionChart) {
    riskDistributionChart.destroy();
  }

  if (predictionDistributionChart) {
    predictionDistributionChart.destroy();
  }

  const riskCanvas = document.getElementById("riskDistributionChart");
  const predictionCanvas = document.getElementById("predictionDistributionChart");

  // Prepare risk data and remove categories with zero predictions.
  const riskCategories = [
    { label: "Minimal", value: Number(riskCounts.minimal) || 0, color: "#16a34a" },
    { label: "Low", value: Number(riskCounts.low) || 0, color: "#eab308" },
    { label: "Medium", value: Number(riskCounts.medium) || 0, color: "#f97316" },
    { label: "High", value: Number(riskCounts.high) || 0, color: "#dc2626" }
  ].filter(item => item.value > 0);

  // Prepare prediction data and remove categories with zero predictions.
  const predictionCategories = [
    { label: "Potentially Suspicious", value: suspicious, color: "#dc2626" },
    { label: "Likely Legitimate", value: legitimate, color: "#16a34a" }
  ].filter(item => item.value > 0);

  function createDoughnut(canvas, categories, centerLabel) {
    if (!canvas || typeof Chart === "undefined") return null;

    return new Chart(canvas, {
      type: "doughnut",

      data: {
        labels: categories.map(item => item.label),

        datasets: [{
          data: categories.map(item => item.value),
          backgroundColor: categories.map(item => item.color),
          borderWidth: 0,
          hoverOffset: 6
        }]
      },

      plugins: [
        createCenterTextPlugin(
          categories.reduce((sum, item) => sum + item.value, 0),
          centerLabel
        )
      ],

      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "68%",

        plugins: {
          legend: {
            position: "bottom",

            labels: {
              usePointStyle: true,
              padding: 16,

              generateLabels(chart) {
                const dataset = chart.data.datasets[0];

                const sum = dataset.data.reduce(
                  (acc, value) => acc + Number(value || 0), 0
                );

                return chart.data.labels.map((label, index) => {
                  const value = Number(dataset.data[index]) || 0;
                  const percentage = sum ? (value / sum) * 100 : 0;

                  return {
                    text: `${label}: ${value} (${percentage.toFixed(1)}%)`,
                    fillStyle: dataset.backgroundColor[index],
                    strokeStyle: dataset.backgroundColor[index],
                    lineWidth: 0,
                    hidden: false,
                    index
                  };
                });
              }
            }
          },

          tooltip: {
            callbacks: {
              label(context) {
                const value = Number(context.parsed) || 0;
                const percentage = total
                  ? (value / total) * 100
                  : 0;

                return `${context.label}: ${value} (${percentage.toFixed(1)}%)`;
              }
            }
          }
        }
      }
    });
  }

  riskDistributionChart = createDoughnut(
    riskCanvas,
    riskCategories,
    "Risk levels"
  );

  predictionDistributionChart = createDoughnut(
    predictionCanvas,
    predictionCategories,
    "Predictions"
  );
}

function renderPredictionHistory(predictions) {

  const searchInput = document.getElementById("history-search");
  const statusFilter = document.getElementById("history-status-filter");
  const riskFilter = document.getElementById("history-risk-filter");

  const searchTerm = searchInput?.value.toLowerCase().trim() || "";
  const selectedStatus = statusFilter?.value || "all";
  const selectedRisk = riskFilter?.value || "all";

  predictions = predictions.filter(prediction => {
    const searchableText = [
      prediction.device_type,
      prediction.browser,
      prediction.operating_system
    ].join(" ").toLowerCase();

    const matchesSearch = searchableText.includes(searchTerm);

    const matchesStatus =
      selectedStatus === "all" ||
      prediction.label === selectedStatus;

    const matchesRisk =
      selectedRisk === "all" ||
      String(prediction.risk).toLowerCase() === selectedRisk;

    return matchesSearch && matchesStatus && matchesRisk;
  });

  const tbody = document.getElementById(
    "prediction-history-body"
  );

  if (!tbody) return;

  tbody.replaceChildren();

  if (!predictions.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");

    cell.colSpan = 6;
    cell.textContent = "No predictions yet.";

    row.appendChild(cell);
    tbody.appendChild(row);
    return;
  }

  [...predictions].reverse().forEach(prediction => {
    const row = document.createElement("tr");


    const values = [
      prediction.click_time
        ? new Date(prediction.click_time).toLocaleTimeString()
        : "—",

      prediction.device_type ?? "—",
      prediction.browser ?? "—",
      prediction.operating_system ?? "—",

      `${(Number(prediction.probability) * 100).toFixed(2)}%`
    ];

    values.forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });

    // Result column: prediction label + risk badge
    const resultCell = document.createElement("td");

    const label = document.createElement("span");
    label.textContent = prediction.label ?? "—";

    const risk = String(prediction.risk || "minimal").toLowerCase();

    const riskBadge = document.createElement("span");
    riskBadge.className = `history-risk risk-${risk}`;
    riskBadge.textContent = risk.charAt(0).toUpperCase() + risk.slice(1);

    resultCell.append(label, document.createTextNode(" "), riskBadge);
    row.appendChild(resultCell);

    tbody.appendChild(row);
  });
}

let historyPredictions = [];

function setupHistoryFilters() {
  const searchInput = document.getElementById("history-search");
  const statusFilter = document.getElementById("history-status-filter");
  const riskFilter = document.getElementById("history-risk-filter");

  if (!searchInput || !statusFilter || !riskFilter) return;

  const applyFilters = () => {
    renderPredictionHistory(historyPredictions);
  };

  searchInput.addEventListener("input", applyFilters);
  statusFilter.addEventListener("change", applyFilters);
  riskFilter.addEventListener("change", applyFilters);
}




// ============================================================
// Reset History
// ============================================================

const resetButton = document.getElementById("reset-history");

if (resetButton) {
  resetButton.addEventListener("click", async () => {
    try {
      const response = await fetch(
        `${API}/api/history/reset`,
        { method: "POST" }
      );

      if (!response.ok) {
        throw new Error("History reset endpoint unavailable.");
      }

      await loadDashboard();
      await loadPredictionHistory();

      document.getElementById("result-content")
        ?.classList.add("hidden");

      document.getElementById("result-empty")
        ?.classList.remove("hidden");

    } catch (error) {
      console.error("Reset error:", error);
      alert(error.message);
    }
  });
}

// ============================================================
// Initial Load
// ============================================================

// Initial dashboard load
setupHistoryFilters();
loadDashboard();
loadPredictionHistory();

// Automatically refresh dashboard every 30 seconds
setInterval(() => {
  loadDashboard();
  loadPredictionHistory();
}, 30000);


const c2idtlForm = document.getElementById("c2idtl-form");

if (c2idtlForm) {
  c2idtlForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const errorElement = document.getElementById("c2idtl-error");
    const resultElement = document.getElementById("c2idtl-result");
    const button = c2idtlForm.querySelector('button[type="submit"]');

    errorElement.classList.add("hidden");
    resultElement.classList.add("hidden");
    button.disabled = true;
    button.textContent = "Running C2IDTL...";

    try {
      const formData = Object.fromEntries(
        new FormData(c2idtlForm).entries()
      );

      const payload = {
        ip: Number(formData.ip),
        app: Number(formData.app),
        device: Number(formData.device),
        os: Number(formData.os),
        channel: Number(formData.channel),
        click_time: new Date(formData.click_time).toISOString()
      };
      console.log("C2IDTL payload:", payload);

      const response = await fetch(`${API}/api/c2idtl/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "C2IDTL prediction failed.");
      }


      const attribution = result.predicted_class === 1
        ? "Attributed (class 1)"
        : "Not attributed (class 0)";

      const score = Number(result.probability_percent);

      resultElement.innerHTML = `
      <div class="c2idtl-result-card">
        <p class="c2idtl-result-eyebrow">C2IDTL · ResNet50</p>
        <h3>Attribution prediction</h3>
        <div class="c2idtl-score">
         ${Number.isFinite(score) ? score.toFixed(2) : "N/A"}%
        </div>

        <div class="c2idtl-score-track">
          <div class="c2idtl-score-fill"
           style="width: ${Number.isFinite(score)
            ? Math.max(0, Math.min(100, score))
            : 0}%">
          </div>
        </div>

          <p class="c2idtl-prediction">
            <strong>Predicted class:</strong> ${attribution}
          </p>

          <p class="c2idtl-method">
            Click features → feature-importance image → ResNet50
          </p>

          <p class="c2idtl-warning">
            ${result.warning}
          </p>
      </div>
`;

      resultElement.classList.remove("hidden");
      showResultModal("c2idtl-result-modal");
      await loadC2IDTLHistory();
    } catch (error) {
      console.error("C2IDTL prediction error:", error);
      errorElement.textContent = error.message;
      errorElement.classList.remove("hidden");
    } finally {
      button.disabled = false;
      button.textContent = "Run C2IDTL Prediction";
    }
  });
}


const c2idtlHistoryBody = document.getElementById("c2idtlHistoryBody");
const c2idtlHistoryStatus = document.getElementById("c2idtlHistoryStatus");
const refreshC2IDTLHistoryButton = document.getElementById(
  "refreshC2IDTLHistory"
);

async function loadC2IDTLHistory() {
  if (!c2idtlHistoryBody || !c2idtlHistoryStatus) return;

  c2idtlHistoryStatus.textContent = "Loading C2IDTL history...";
  refreshC2IDTLHistoryButton.disabled = true;

  try {
    const response = await fetch(`${API}/api/c2idtl/history`, {
      headers: { Accept: "application/json" }
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Could not load C2IDTL history.");
    }

    c2idtlHistoryBody.replaceChildren();

    for (const record of data.predictions || []) {
      const row = document.createElement("tr");

      const values = [
        record.id,
        record.click_time
          ? new Date(record.click_time).toLocaleString()
          : "—",
        record.ip,
        record.app,
        record.device,
        record.os,
        record.channel,
        `${Number(record.probability_percent ?? record.probability * 100).toFixed(2)}%`,
        Number(record.predicted_class) === 1
          ? "Attributed (1)"
          : "Not attributed (0)"
      ];

      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = String(value ?? "—");
        row.appendChild(cell);
      }

      c2idtlHistoryBody.appendChild(row);
    }

    c2idtlHistoryStatus.textContent =
      `Showing ${data.count ?? (data.predictions || []).length} saved C2IDTL prediction(s).`;
  } catch (error) {
    c2idtlHistoryStatus.textContent = error.message;
  } finally {
    refreshC2IDTLHistoryButton.disabled = false;
  }
}

refreshC2IDTLHistoryButton?.addEventListener(
  "click",
  loadC2IDTLHistory
);

document.addEventListener("DOMContentLoaded", loadC2IDTLHistory);
