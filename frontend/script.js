const API = ""; // same origin

const V2_FEATURE_GROUPS = [
  {
    label: "Basic categorical IDs",
    features: ["ip", "app", "device", "os", "channel"]
  },
  {
    label: "Historical count features",
    features: [
      "ip_prev_clicks",
      "app_prev_clicks",
      "device_prev_clicks",
      "os_prev_clicks",
      "channel_prev_clicks"
    ]
  },
  {
    label: "IP interaction counts",
    features: [
      "ip_app_prev_clicks",
      "ip_device_prev_clicks",
      "ip_os_prev_clicks",
      "ip_channel_prev_clicks"
    ]
  },
  {
    label: "Previous-click time gaps",
    features: [
      "seconds_since_prev_ip_click",
      "seconds_since_prev_ip_app_click",
      "seconds_since_prev_ip_device_click"
    ]
  },
  {
    label: "Temporal features",
    features: ["hour", "day", "day_of_week"]
  }
];

function groupModelFeatures(features) {
  const ungrouped = new Set(features);
  const groups = V2_FEATURE_GROUPS.map(group => {
    const included = group.features.filter(feature => {
      if (!ungrouped.has(feature)) return false;
      ungrouped.delete(feature);
      return true;
    });

    return { label: group.label, features: included };
  });

  if (ungrouped.size > 0) {
    groups.push({
      label: "Other model features",
      features: [...ungrouped]
    });
  }

  return groups.filter(group => group.features.length > 0);
}

// ============================================================
// Tabs
// ============================================================

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {

    document.querySelectorAll(".tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });

    document.querySelectorAll(".panel-view").forEach(p => {
      p.classList.remove("active");
    });

    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    document
      .getElementById(tab.dataset.tab)
      .classList.add("active");
  });
});


// ============================================================
// Dashboard
// ============================================================

async function loadDashboard() {

  try {

    const res = await fetch(`${API}/api/stats`);

    if (!res.ok) {
      throw new Error("Backend not ready");
    }

    const stats = await res.json();

    document
      .getElementById("dashboard-loading")
      .classList.add("hidden");

    document
      .getElementById("dashboard-content")
      .classList.remove("hidden");


    // --------------------------------------------------------
    // Model information
    // --------------------------------------------------------

    const modelName =
      document.getElementById("best-model-name");

    if (modelName) {
      modelName.textContent = stats.model;
    }


    const modelSub =
      document.getElementById("best-model-sub");

    if (modelSub) {

      modelSub.textContent =
        `ROC-AUC ${stats.performance.roc_auc.toFixed(4)} · ` +
        `PR-AUC ${stats.performance.pr_auc.toFixed(4)}`;
    }


    // --------------------------------------------------------
    // Dataset statistics
    // --------------------------------------------------------

    const rows =
      document.getElementById("stat-rows");

    if (rows) {
      rows.textContent = "99,999";
    }


    const fraudRate =
      document.getElementById("stat-fraud-rate");

    if (fraudRate) {

      // This is attribution rate, NOT confirmed fraud rate.
      fraudRate.textContent = "0.227%";
    }


    const cmModel =
      document.getElementById("cm-model-name");

    if (cmModel) {
      cmModel.textContent = stats.model;
    }


    // --------------------------------------------------------
    // V2 performance chart
    // --------------------------------------------------------

    renderModelChart(stats);


    // --------------------------------------------------------
    // Feature information
    // --------------------------------------------------------

    renderFeatureChart(stats.features);
    renderFeatureGroups(stats.features);


    // --------------------------------------------------------
    // Results table
    // --------------------------------------------------------

    renderResultsTable(stats);


  } catch (e) {

    console.error(e);

    document
      .getElementById("dashboard-loading")
      .classList.add("hidden");

    document
      .getElementById("dashboard-error")
      .classList.remove("hidden");
  }
}


// ============================================================
// V2 Performance Chart
// ============================================================

function renderModelChart(stats) {

  const canvas =
    document.getElementById("modelChart");

  if (!canvas) {
    return;
  }

  new Chart(canvas, {

    type: "bar",

    data: {

      labels: [
        "ROC-AUC",
        "PR-AUC",
        "Precision",
        "Recall",
        "F1"
      ],

      datasets: [
        {
          label: "V2 Random Forest",
          data: [
            stats.performance.roc_auc * 100,
            stats.performance.pr_auc * 100,
            stats.performance.precision * 100,
            stats.performance.recall * 100,
            stats.performance.f1 * 100
          ],
          backgroundColor: "#161B27"
        }
      ]
    },

    options: {

      responsive: true,

      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          grid: {
            color: "#EEF1F5"
          }
        },

        x: {
          grid: {
            display: false
          }
        }
      },

      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 10,
            font: {
              family: "Inter"
            }
          }
        }
      }
    }
  });
}


// ============================================================
// Feature Chart
// ============================================================

function renderFeatureChart(features) {

  const canvas =
    document.getElementById("importanceChart");

  if (!canvas) {
    return;
  }

  const groups = groupModelFeatures(features);

  new Chart(canvas, {

    type: "bar",

    data: {

      labels: groups.map(group => group.label),

      datasets: [
        {
          label: "Number of features",
          data: groups.map(group => group.features.length),
          backgroundColor: "#161B27"
        }
      ]
    },

    options: {

      indexAxis: "y",

      responsive: true,

      scales: {

        x: {
          beginAtZero: true,
          ticks: {
            stepSize: 1
          },
          grid: {
            color: "#EEF1F5"
          }
        },

        y: {
          grid: {
            display: false
          }
        }

      },

      plugins: {

        legend: {
          display: false
        }

      }

    }

  });
}


function renderFeatureGroups(features) {
  const container = document.getElementById("feature-group-grid");

  if (!container) return;

  container.replaceChildren();

  groupModelFeatures(features).forEach(group => {
    const card = document.createElement("div");
    card.className = "cm-cell";

    const count = document.createElement("span");
    count.className = "cm-count";
    count.textContent = String(group.features.length);

    const label = document.createElement("span");
    label.className = "cm-label";
    label.textContent = group.label;

    const list = document.createElement("ul");
    list.className = "feature-group-features";

    group.features.forEach(feature => {
      const item = document.createElement("li");
      item.textContent = feature;
      list.appendChild(item);
    });

    card.append(count, label, list);
    container.appendChild(card);
  });
}


// ============================================================
// Results Table
// ============================================================

function renderResultsTable(stats) {

  const tbody =
    document.querySelector("#results-table tbody");

  if (!tbody) {
    return;
  }


  tbody.replaceChildren();

  (stats.comparison || []).forEach(result => {
    const row = document.createElement("tr");
    const values = [
      result.version,
      result.model,
      `${(result.precision * 100).toFixed(2)}%`,
      `${(result.recall * 100).toFixed(2)}%`,
      `${(result.f1 * 100).toFixed(2)}%`,
      result.roc_auc.toFixed(4),
      result.pr_auc.toFixed(4)
    ];

    values.forEach(value => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    });

    if (result.deployed) {
      row.classList.add("deployed-model-row");
    }

    tbody.appendChild(row);
  });
}


// ============================================================
// Live Predictor
// ============================================================

const predictForm =
  document.getElementById("predict-form");


if (predictForm) {

  predictForm.addEventListener(
    "submit",
    async (e) => {

      e.preventDefault();

      const form = e.target;

      const data =
        Object.fromEntries(
          new FormData(form).entries()
        );


      const payload = {

        ip: Number(data.ip),

        app: Number(data.app),

        device: Number(data.device),

        os: Number(data.os),

        channel: Number(data.channel),

        click_time: data.click_time
      };


      document
        .getElementById("result-empty")
        .classList.add("hidden");

      document
        .getElementById("result-error")
        .classList.add("hidden");


      try {

        const res = await fetch(
          `${API}/api/predict`,
          {
            method: "POST",

            headers: {
              "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)
          }
        );


        if (!res.ok) {

          const error =
            await res.json();

          throw new Error(
            error.detail ||
            "Prediction failed"
          );
        }


        const result =
          await res.json();


        renderResult(result);


      } catch (err) {

        document
          .getElementById("result-content")
          .classList.add("hidden");

        const errEl =
          document.getElementById("result-error");

        errEl.textContent =
          err.message;

        errEl.classList.remove("hidden");
      }
    }
  );
}


// ============================================================
// Render Prediction
// ============================================================

function renderResult(result) {

  const resultContent =
    document.getElementById("result-content");

  const resultEmpty =
    document.getElementById("result-empty");

  const resultError =
    document.getElementById("result-error");

  resultEmpty.classList.add("hidden");
  resultError.classList.add("hidden");
  resultContent.classList.remove("hidden");


  // ----------------------------------------------------------
  // Verdict
  // ----------------------------------------------------------

  const badge =
    document.getElementById("verdict-badge");

  const probability =
    Number(result.probability) || 0;

  const risk =
    String(result.risk || "minimal");

  const label =
    String(result.label || "Prediction");


  badge.textContent =
    `${label} · ${(probability * 100).toFixed(1)}%`;

  badge.className =
    `verdict risk-${risk}`;


  // ----------------------------------------------------------
  // Probability meter
  // ----------------------------------------------------------

  const pct =
    Math.round(probability * 100);

  const meter =
    document.getElementById("meter-fill");

  meter.style.width =
    `${pct}%`;

  meter.style.background =
    risk === "minimal"
      ? "#0F7B6C"
      : risk === "low"
        ? "#C98A1F"
        : "#B3261E";


  document
    .getElementById("meter-label")
    .textContent =
      `${pct}% attribution probability`;


  // ----------------------------------------------------------
  // Behavioral signals
  // ----------------------------------------------------------

  const list =
    document.getElementById("factors-list");

  const signals =
    Array.isArray(result.behavioral_signals)
      ? result.behavioral_signals
      : [];


  if (signals.length === 0) {

    list.innerHTML = `
      <li>
        <span class="factor-name">
          No behavioral history available
        </span>
        <span class="factor-value">
          —
        </span>
      </li>
    `;

    return;
  }


  list.innerHTML =
    signals.map(signal => {

      const description =
        String(
          signal.description ||
          signal.feature ||
          "Behavioral signal"
        );

      const value =
        Number(signal.value);


      return `
        <li>
          <span class="factor-name">
            ${description}
          </span>

          <span class="factor-value">
            ${
              Number.isFinite(value)
                ? formatSignalValue(
                    signal.feature || "",
                    value
                  )
                : "—"
            }
          </span>
        </li>
      `;

    }).join("");
}


// ============================================================
// Format behavioral values
// ============================================================

function formatSignalValue(
  feature,
  value
) {

  if (
    feature.startsWith(
      "seconds_since"
    )
  ) {

    if (value < 0) {
      return "No previous click";
    }

    if (value < 60) {
      return `${value.toFixed(0)} sec`;
    }

    if (value < 3600) {
      return `${(value / 60).toFixed(1)} min`;
    }

    return `${(value / 3600).toFixed(1)} hr`;
  }


  return value.toFixed(0);
}


// ============================================================
// Initial load
// ============================================================

loadDashboard();

// ============================================================
// Prediction History
// ============================================================

async function loadPredictionHistory() {

  try {

    const res =
      await fetch(`${API}/api/history/recent`);

    if (!res.ok) {
      return;
    }

    const data =
      await res.json();

    renderPredictionHistory(
      data.predictions || []
    );

  } catch (error) {

    console.error(
      "Could not load prediction history:",
      error
    );
  }
}


function renderPredictionHistory(
  predictions
) {

  const tbody =
    document.getElementById(
      "prediction-history-body"
    );

  if (!tbody) {
    return;
  }


  if (predictions.length === 0) {

    tbody.innerHTML = `
      <tr>
        <td colspan="6">
          No predictions yet.
        </td>
      </tr>
    `;

    return;
  }


  // Most recent first
  const recent =
    [...predictions].reverse();


  tbody.innerHTML =
    recent.map(prediction => {

      const probability =
        Number(prediction.probability) || 0;

      const pct =
        (probability * 100).toFixed(1);

      const time =
        new Date(
          prediction.click_time
        ).toLocaleTimeString();


      return `
        <tr>

          <td>${time}</td>

          <td>${prediction.ip}</td>

          <td>${prediction.app}</td>

          <td>${prediction.device}</td>

          <td>${pct}%</td>

          <td>
            <span class="history-risk risk-${prediction.risk}">
              ${prediction.label}
            </span>
          </td>

        </tr>
      `;

    }).join("");
}


// Load history when page opens
loadPredictionHistory();


// Refresh history after every prediction
const originalPredictForm =
  document.getElementById("predict-form");

if (originalPredictForm) {

  originalPredictForm.addEventListener(
    "submit",
    () => {

      // Give the prediction request time to finish.
      setTimeout(
        loadPredictionHistory,
        300
      );

    }
  );
}


// Reset history
const resetButton =
  document.getElementById("reset-history");

if (resetButton) {

  resetButton.addEventListener(
    "click",
    async () => {

      try {

        const res =
          await fetch(
            `${API}/api/history/reset`,
            {
              method: "POST"
            }
          );

        if (!res.ok) {
          throw new Error(
            "Could not reset history"
          );
        }

        await loadPredictionHistory();

        // Clear the current result display.
        document
          .getElementById("result-content")
          ?.classList.add("hidden");

        document
          .getElementById("result-empty")
          ?.classList.remove("hidden");

      } catch (error) {

        console.error(error);

      }

    }
  );
}
