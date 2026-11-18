const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const COLORS = { sales: "#41d5de", redemptions: "#638dff", actual: "#e7eef4", prediction: "#41d5de" };
const HORIZON_LABELS = { 1: "15 minutes", 2: "30 minutes", 4: "1 hour", 8: "2 hours" };
let overviewData;
let modelData;
let forecastData = [];

function number(value, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function dateLabel(value, withTime = false) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, withTime
    ? { dateStyle: "medium", timeStyle: "short" }
    : { dateStyle: "medium" }).format(date);
}

async function api(path) {
  const response = await fetch(path);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setText(selector, value) { $(selector).textContent = value; }

function svgElement(name, attrs = {}, text = "") {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text) node.textContent = text;
  return node;
}

function drawAxes(svg, values, yMin = 0, yMax = null) {
  svg.replaceChildren();
  const width = 1000, height = 300, left = 64, right = 18, top = 14, bottom = 34;
  const max = yMax ?? Math.max(1, ...values.filter(Number.isFinite));
  const range = Math.max(1, max - yMin);
  const x = (index, length) => left + index * (width - left - right) / Math.max(1, length - 1);
  const y = (value) => top + (height - top - bottom) * (1 - (value - yMin) / range);
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = yMin + range * tick / 4;
    const position = y(value);
    svg.append(svgElement("line", { x1: left, x2: width - right, y1: position, y2: position, class: "chart-grid" }));
    svg.append(svgElement("text", { x: left - 9, y: position + 3, "text-anchor": "end", class: "chart-axis" }, number(value)));
  }
  return { x, y, width, height, left, right, top, bottom, max, range };
}

function drawHistory(records) {
  const svg = $("#history-chart");
  if (!records?.length) { svg.replaceChildren(svgElement("text", { x: 500, y: 145, "text-anchor": "middle", class: "chart-label" }, "No observed history in the selected range")); return; }
  const series = records.flatMap((row) => [row["Sales Count"], row["Redemption Count"]]).filter(Number.isFinite);
  const axes = drawAxes(svg, series);
  const length = records.length;
  [["Sales Count", "chart-line-sales"], ["Redemption Count", "chart-line-redemptions"]].forEach(([key, cssClass]) => {
    let segment = [];
    const flush = () => {
      if (segment.length > 1) svg.append(svgElement("polyline", { points: segment.join(" "), class: cssClass }));
      segment = [];
    };
    records.forEach((row, index) => {
      const value = row[key];
      if (value === null || !Number.isFinite(Number(value))) { flush(); return; }
      segment.push(`${axes.x(index, length)},${axes.y(Number(value))}`);
    });
    flush();
  });
  const first = records[0].Timestamp, last = records.at(-1).Timestamp;
  svg.append(svgElement("text", { x: axes.left, y: 290, class: "chart-axis" }, dateLabel(first)));
  svg.append(svgElement("text", { x: 980, y: 290, "text-anchor": "end", class: "chart-axis" }, dateLabel(last)));
}

function drawForecast(rows) {
  const svg = $("#forecast-chart");
  if (!rows.length) return;
  const values = rows.flatMap((row) => [row.lower_95, row.upper_95, row.prediction]).filter(Number.isFinite);
  const axes = drawAxes(svg, values, 0);
  const baseline = 250;
  rows.forEach((row, index) => {
    const x = axes.x(index, rows.length);
    svg.append(svgElement("line", { x1: x, x2: x, y1: axes.y(row.upper_95), y2: axes.y(row.lower_95), class: "chart-whisker95" }));
    svg.append(svgElement("line", { x1: x, x2: x, y1: axes.y(row.upper_80), y2: axes.y(row.lower_80), class: "chart-whisker80" }));
    svg.append(svgElement("circle", { cx: x, cy: axes.y(row.prediction), r: 5, class: "chart-point" }));
    svg.append(svgElement("text", { x, y: baseline, "text-anchor": "middle", class: "chart-label" }, HORIZON_LABELS[row.horizon_steps]));
  });
}

function drawBacktest(records) {
  const svg = $("#backtest-chart");
  const sampled = records.length > 400
    ? records.filter((_, index) => index % Math.ceil(records.length / 400) === 0)
    : records;
  if (!sampled.length) { svg.replaceChildren(svgElement("text", { x: 500, y: 120, "text-anchor": "middle", class: "chart-label" }, "No held-out predictions available")); return; }
  const values = sampled.flatMap((row) => [row.actual, row.prediction]).filter(Number.isFinite);
  const axes = drawAxes(svg, values);
  [["actual", "chart-line-actual"], ["prediction", "chart-line-prediction"]].forEach(([key, className]) => {
    const points = sampled.map((row, index) => `${axes.x(index, sampled.length)},${axes.y(Number(row[key]))}`).join(" ");
    svg.append(svgElement("polyline", { points, class: className }));
  });
  svg.append(svgElement("text", { x: 66, y: 19, class: "chart-label" }, `${sampled.length.toLocaleString()} sampled test points`));
}

function drawResiduals(records) {
  const svg = $("#residual-chart");
  if (!records?.length) { svg.replaceChildren(svgElement("text", { x: 500, y: 120, "text-anchor": "middle", class: "chart-label" }, "No held-out residuals available")); return; }
  const residuals = records.map((row) => Number(row.residual)).filter(Number.isFinite);
  if (!residuals.length) return;
  const minimum = Math.min(...residuals), maximum = Math.max(...residuals);
  const binCount = Math.min(24, Math.max(6, Math.ceil(Math.sqrt(residuals.length))));
  const step = Math.max((maximum - minimum) / binCount, 1);
  const bins = Array.from({ length: binCount }, () => 0);
  residuals.forEach((value) => {
    const index = Math.min(binCount - 1, Math.floor((value - minimum) / step));
    bins[index] += 1;
  });
  const axes = drawAxes(svg, bins, 0);
  const baseline = 216, barWidth = (axes.width - axes.left - axes.right) / binCount;
  bins.forEach((count, index) => {
    const x = axes.left + index * barWidth + 2;
    const y = axes.y(count);
    svg.append(svgElement("rect", { x, y, width: Math.max(1, barWidth - 4), height: baseline - y, rx: 2, fill: "rgba(65,213,222,.55)" }));
  });
  svg.append(svgElement("text", { x: axes.left, y: 238, class: "chart-axis" }, number(minimum, 1)));
  svg.append(svgElement("text", { x: 980, y: 238, "text-anchor": "end", class: "chart-axis" }, number(maximum, 1)));
  svg.append(svgElement("text", { x: 500, y: 238, "text-anchor": "middle", class: "chart-axis" }, "Residual (tickets)"));
}

function renderOverview(data) {
  setText("#metric-sales", number(data.sales_total));
  setText("#metric-redemptions", number(data.redemptions_total));
  setText("#metric-peak", number(data.peak_activity));
  setText("#metric-latest", number(data.latest_activity));
  setText("#metric-latest-sub", `${number(data.latest_sales)} sales · ${number(data.latest_redemptions)} redemptions`);
  setText("#coverage-label", `${dateLabel(data.first_observation)} — ${dateLabel(data.latest_observation)}`);
  setText("#sidebar-latest", `Latest: ${dateLabel(data.latest_observation, true)}`);
}

function renderForecast(rows) {
  forecastData = rows;
  const body = $("#forecast-table");
  const list = $("#overview-forecast");
  const risks = $("#risk-list");
  body.replaceChildren();
  list.replaceChildren();
  risks.replaceChildren();
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    [HORIZON_LABELS[row.horizon_steps], dateLabel(row.timestamp, true), number(row.prediction, 1),
      `${number(row.lower_80, 1)} – ${number(row.upper_80, 1)}`,
      `${number(row.lower_95, 1)} – ${number(row.upper_95, 1)}`, row.model]
      .forEach((value) => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
    body.append(tr);

    const mini = document.createElement("div");
    mini.className = "forecast-mini";
    const title = document.createElement("span"); title.textContent = HORIZON_LABELS[row.horizon_steps];
    const right = document.createElement("div");
    const estimate = document.createElement("strong"); estimate.textContent = number(row.prediction, 1);
    const model = document.createElement("small"); model.textContent = row.model.replaceAll("_", " ");
    right.append(estimate, document.createElement("br"), model);
    mini.append(title, right); list.append(mini);

    const risk = document.createElement("div"); risk.className = "risk-item";
    const horizon = document.createElement("strong"); horizon.textContent = HORIZON_LABELS[row.horizon_steps];
    const prediction = document.createElement("span"); prediction.textContent = `${number(row.prediction, 1)} tickets predicted`;
    const badge = document.createElement("b"); badge.className = `risk-badge ${row["Demand Level"].toLowerCase().replace(" ", "-")}`;
    badge.textContent = row["Demand Level"];
    risk.append(horizon, badge, prediction);
    if (row["Spike Warning"]) {
      const warning = document.createElement("span"); warning.textContent = "Above historical P90";
      risk.append(warning);
    }
    risks.append(risk);
  });
  drawForecast(rows);
  $("#forecast-alert").classList.add("hidden");
}

function renderModels(data) {
  modelData = data;
  setText("#validation-method", data.validation_method);
  const validationBody = $("#model-table"), testBody = $("#test-table"), rollingBody = $("#rolling-table");
  validationBody.replaceChildren(); testBody.replaceChildren(); rollingBody.replaceChildren();
  data.models.forEach((item) => {
    item.validation.forEach((score) => {
      const tr = document.createElement("tr");
      [item.target.replace(" Count", ""), HORIZON_LABELS[item.horizon_steps], score.model,
        number(score.MAE, 2), number(score.RMSE, 2),
        score.MAPE === null ? "N/A" : `${number(score.MAPE, 1)}%`,
        score.WAPE === null ? "N/A" : `${number(score.WAPE, 1)}%`]
        .forEach((value) => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
      const status = document.createElement("td");
      status.className = score.model === item.selected_model ? "selected-pill" : "not-selected";
      status.textContent = score.model === item.selected_model ? "SELECTED" : "—";
      tr.append(status); validationBody.append(tr);
    });
    const test = document.createElement("tr");
    [item.target.replace(" Count", ""), HORIZON_LABELS[item.horizon_steps], item.selected_model,
      number(item.test.MAE, 2), number(item.test.RMSE, 2),
      item.test.MAPE === null ? "N/A" : `${number(item.test.MAPE, 1)}%`,
      item.test.WAPE === null ? "N/A" : `${number(item.test.WAPE, 1)}%`]
      .forEach((value) => { const td = document.createElement("td"); td.textContent = value; test.append(td); });
    testBody.append(test);
    item.rolling_origin.forEach((fold) => {
      const tr = document.createElement("tr");
      [item.target.replace(" Count", ""), HORIZON_LABELS[item.horizon_steps], `Fold ${fold.fold}`,
        fold.model, number(fold.MAE, 2), number(fold.RMSE, 2), `${number(fold.WAPE, 1)}%`]
        .forEach((value) => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
      rollingBody.append(tr);
    });
  });
}

function renderQuality(quality) {
  setText("#quality-rows", number(quality.observed_rows));
  setText("#quality-gaps", number(quality.missing_intervals));
  setText("#quality-duplicates", number(quality.duplicate_timestamp_rows));
  setText("#quality-invalid", number(quality.invalid_timestamps + quality.invalid_or_negative_counts));
}

async function loadHistory() {
  const data = await api(`/api/history?days=${encodeURIComponent($("#history-days").value)}`);
  drawHistory(data.records);
}

async function loadForecast() {
  const alert = $("#forecast-alert");
  alert.classList.add("hidden");
  const query = new URLSearchParams({
    target: $("#forecast-target").value,
    model: $("#forecast-model").value,
  });
  if ($("#forecast-origin").value) query.set("origin", $("#forecast-origin").value.replace("T", " "));
  try {
    const data = await api(`/api/forecast?${query}`);
    renderForecast(data.forecasts);
    await loadBacktest();
  } catch (error) {
    alert.textContent = error.message;
    alert.classList.remove("hidden");
  }
}

async function loadBacktest() {
  const query = new URLSearchParams({
    target: $("#forecast-target").value,
    horizon: $("#backtest-horizon").value,
  });
  const data = await api(`/api/backtest?${query}`);
  drawBacktest(data.records);
  drawResiduals(data.records);
}

function setupForecastControls() {
  const modelSelect = $("#forecast-model");
  const models = new Set(["best"]);
  modelData.models.forEach((item) => item.validation.forEach((score) => models.add(score.model)));
  [...models].forEach((model) => {
    if (model === "best") return;
    const option = document.createElement("option"); option.value = model;
    option.textContent = model.replaceAll("_", " ");
    modelSelect.append(option);
  });
  const localValue = overviewData.latest_observation.slice(0, 16);
  $("#forecast-origin").value = localValue;
  $("#forecast-origin").max = localValue;
  $("#forecast-origin").min = modelData.minimum_forecast_origin.slice(0, 16);
}

function setSection(name) {
  $$(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.section === name));
  $$(".page-section").forEach((section) => section.classList.toggle("active", section.id === `section-${name}`));
  const titles = { overview: "Executive overview", forecast: "Predictive forecasting", models: "Model intelligence", data: "Data & methodology" };
  setText("#page-title", titles[name]);
  if (name === "forecast") loadForecast().catch(showError);
  if (name === "overview") loadHistory().catch(showError);
}

function showError(error) {
  const alert = $("#forecast-alert");
  alert.textContent = error.message;
  alert.classList.remove("hidden");
}

async function initialize() {
  try {
    [overviewData, modelData] = await Promise.all([api("/api/overview"), api("/api/models")]);
    renderOverview(overviewData);
    renderModels(modelData);
    renderQuality(overviewData.quality);
    setupForecastControls();
    await loadHistory();
    await loadBacktest();
    const initial = await api("/api/forecast?target=Sales%20Count&model=best");
    renderForecast(initial.forecasts);
  } catch (error) {
    $("#forecast-alert").textContent = error.message;
    $("#forecast-alert").classList.remove("hidden");
    $("#overview-forecast").textContent = `Unable to load analytics: ${error.message}`;
  }
}

$$(".nav-link").forEach((button) => button.addEventListener("click", () => setSection(button.dataset.section)));
$$("[data-go]").forEach((button) => button.addEventListener("click", () => setSection(button.dataset.go)));
$("#history-days").addEventListener("change", () => loadHistory().catch(showError));
$("#refresh-overview").addEventListener("click", () => initialize().catch(showError));
$("#run-forecast").addEventListener("click", () => loadForecast());
$("#forecast-target").addEventListener("change", () => loadForecast());
$("#forecast-model").addEventListener("change", () => loadForecast());
$("#forecast-origin").addEventListener("change", () => loadForecast());
$("#backtest-horizon").addEventListener("change", () => loadBacktest().catch(showError));
initialize();
