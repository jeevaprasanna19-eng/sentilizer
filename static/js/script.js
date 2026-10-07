/* NLP Sentiment Analyzer - frontend logic (vanilla JS, Fetch API) */
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  input: $("feedbackInput"), charCount: $("charCount"),
  analyzeBtn: $("analyzeBtn"), clearBtn: $("clearBtn"),
  uploadBtn: $("uploadBtn"), fileInput: $("fileInput"),
  loading: $("loading"), loadingText: $("loadingText"), errorBox: $("errorBox"),
  resultCard: $("resultCard"), resultSentiment: $("resultSentiment"),
  resultConfidence: $("resultConfidence"), confidenceBar: $("confidenceBar"),
  resultText: $("resultText"), resultScore: $("resultScore"),
  uploadResults: $("uploadResults"), uploadSummary: $("uploadSummary"),
  uploadTableBody: $("uploadTableBody"),
  historyBody: $("historyBody"), historySearch: $("historySearch"),
  historyFilter: $("historyFilter"), clearHistoryBtn: $("clearHistoryBtn"),
};

const BADGE_CLASS = { Positive: "badge-positive", Negative: "badge-negative", Neutral: "badge-neutral" };
const MAX_UPLOAD_MB = 2;   // keep in sync with models/model_config.py
let historyData = [];

/* ---------- Helpers ---------- */
function formatPercent(value) {
  return (Number(value) * 100).toFixed(1) + "%";
}

function createBadge(sentiment) {
  const span = document.createElement("span");
  span.className = "badge " + (BADGE_CLASS[sentiment] || "");
  span.textContent = sentiment;   // textContent => text is never interpreted as HTML
  return span;
}

function createCell(label, content) {
  const td = document.createElement("td");
  td.dataset.label = label;
  if (content instanceof Node) td.appendChild(content);
  else td.textContent = content;
  return td;
}

/** fetch wrapper: handles network errors and server error messages. */
async function apiRequest(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    throw new Error("Cannot reach the server. Please check that the app is running and try again.");
  }
  let data = null;
  try { data = await response.json(); } catch (err) { /* non-JSON response */ }
  if (!response.ok) {
    throw new Error((data && data.error) || "Something went wrong. Please try again.");
  }
  return data;
}

/* ---------- UI state ---------- */
function showLoading(isLoading, message = "Analyzing with BERT...") {
  els.loading.hidden = !isLoading;
  els.loadingText.textContent = message;
  els.analyzeBtn.disabled = isLoading;
  els.uploadBtn.disabled = isLoading;
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.hidden = false;
}

function hideMessages() {
  els.errorBox.hidden = true;
}

function showResult(data) {
  const sentiment = data.sentiment;
  els.resultCard.dataset.sentiment = sentiment;
  els.resultSentiment.className = "badge " + (BADGE_CLASS[sentiment] || "");
  els.resultSentiment.textContent = sentiment;
  els.resultConfidence.textContent = formatPercent(data.confidence);
  els.confidenceBar.style.width = "0%";
  els.resultText.textContent = data.text;
  els.resultScore.textContent = data.score !== undefined ? data.score : "-";
  els.uploadResults.hidden = true;
  els.resultCard.hidden = false;
  // Animate the bar after the card is visible
  requestAnimationFrame(() => { els.confidenceBar.style.width = (data.confidence * 100).toFixed(1) + "%"; });
}

/* ---------- Actions ---------- */
async function analyzeSentiment() {
  hideMessages();
  const text = els.input.value.trim();
  if (!text) {
    showError("Please enter some feedback before analyzing.");
    return;
  }
  showLoading(true, "Analyzing with BERT...");
  try {
    const data = await apiRequest("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    showResult(data);
    await loadHistory();   // refreshes the table and statistics
  } catch (err) {
    showError(err.message);
  } finally {
    showLoading(false);
  }
}

async function uploadFeedback() {
  hideMessages();
  const file = els.fileInput.files[0];
  if (!file) return;

  const name = file.name.toLowerCase();
  if (!name.endsWith(".csv") && !name.endsWith(".txt")) {
    showError("Unsupported file type. Please upload a .csv or .txt file.");
    els.fileInput.value = "";
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    showError(`The file is too large. The maximum size is ${MAX_UPLOAD_MB} MB.`);
    els.fileInput.value = "";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  showLoading(true, "Analyzing file with BERT...");
  try {
    const data = await apiRequest("/upload", { method: "POST", body: formData });
    renderUploadResults(data);
    await loadHistory();
  } catch (err) {
    showError(err.message);
  } finally {
    els.fileInput.value = "";   // allow re-uploading the same file
    showLoading(false);
  }
}

function renderUploadResults(data) {
  els.resultCard.hidden = true;
  els.uploadTableBody.replaceChildren();
  data.results.forEach((row) => {
    const tr = document.createElement("tr");
    tr.append(
      createCell("Feedback", row.feedback),
      createCell("Sentiment", createBadge(row.sentiment)),
      createCell("Confidence", formatPercent(row.confidence)),
    );
    els.uploadTableBody.appendChild(tr);
  });
  const s = data.summary;
  els.uploadSummary.textContent =
    `Analyzed ${data.count} entries - Positive: ${s.Positive}, Negative: ${s.Negative}, Neutral: ${s.Neutral}`;
  els.uploadResults.hidden = false;
}

async function loadHistory() {
  try {
    const data = await apiRequest("/history");
    historyData = data.history;
    updateStatistics(data.stats);
    filterHistory();
  } catch (err) {
    showError(err.message);
  }
}

function updateStatistics(stats) {
  $("statTotal").textContent = stats.total;
  $("statPositive").textContent = stats.Positive;
  $("statNegative").textContent = stats.Negative;
  $("statNeutral").textContent = stats.Neutral;
}

function filterHistory() {
  const query = els.historySearch.value.trim().toLowerCase();
  const filter = els.historyFilter.value;
  const rows = historyData.filter((r) =>
    (filter === "all" || r.sentiment === filter) &&
    (!query || r.feedback.toLowerCase().includes(query))
  );

  els.historyBody.replaceChildren();
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "empty";
    td.textContent = historyData.length ? "No feedback matches your search." : "No feedback analyzed yet.";
    tr.appendChild(td);
    els.historyBody.appendChild(tr);
    return;
  }
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.append(
      createCell("Timestamp", r.timestamp),
      createCell("Feedback", r.feedback),
      createCell("Sentiment", createBadge(r.sentiment)),
      createCell("Confidence", formatPercent(r.confidence)),
    );
    els.historyBody.appendChild(tr);
  });
}

async function clearHistory() {
  if (!window.confirm("Delete all feedback history? This cannot be undone.")) return;
  hideMessages();
  try {
    await apiRequest("/clear-history", { method: "POST" });
    await loadHistory();
  } catch (err) {
    showError(err.message);
  }
}

/* ---------- Theme ---------- */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem("theme", theme); } catch (e) { /* storage unavailable */ }
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) { /* ignore */ }
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(saved || (prefersDark ? "dark" : "light"));
}

/* ---------- Event wiring ---------- */
els.analyzeBtn.addEventListener("click", analyzeSentiment);
els.clearBtn.addEventListener("click", () => {
  els.input.value = "";
  els.charCount.textContent = "0";
  els.resultCard.hidden = true;
  els.uploadResults.hidden = true;
  hideMessages();
  els.input.focus();
});
els.input.addEventListener("input", () => { els.charCount.textContent = els.input.value.length; });
els.input.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") analyzeSentiment();
});
els.uploadBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", uploadFeedback);
els.historySearch.addEventListener("input", filterHistory);
els.historyFilter.addEventListener("change", filterHistory);
els.clearHistoryBtn.addEventListener("click", clearHistory);
$("navAnalyze").addEventListener("click", () => $("analyzeSection").scrollIntoView({ behavior: "smooth" }));
$("navHistory").addEventListener("click", () => $("historySection").scrollIntoView({ behavior: "smooth" }));
$("themeToggle").addEventListener("click", () => {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
});

initTheme();
loadHistory();