const $ = id => document.getElementById(id);
let config = null;
let currentExample = null;
let upload = { before: null, after: null };
let uploadOriginals = {}, uploadHighlights = null;
let highlightRequest = 0;
let examples = [];
let requestVersion = 0;
let sourceMode = "examples";
let uploadedResult = null;
let lastExampleId = "L492";
const fileVersions = { before: 0, after: 0 };
const labels = { L81: "Rice", L133: "Rice", L388: "Tofu", L492: "Tempeh" };

async function requestJSON(path, payload, timeout = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      signal: controller.signal,
      ...(payload === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The request failed. Please retry.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("The request timed out. Please retry; the server may still be waking up.");
    throw error;
  } finally { clearTimeout(timer); }
}

function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

function result(data) {
  document.querySelector(".result-panel").hidden = false;
  const estimate = 100 * data.estimated_fraction;
  $("estimate").textContent = Math.round(estimate);
  const known = Number.isFinite(data.recorded_fraction);
  $("reference").hidden = !known;
  if (known) {
    const recorded = data.recorded_fraction * 100;
    const error = Math.abs(estimate - recorded);
    $("recorded").textContent = recorded.toFixed(1) + "%";
    $("error").textContent = error.toFixed(1);
    $("sample-explanation").hidden = false;
    $("sample-explanation").textContent = recorded === 0
      ? `Recorded weights give 0% remaining. The ${estimate.toFixed(1)}% prediction is a model error.`
      : `The model is ${error.toFixed(1)} percentage points ${estimate >= recorded ? "above" : "below"} the recorded weight ratio.`;
    $("result-explanation").textContent = error > 20
      ? "A difficult pair: the estimate differs substantially from the recorded mass fraction. Similar appearance does not always mean similar mass."
      : "Compared with the dataset’s recorded before/after mass ratio. This pair was held out from the demo model’s calibration groups.";
  } else {
    $("sample-explanation").hidden = true;
    $("result-explanation").textContent = `${config?.mode === "hosted" ? "Your photos were analyzed on the server and were not saved." : "New photos were encoded on this machine."} Accuracy for this pair is unknown without a measured before/after mass ratio.`;
  }
}

async function analyze() {
  const version = ++requestVersion;
  clearResult();
  $("input-feedback").hidden = true;
  $("analyze").disabled = true;
  document.querySelector(".result-panel").setAttribute("aria-busy", "true");
  $("analyze").querySelector(".button-text").textContent = "Analyzing…";
  status(currentExample ? "Comparing the saved image features…" : "Analyzing photos…");
  try {
    const payload = currentExample ? { example_id: currentExample.id } : {...upload};
    if (!currentExample && (!upload.before || !upload.after)) throw new Error("Choose both a before photo and an after photo.");
    if (!currentExample && $("show-highlight").checked) highlightUploads();
    const path = !currentExample && config?.mode === "hosted" ? "/api/upload" : "/api/predict";
    const data = await requestJSON(path, payload, currentExample ? 10000 : 180000);
    if (version !== requestVersion) return;
    if (data.status === "needs_review" || data.status === "rejected") {
      $("input-feedback-title").textContent = "Choose another photo";
      $("input-issues").replaceChildren();
      for (const issue of data.input_checks.issues) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = issue.image === "after" ? "#after-input" : "#before-input";
        link.textContent = issue.message;
        link.addEventListener("click", event => { event.preventDefault(); document.querySelector(link.getAttribute("href")).focus(); });
        item.append(link); $("input-issues").append(item);
      }
      $("input-feedback").hidden = false;
      $("input-feedback").focus();
      status("");
      return;
    }
    if (!currentExample) uploadedResult = data;
    result(data);
    status("");
  } catch (error) {
    if (version === requestVersion) status(error.message, true);
  } finally {
    if (version === requestVersion) {
      $("analyze").disabled = false;
      $("analyze").querySelector(".button-text").textContent = "Analyze pair";
      document.querySelector(".result-panel").setAttribute("aria-busy", "false");
    }
  }
}

function chooseExample(example) {
  ++requestVersion;
  fileVersions.before++; fileVersions.after++;
  ++highlightRequest;
  setMode("examples");
  currentExample = example;
  lastExampleId = example.id;
  $("highlight-status").hidden = true;
  showSamplePhotos();
  document.querySelectorAll(".example").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.id === example.id)));
  analyze();
}

function showSamplePhotos() {
  if (sourceMode === "upload") {
    for (const slot of ["before", "after"]) {
      const highlighted = $("show-highlight").checked && uploadHighlights;
      const src = highlighted ? uploadHighlights[slot] : uploadOriginals[slot];
      if (src) $(slot + "-image").src = src;
      else $(slot + "-image").removeAttribute("src");
      $(slot + "-image").alt = `${slot === "before" ? "Meal before eating" : "The same meal after eating"}${highlighted ? ", with experimental food prediction highlighted in blue" : ""}`;
    }
    return;
  }
  if (!currentExample) return;
  const highlighted = $("show-highlight").checked;
  for (const slot of ["before", "after"]) {
    $(slot + "-image").src = highlighted ? `/examples/images/overlays/${currentExample.id}_${slot}.jpg` : currentExample[slot];
    $(slot + "-image").alt = `${slot === "before" ? "Meal before eating" : "The same meal after eating"}${highlighted ? ", with experimental food prediction highlighted in blue" : ""}`;
  }
}
$("show-highlight").addEventListener("change", () => {
  $("highlight-status").hidden = true;
  showSamplePhotos();
  if (sourceMode === "upload" && $("show-highlight").checked && upload.before && upload.after) highlightUploads();
});
for (const slot of ["before", "after"]) $(slot + "-image").addEventListener("error", event => {
  if (sourceMode !== "examples" || !event.target.getAttribute("src")?.includes("/overlays/")) return;
  $("show-highlight").checked = false;
  $("highlight-status").textContent = "Highlight unavailable for this sample. Showing the original photos.";
  $("highlight-status").hidden = false;
  showSamplePhotos();
});

async function highlightUploads() {
  if (config?.upload_highlights === false) return;
  if (sourceMode !== "upload" || !upload.before || !upload.after) return;
  if (uploadHighlights) { showSamplePhotos(); return; }
  const version = ++highlightRequest;
  $("highlight-status").hidden = false;
  $("highlight-status").textContent = "Highlighting food…";
  try {
    const data = await requestJSON("/api/highlight", upload, 120000);
    if (version !== highlightRequest || sourceMode !== "upload") return;
    uploadHighlights = data.images;
    showSamplePhotos();
    $("highlight-status").hidden = true;
  } catch (error) {
    if (version !== highlightRequest || sourceMode !== "upload") return;
    $("highlight-status").textContent = error.message;
  }
}

async function chooseFile(slot, file) {
  if (!file || sourceMode !== "upload") return;
  if (!["image/jpeg", "image/png"].includes(file.type) || file.size > 8 * 1024 * 1024) {
    status("Choose a JPEG or PNG smaller than 8 MB.", true); return;
  }
  const fileVersion = ++fileVersions[slot];
  let value = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read this photo."));
    reader.readAsDataURL(file);
  });
  if (config?.mode === "hosted") {
    const photo = new Image(); photo.src = value;
    await photo.decode();
    const scale = Math.min(1, 1600 / Math.max(photo.naturalWidth, photo.naturalHeight));
    if (scale < 1 || file.size > 4 * 1024 * 1024) {
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(photo.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(photo.naturalHeight * scale));
      const context = canvas.getContext("2d");
      context.fillStyle = "white"; context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(photo, 0, 0, canvas.width, canvas.height);
      value = canvas.toDataURL("image/jpeg", 0.92);
    }
  }
  if (fileVersion !== fileVersions[slot]) return;
  ++requestVersion;
  setMode("upload");
  if (currentExample) {
    currentExample = null;
    upload = { before: null, after: null };
    for (const other of ["before", "after"]) if (other !== slot) $(other + "-image").removeAttribute("src");
  }
  ++highlightRequest;
  uploadHighlights = null;
  uploadedResult = null;
  uploadOriginals[slot] = value;
  upload[slot] = value.split(",")[1];
  $("highlight-status").hidden = true;
  showSamplePhotos();
  $("input-feedback").hidden = true;
  $(slot + "-image").src = value;
  $(slot + "-image").alt = slot === "before" ? "Meal before eating" : "The same meal after eating";
  clearResult();
  $("reference").hidden = true;
  $("result-explanation").textContent = "Choose both photos, then run the model.";
  document.querySelectorAll(".example").forEach(button => button.setAttribute("aria-pressed", "false"));
  $("analyze").disabled = false;
  $("analyze").querySelector(".button-text").textContent = "Analyze pair";
  document.querySelector(".result-panel").setAttribute("aria-busy", "false");
  status("");
}

async function loadBenchmark() {
  const data = await requestJSON("/api/benchmark");
  const container = $("benchmark-charts");
  container.replaceChildren();
  const methods = [
    ["Mean", "Constant average", "Uses the training-set average for every pair."],
    ["Median", "Constant median", "Uses the training-set middle value for every pair."],
    ["Appearance", "Image appearance", "Uses image features without an explicit change calculation."],
    ["Change", "Before/after change", "Uses differences between the two images. The demo uses this model family."]
  ];
  const table = document.createElement("table"); table.className = "accuracy-table";
  const caption = document.createElement("caption"); caption.textContent = "Average error · percentage points · lower is better";
  table.append(caption);
  const head = document.createElement("thead"), header = document.createElement("tr");
  for (const title of ["Method", "LeFood", "ACETADA"]) {
    const th = document.createElement("th"); th.scope = "col"; th.textContent = title;
    if (data.summary[title]) { const small = document.createElement("span"); small.textContent = `${data.summary[title].records} pairs`; th.append(small); }
    header.append(th);
  }
  head.append(header); table.append(head);
  const body = document.createElement("tbody");
  for (const [key, title, explanation] of methods) {
    const row = document.createElement("tr"), label = document.createElement("th");
    label.scope = "row"; label.textContent = title;
    row.append(label);
    for (const name of ["LeFood", "ACETADA"]) {
      const cell = document.createElement("td");
      cell.textContent = data.summary[name].mae_percentage_points[key].toFixed(2);
      row.append(cell);
    }
    body.append(row);
  }
  table.append(body); container.append(table);
}

$("analyze").addEventListener("click", analyze);
for (const slot of ["before", "after"]) {
  $(slot + "-input").addEventListener("change", e => chooseFile(slot, e.target.files[0]).catch(e => status(e.message, true)));
  const zone = $(slot + "-zone");
  zone.addEventListener("dragover", e => { e.preventDefault(); if (sourceMode === "upload") zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("dragging"); chooseFile(slot, e.dataTransfer.files[0]).catch(e => status(e.message, true)); });
}

async function start() {
  config = await requestJSON("/api/config", undefined, 90000);
  $("mode-upload").hidden = !config.uploads;
  $("public-mode-note").hidden = config.mode !== "public";
  document.querySelector(".workspace-toolbar").hidden = !config.uploads;
  $("runtime-label").textContent = "Aftermeal";
  $("privacy-description").textContent = config.mode === "hosted"
    ? "Your photo pair is sent securely to the server for analysis, processed in memory, and not saved. Large photos are resized in your browser before upload."
    : config.mode === "public" ? "This demo uses sample photos and does not collect uploads." : "Your photos stay on this computer and are not saved.";
  if (config.mode === "hosted") $("upload-guide").textContent = "Same meal, before and after. JPG or PNG · up to 8 MB. Photos are resized, sent to the server, and not saved. Food highlights are available for sample pairs.";
  examples = await requestJSON("/api/examples");
  const order = ["L133", "L492", "L388"];
  for (const id of order) {
    const example = examples.find(item => item.id === id);
    const button = document.createElement("button"); button.type = "button"; button.className = "example"; button.dataset.id = id;
    button.setAttribute("aria-label", labels[id]);
    const thumbnails = document.createElement("span"); thumbnails.className = "example-images";
    const img = document.createElement("img"); img.src = example.after; img.alt = "";
    thumbnails.append(img);
    const name = document.createElement("span"); name.className = "example-name"; name.textContent = labels[id];
    button.append(thumbnails, name); button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => chooseExample(example));
    $("examples").append(button);
  }
  chooseExample(examples.find(item => item.id === "L492"));
  await loadBenchmark();
}
start().catch(error => status(error.message, true));

function clearResult() {
  document.querySelector(".result-panel").hidden = true;
  $("estimate").textContent = "—";
  $("reference").hidden = true;
  $("sample-explanation").hidden = true;
}

function setMode(mode) {
  sourceMode = mode;
  document.body.dataset.source = mode;
  document.querySelector(".action-row").hidden = mode !== "upload";
  for (const slot of ["before", "after"]) $(slot + "-input").disabled = mode !== "upload";
  $("mode-examples").setAttribute("aria-pressed", String(mode === "examples"));
  $("mode-upload").setAttribute("aria-pressed", String(mode === "upload"));
  $("sample-tray").hidden = mode !== "examples";
  $("upload-guide").hidden = mode !== "upload";
  $("highlight-controls").hidden = mode === "upload" && config?.upload_highlights === false;
  $("input-feedback").hidden = true;
}

$("mode-upload").addEventListener("click", () => {
  if (!config?.uploads || sourceMode === "upload") return;
  ++requestVersion;
  fileVersions.before++; fileVersions.after++;
  ++highlightRequest;
  currentExample = null;
  $("highlight-status").hidden = true;
  setMode("upload");
  showSamplePhotos();
  clearResult();
  if (uploadedResult) result(uploadedResult);
  $("analyze").disabled = false;
  $("analyze").querySelector(".button-text").textContent = "Analyze pair";
  document.querySelector(".result-panel").setAttribute("aria-busy", "false");
  status("");
});
$("mode-examples").addEventListener("click", () => {
  if (sourceMode !== "examples" && examples.length) chooseExample(examples.find(item => item.id === lastExampleId));
});

let failureLoaded = false;
let failureLoading = false;
async function loadFailure() {
  if (failureLoaded || failureLoading) return;
  failureLoading = true;
  $("failure-status").textContent = "Loading result…";
  $("retry-failure").hidden = true;
  try {
    const data = await requestJSON("/api/predict", {example_id:"L81"});
    $("failure-estimate").textContent = (data.estimated_fraction * 100).toFixed(1) + "%";
    $("failure-recorded").textContent = (data.recorded_fraction * 100).toFixed(1) + "%";
    $("failure-error").textContent = (Math.abs(data.estimated_fraction - data.recorded_fraction) * 100).toFixed(1) + " points";
    $("failure-metrics").hidden = false;
    $("failure-status").textContent = "";
    failureLoaded = true;
  } catch {
    $("failure-status").textContent = "Could not load this result.";
    $("retry-failure").hidden = false;
  } finally { failureLoading = false; }
}
$("failure-case").addEventListener("toggle", () => { if ($("failure-case").open) loadFailure(); });
$("retry-failure").addEventListener("click", loadFailure);

let activeView = null;
const viewScroll = {};
function showView(name, updateHash = true) {
  if (!["analyze", "benchmark", "about"].includes(name)) name = "analyze";
  const changed = activeView !== name;
  if (changed && activeView) viewScroll[activeView] = window.scrollY;
  document.querySelectorAll("[data-view]").forEach(tab => {
    const selected = tab.dataset.view === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $("view-" + tab.dataset.view).hidden = !selected;
  });
  document.title = `Aftermeal — ${{analyze:"Analyze", benchmark:"Accuracy", about:"About"}[name]}`;
  activeView = name;
  if (changed) window.scrollTo(0, viewScroll[name] || 0);
  if (updateHash && location.hash !== "#" + name) location.hash = name;
}
document.querySelectorAll("[data-view]").forEach((tab, index, tabs) => {
  tab.addEventListener("click", () => showView(tab.dataset.view));
  tab.addEventListener("keydown", event => {
    const keys = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index - 1 + tabs.length) % tabs.length, Home: 0, End: tabs.length - 1 };
    if (!(event.key in keys)) return;
    event.preventDefault();
    const next = tabs[keys[event.key]]; next.focus(); showView(next.dataset.view);
  });
});
document.querySelector(".skip-link").addEventListener("click", event => {
  event.preventDefault();
  $("main").focus();
});
window.addEventListener("hashchange", () => showView(location.hash.slice(1), false));
if (location.pathname === "/case-study" || location.pathname === "/web/case-study.html") {
  history.replaceState(null, "", "/#about");
}
showView(location.hash.slice(1), false);
