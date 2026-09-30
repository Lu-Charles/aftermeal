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
    if (error.name === "AbortError") throw new Error("The request timed out. Please retry; local first-use model setup may still be running.");
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
    $("result-explanation").textContent = "New photos were encoded on this machine. Accuracy for this pair is unknown without a measured before/after mass ratio.";
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
    const payload = currentExample ? { example_id: currentExample.id } : {...upload, starting_portion:$("starting-portion").value||null};
    if (!currentExample && (!upload.before || !upload.after)) throw new Error("Choose both a before photo and an after photo.");
    if (!currentExample && $("show-highlight").checked) highlightUploads();
    const data = await requestJSON("/api/predict", payload, currentExample ? 10000 : 120000);

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
  if (sourceMode !== "upload" || !upload.before || !upload.after) return;
  if (uploadHighlights) { showSamplePhotos(); return; }
  const version = ++highlightRequest;
  $("highlight-status").hidden = false;
  $("highlight-status").textContent = "Highlighting food…";
  try {
    const data = await requestJSON("/api/highlight", upload, 120000);

    uploadHighlights = data.images;
    showSamplePhotos();
    $("highlight-status").hidden = true;
  } catch (error) {

    $("highlight-status").textContent = error.message;
  }
}

async function chooseFile(slot, file) {
  if (!file || sourceMode !== "upload") return;
  if (!["image/jpeg", "image/png"].includes(file.type) || file.size > 8 * 1024 * 1024) {
    status("Choose a JPEG or PNG smaller than 8 MB.", true); return;
  }
  const fileVersion = ++fileVersions[slot];
  const value = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read this photo."));
    reader.readAsDataURL(file);
  });

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


$("analyze").addEventListener("click", analyze);
for (const slot of ["before", "after"]) {
  $(slot + "-input").addEventListener("change", e => chooseFile(slot, e.target.files[0]).catch(e => status(e.message, true)));
  const zone = $(slot + "-zone");
  zone.addEventListener("dragover", e => { e.preventDefault(); if (sourceMode === "upload") zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", e => { e.preventDefault(); zone.classList.remove("dragging"); chooseFile(slot, e.dataTransfer.files[0]).catch(e => status(e.message, true)); });
}

async function start() {
  config = await requestJSON("/api/config");
  $("mode-upload").hidden = !config.uploads;
  $("public-mode-note").hidden = config.mode !== "public";
  document.querySelector(".workspace-toolbar").hidden = !config.uploads;
  $("runtime-label").textContent = "Aftermeal";
  $("privacy-description").textContent = "Sample predictions use bundled features. Photo inference, when enabled in the local service, runs on this computer. The separate collection tool explicitly saves its records locally.";
  examples = await requestJSON("/api/examples");
  const order = ["L133", "L492", "L388", "L81"];
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
  $("starting-control").hidden = mode !== "upload";
  sourceMode = mode;
  document.body.dataset.source = mode;
  document.querySelector(".action-row").hidden = mode !== "upload";
  for (const slot of ["before", "after"]) $(slot + "-input").disabled = mode !== "upload";
  $("mode-examples").setAttribute("aria-pressed", String(mode === "examples"));
  $("mode-upload").setAttribute("aria-pressed", String(mode === "upload"));
  $("sample-tray").hidden = mode !== "examples";
  $("upload-guide").hidden = mode !== "upload";
  $("highlight-controls").hidden = false;
  $("input-feedback").hidden = true;
}

$("mode-upload").addEventListener("click", () => {
  if (!config?.uploads || sourceMode === "upload") return;
  ++requestVersion;
  fileVersions.before++; fileVersions.after++;
  ++highlightRequest;
  currentExample = null;
  $("highlight-status").hidden = true;
  upload = {before:null,after:null}; uploadOriginals = {}; uploadHighlights = null; uploadedResult = null;
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
  if (sourceMode !== "examples" && examples.length) chooseExample(examples.find(item => item.id === "L492"));
});


async function loadBenchmark() { const data=await requestJSON('/api/benchmark'); const target=$('benchmark-charts');target.replaceChildren();for(const [source,metrics] of Object.entries(data.summary)){const heading=document.createElement('h2');heading.textContent=source;target.append(heading);for(const [method,error] of Object.entries(metrics.mae_percentage_points)){const row=document.createElement('p');row.textContent=`${method}: ${error.toFixed(2)} pp`;const bar=document.createElement('progress');bar.max=100;bar.value=error;row.append(bar);target.append(row);}}}
