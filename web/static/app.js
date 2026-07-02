const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileDrop = document.getElementById("file-drop");
const fileLabel = document.getElementById("file-label");
const backendSelect = document.getElementById("backend-select");
const modelSizeLabel = document.getElementById("model-size-label");
const modelSizeSelect = document.getElementById("model-size-select");
const languageInput = document.getElementById("language-input");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const audioPlayer = document.getElementById("audio-player");
const fullTextEl = document.getElementById("full-text");
const segmentsEl = document.getElementById("segments");
const segmentMetaEl = document.getElementById("segment-meta");
const keywordsEl = document.getElementById("keywords");
const downloadJson = document.getElementById("download-json");
const downloadSrt = document.getElementById("download-srt");
const downloadTxt = document.getElementById("download-txt");

let currentSegments = [];

function toggleModelSizeVisibility() {
  modelSizeLabel.style.display = backendSelect.value === "whisper" ? "flex" : "none";
}
backendSelect.addEventListener("change", toggleModelSizeVisibility);
toggleModelSizeVisibility();

fileInput.addEventListener("change", () => {
  fileLabel.textContent = fileInput.files.length ? fileInput.files[0].name : "Choose an audio file (WAV, MP3, M4A, ...) or drag it here";
});

["dragenter", "dragover"].forEach((evt) =>
  fileDrop.addEventListener(evt, (e) => {
    e.preventDefault();
    fileDrop.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  fileDrop.addEventListener(evt, (e) => {
    e.preventDefault();
    fileDrop.classList.remove("dragover");
  })
);
fileDrop.addEventListener("drop", (e) => {
  const dropped = e.dataTransfer.files;
  if (dropped.length) {
    fileInput.files = dropped;
    fileLabel.textContent = dropped[0].name;
  }
});

function setStatus(message, isError = false) {
  statusEl.hidden = !message;
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function renderResults(data) {
  currentSegments = data.segments;

  audioPlayer.src = data.audio_url;
  fullTextEl.textContent = data.full_text || "(no speech detected)";
  segmentMetaEl.textContent = `${data.segments.length} segment(s) - language: ${data.language || "unknown"} - duration: ${data.duration_seconds.toFixed(1)}s`;

  segmentsEl.innerHTML = "";
  data.segments.forEach((seg, idx) => {
    const li = document.createElement("li");
    li.dataset.index = idx;
    li.dataset.start = seg.start;
    li.dataset.end = seg.end;
    li.innerHTML = `<span class="timestamp">${formatTime(seg.start)} - ${formatTime(seg.end)}</span><span>${seg.text}</span>`;
    li.addEventListener("click", () => {
      audioPlayer.currentTime = seg.start;
      audioPlayer.play();
    });
    segmentsEl.appendChild(li);
  });

  keywordsEl.innerHTML = "";
  (data.keywords || []).forEach((kw) => {
    const chip = document.createElement("span");
    chip.className = "keyword-chip";
    chip.textContent = `${kw.word} (${kw.count})`;
    keywordsEl.appendChild(chip);
  });

  downloadJson.href = data.downloads.json;
  downloadSrt.href = data.downloads.srt;
  downloadTxt.href = data.downloads.txt;

  resultsEl.hidden = false;
}

audioPlayer.addEventListener("timeupdate", () => {
  const t = audioPlayer.currentTime;
  const items = segmentsEl.querySelectorAll("li");
  items.forEach((li) => {
    const start = parseFloat(li.dataset.start);
    const end = parseFloat(li.dataset.end);
    li.classList.toggle("active", t >= start && t < end);
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("backend", backendSelect.value);
  formData.append("model_size", modelSizeSelect.value);
  formData.append("language", languageInput.value.trim());

  submitBtn.disabled = true;
  resultsEl.hidden = true;
  setStatus(
    backendSelect.value === "whisper"
      ? "Transcribing with Whisper - the first request may take a while while the model downloads..."
      : "Transcribing..."
  );

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Request failed with status ${res.status}`);
    }
    setStatus("");
    renderResults(data);
  } catch (err) {
    setStatus(err.message || "Something went wrong", true);
  } finally {
    submitBtn.disabled = false;
  }
});
