/**
 * EDiTH Frontend — Application Logic
 *
 * Handles image upload, API communication with the FastAPI backend,
 * and dynamic rendering of analysis results.
 */

// ============================================================
// Configuration
// Dynamic API base: uses origin if served directly from backend, or defaults to port 8000
const API_BASE = (window.location.protocol.startsWith("http") && (window.location.port === "8000" || window.location.port === ""))
    ? `${window.location.origin}/api`
    : "http://127.0.0.1:8000/api";

const DR_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];
const SEVERITY_COLORS = ["#34d399", "#fbbf24", "#fb923c", "#f87171", "#dc2626"];

// ============================================================
// DOM Elements
// ============================================================
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("previewImage");
const previewFilename = document.getElementById("previewFilename");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const uploadSection = document.getElementById("uploadSection");
const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");
const retryBtn = document.getElementById("retryBtn");
const newAnalysisBtn = document.getElementById("newAnalysisBtn");
const serverStatus = document.getElementById("serverStatus");

let selectedFile = null;
let originalImageDataUrl = null;
let gradcamImageDataUrl = null;
let loadingInterval = null;

// ============================================================
// Server Health Check
// ============================================================
async function checkServerHealth() {
    const dot = serverStatus.querySelector(".status-dot");
    const text = serverStatus.querySelector(".status-text");

    try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();

        dot.className = "status-dot status-dot--online";
        const loadedModels = Object.entries(data.models || {})
            .filter(([, v]) => v)
            .map(([k]) => k);
        text.textContent = loadedModels.length > 0
            ? `Online • ${loadedModels.length} model(s) loaded`
            : "Online • No models loaded";
    } catch {
        dot.className = "status-dot status-dot--offline";
        text.textContent = "Server offline";
    }
}

// ============================================================
// Upload Handling
// ============================================================
uploadZone.addEventListener("click", () => fileInput.click());

// Reset file input value on click so re-uploading the same file fires change event
fileInput.addEventListener("click", (e) => {
    e.target.value = "";
});

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith("image/")) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (file.size > 20 * 1024 * 1024) {
        alert("File too large. Maximum size is 20MB.");
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        originalImageDataUrl = e.target.result;
        previewImage.src = e.target.result;
        previewFilename.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        uploadZone.style.display = "none";
        preview.style.display = "flex";
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", resetUpload);

const ICON_ANALYZE = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/><line x1="12" y1="3" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="21"/><line x1="3" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="21" y2="12"/></svg>';
const ICON_SPINNER = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="animation: spin 0.8s linear infinite;"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>';

function resetUpload() {
    selectedFile = null;
    originalImageDataUrl = null;
    gradcamImageDataUrl = null;
    fileInput.value = "";
    previewImage.src = "";
    previewFilename.textContent = "";

    uploadZone.style.display = "flex";
    preview.style.display = "none";
    uploadSection.style.display = "block";
    loadingSection.style.display = "none";
    resultsSection.style.display = "none";
    errorSection.style.display = "none";

    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `<span class="btn__icon">${ICON_ANALYZE}</span> Analyze`;

    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
}

// ============================================================
// Analysis
// ============================================================
analyzeBtn.addEventListener("click", startAnalysis);
retryBtn.addEventListener("click", resetUpload);
newAnalysisBtn.addEventListener("click", resetUpload);

const topNewAnalysisBtn = document.getElementById("topNewAnalysisBtn");
if (topNewAnalysisBtn) {
    topNewAnalysisBtn.addEventListener("click", resetUpload);
}

async function startAnalysis() {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `<span class="btn__icon">${ICON_SPINNER}</span> Analyzing...`;

    // Show loading
    uploadSection.style.display = "none";
    loadingSection.style.display = "block";
    resultsSection.style.display = "none";
    errorSection.style.display = "none";

    // Animate loading steps
    animateLoadingSteps();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error: ${res.status}`);
        }

        const data = await res.json();
        renderResults(data);

    } catch (err) {
        showError(err.message || "Failed to connect to the server. Is the backend running?");
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<span class="btn__icon">${ICON_ANALYZE}</span> Analyze`;
        if (loadingInterval) {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }
}

function animateLoadingSteps() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }

    const steps = document.querySelectorAll(".loading-step");
    steps.forEach((s) => {
        s.classList.remove("active", "done");
    });

    let current = 0;
    loadingInterval = setInterval(() => {
        if (current > 0) {
            steps[current - 1].classList.remove("active");
            steps[current - 1].classList.add("done");
        }
        if (current < steps.length) {
            steps[current].classList.add("active");
            current++;
        } else {
            clearInterval(loadingInterval);
            loadingInterval = null;
        }
    }, 1200);
}

function showError(message) {
    loadingSection.style.display = "none";
    errorSection.style.display = "block";
    errorMessage.textContent = message;
}

// ============================================================
// Render Results
// ============================================================
function renderResults(data) {
    loadingSection.style.display = "none";
    resultsSection.style.display = "flex";

    renderQuality(data.quality);
    renderClassification(data.classification);
    renderGradCAM(data.gradcam_image);
    renderSimilarCases(data.similar_cases);
    renderReport(data.report);

    const processingTime = document.getElementById("processingTime");
    processingTime.textContent = data.processing_time_s
        ? `Processed in ${data.processing_time_s}s`
        : "";
}

// --- Quality ---
function renderQuality(quality) {
    const container = document.getElementById("qualityResult");
    if (!quality) {
        container.innerHTML = '<p class="cases-empty">Quality assessment unavailable</p>';
        return;
    }

    const badgeClass = quality.gradable ? "quality-badge--pass" : "quality-badge--fail";
    const badgeIcon = quality.gradable
        ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 5px;"><polyline points="20 6 9 17 4 12"/></svg>'
        : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right: 5px;"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    const badgeLabel = quality.gradable ? "Gradable" : "Ungradable";

    let issuesHtml = "";
    if (quality.issues && quality.issues.length > 0) {
        issuesHtml = quality.issues
            .map((issue) => `<li style="color: var(--warning); font-size: var(--font-size-sm);">${issue}</li>`)
            .join("");
        issuesHtml = `<ul style="margin-top: var(--space-md); list-style: none; padding: 0;">${issuesHtml}</ul>`;
    }

    container.innerHTML = `
        <div class="quality-badge ${badgeClass}">${badgeIcon}${badgeLabel}</div>
        <div class="quality-details">
            <div class="quality-stat">
                <span class="quality-stat__label">Confidence</span>
                <span class="quality-stat__value">${(quality.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="quality-stat">
                <span class="quality-stat__label">Blur Score</span>
                <span class="quality-stat__value">${quality.blur_score ?? "N/A"}</span>
            </div>
            <div class="quality-stat">
                <span class="quality-stat__label">Brightness</span>
                <span class="quality-stat__value">${quality.mean_brightness ?? "N/A"}</span>
            </div>
        </div>
        ${issuesHtml}
    `;
}

// --- Classification ---
function renderClassification(classification) {
    const container = document.getElementById("classificationResult");

    if (!classification || classification.grade < 0) {
        container.innerHTML = '<p class="cases-empty">Classification model not loaded</p>';
        return;
    }

    const grade = classification.grade;
    const label = classification.label;
    const confidence = classification.confidence;
    const probs = classification.probabilities || [];

    // Probability bars
    let probBarsHtml = probs
        .map((p, i) => {
            const color = SEVERITY_COLORS[i];
            const isMax = i === grade;
            return `
                <div class="prob-row">
                    <span class="prob-label" style="${isMax ? "color: var(--text-bright); font-weight: 600;" : ""}">${DR_LABELS[i]}</span>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width: ${p * 100}%; background: ${color};"></div>
                    </div>
                    <span class="prob-value" style="${isMax ? `color: ${color}; font-weight: 700;` : ""}">${(p * 100).toFixed(1)}%</span>
                </div>
            `;
        })
        .join("");

    container.innerHTML = `
        <div class="severity-badge severity-badge--${grade}">
            <span>Grade ${grade}</span>
            <span>${label}</span>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar-label">
                <span>Model Confidence</span>
                <span style="color: ${SEVERITY_COLORS[grade]}; font-weight: 600;">${(confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-bar__fill" style="width: ${confidence * 100}%; background: ${SEVERITY_COLORS[grade]};"></div>
            </div>
        </div>
        <div class="probability-chart">
            <h4 style="font-size: var(--font-size-sm); color: var(--text-muted); margin-bottom: var(--space-sm);">Class Probabilities</h4>
            ${probBarsHtml}
        </div>
    `;
}

// --- Grad-CAM ---
function renderGradCAM(gradcamBase64) {
    const viewer = document.getElementById("gradcamViewer");
    const img = document.getElementById("gradcamImage");

    if (!gradcamBase64) {
        document.getElementById("gradcamCard").style.display = "none";
        return;
    }

    document.getElementById("gradcamCard").style.display = "block";
    gradcamImageDataUrl = `data:image/png;base64,${gradcamBase64}`;
    img.src = gradcamImageDataUrl;

    // Toggle buttons
    const toggleOriginal = document.getElementById("toggleOriginal");
    const toggleHeatmap = document.getElementById("toggleHeatmap");
    const toggleOverlay = document.getElementById("toggleOverlay");

    function setActiveToggle(btn) {
        [toggleOriginal, toggleHeatmap, toggleOverlay].forEach((b) =>
            b.classList.remove("btn--active")
        );
        btn.classList.add("btn--active");
    }

    toggleOriginal.onclick = () => {
        img.src = originalImageDataUrl;
        setActiveToggle(toggleOriginal);
    };

    toggleHeatmap.onclick = () => {
        img.src = gradcamImageDataUrl;
        setActiveToggle(toggleHeatmap);
    };

    toggleOverlay.onclick = () => {
        img.src = gradcamImageDataUrl;
        setActiveToggle(toggleOverlay);
    };
}

// --- Similar Cases ---
function renderSimilarCases(cases) {
    const container = document.getElementById("casesList");

    if (!cases || cases.length === 0) {
        container.innerHTML = '<div class="cases-empty">No similar historical cases available</div>';
        return;
    }

    container.innerHTML = cases
        .map(
            (c) => `
            <div class="case-item" onclick="this.classList.toggle('expanded')">
                <div class="case-item__header">
                    <span class="case-item__id">${c.case_id}</span>
                    <span class="case-item__similarity">${(c.similarity * 100).toFixed(1)}% match</span>
                </div>
                <p class="case-item__text">${c.text}</p>
            </div>
        `
        )
        .join("");
}

// --- Report ---
function renderReport(report) {
    const container = document.getElementById("reportContent");

    if (!report) {
        container.innerHTML = '<p class="cases-empty">Report unavailable</p>';
        return;
    }

    // Basic markdown → HTML conversion for the report
    let html = report
        .replace(/### (.*)/g, "<h3>$1</h3>")
        .replace(/## (.*)/g, "<h2>$1</h2>")
        .replace(/# (.*)/g, "<h1>$1</h1>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>")
        .replace(/^- (.*)/gm, "<li>$1</li>")
        .replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>")
        .replace(/\n\n/g, "</p><p>")
        .replace(/\n/g, "<br>");

    html = `<p>${html}</p>`;

    // Clean up nested tags
    html = html.replace(/<p><h([123])>/g, "<h$1>").replace(/<\/h([123])><\/p>/g, "</h$1>");
    html = html.replace(/<p><ul>/g, "<ul>").replace(/<\/ul><\/p>/g, "</ul>");

    container.innerHTML = html;
}

// ============================================================
// Hero Typing Animation
// ============================================================
function initHeroTyping() {
    const typingElement = document.getElementById("typingHero");
    if (!typingElement) return;

    const phrase = "EDiTH";
    let isDeleting = false;
    let charIndex = 0;

    // Reset initial text so typing begins visibly from empty
    typingElement.textContent = "";

    function typeStep() {
        if (!isDeleting) {
            charIndex++;
            typingElement.textContent = phrase.slice(0, charIndex);

            if (charIndex === phrase.length) {
                // Keep the complete word visible with the cursor blinking
                setTimeout(() => {
                    isDeleting = true;
                    typeStep();
                }, 3500);
                return;
            }

            // Keystroke cadence with natural slight variance
            const speed = 140 + Math.random() * 80;
            setTimeout(typeStep, speed);
        } else {
            charIndex--;
            typingElement.textContent = phrase.slice(0, charIndex);

            if (charIndex === 0) {
                isDeleting = false;
                // Pause before typing starts again
                setTimeout(typeStep, 600);
                return;
            }

            // Quick backspace cadence
            setTimeout(typeStep, 80);
        }
    }

    // Begin typing after initial render
    setTimeout(typeStep, 350);
}

// ============================================================
// Init
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    initHeroTyping();
    checkServerHealth();
    // Re-check health every 30 seconds
    setInterval(checkServerHealth, 30000);
});

