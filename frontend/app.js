const phoneStage = document.getElementById("phoneStage");
const BASE_PHONE_WIDTH = 404;
const BASE_PHONE_HEIGHT = 873.7301635742188;
const VIEWPORT_PAD = 16;

function fitPhoneToViewport() {
	const availableWidth = Math.max(window.innerWidth - VIEWPORT_PAD * 2, 1);
	const availableHeight = Math.max(window.innerHeight - VIEWPORT_PAD * 2, 1);
	const widthScale = availableWidth / BASE_PHONE_WIDTH;
	const heightScale = availableHeight / BASE_PHONE_HEIGHT;
	const scale = Math.min(1, widthScale, heightScale);

	phoneStage.style.setProperty("--phone-scale", String(scale));
}

fitPhoneToViewport();
window.addEventListener("resize", fitPhoneToViewport);

const idleView = document.getElementById("idleView");
const recordingView = document.getElementById("recordingView");
const confirmView = document.getElementById("confirmView");
const learnView = document.getElementById("learnView");
const pageTitle = document.getElementById("pageTitle");
const startButton = document.getElementById("startButton");
const pauseButton = document.getElementById("pauseButton");
const stopButton = document.getElementById("stopButton");
const deleteButton = document.getElementById("deleteButton");
const saveButton = document.getElementById("saveButton");
const editNameButton = document.getElementById("editNameButton");
const previewPlayButton = document.getElementById("previewPlayButton");
const previewTime = document.getElementById("previewTime");
const saveStatus = document.getElementById("saveStatus");
const timerDisplay = document.getElementById("timerDisplay");
const stepRecord = document.getElementById("stepRecord");
const stepConfirm = document.getElementById("stepConfirm");
const stepLearn = document.getElementById("stepLearn");
const audioPreview = document.getElementById("audioPreview");
const recordingName = document.getElementById("recordingName");
const categorySelect = document.getElementById("categorySelect");
const learnDonut = document.getElementById("learnJarContainer");
const learnJarFill = document.getElementById("learnJarFill");
const learnJarFillerText = document.getElementById("learnJarFillerText");
const learnJarPreviousLine = document.getElementById("learnJarPreviousLine");
const learnJarPreviousLabel = document.getElementById("learnJarPreviousLabel");
const learnFillerLabel = learnJarFillerText;
const learnPreviousLabel = learnJarPreviousLabel;
const learnNextButton = document.getElementById("learnNextButton");
const learnViewStep2 = document.getElementById("learnViewStep2");
const learnViewStep3 = document.getElementById("learnViewStep3");
const paceRecommendedRangeArc = document.getElementById("paceRecommendedRangeArc");
const paceCurrentTriangle = document.getElementById("paceCurrentTriangle");
const paceHistoricAverageLine = document.getElementById("paceHistoricAverageLine");
const paceBandLabelFast = document.getElementById("paceBandLabelFast");
const paceBandLabel150 = document.getElementById("paceBandLabel150");
const paceBandLabel125 = document.getElementById("paceBandLabel125");
const paceBandLabel100 = document.getElementById("paceBandLabel100");
const paceBandLabelSlow = document.getElementById("paceBandLabelSlow");
const paceValue = document.getElementById("paceValue");
const paceMessage = document.getElementById("paceMessage");
const learnPaceSubtext = document.getElementById("learnPaceSubtext");
const learnBackButton = document.getElementById("learnBackButton");
const learnStep2NextButton = document.getElementById("learnStep2NextButton");
const learnPitchSubtext = document.getElementById("learnPitchSubtext");
const learnPitchChart = document.getElementById("learnPitchChart");
const learnPitchMessage = document.getElementById("learnPitchMessage");
const learnPitchBackButton = document.getElementById("learnPitchBackButton");
const learnPitchFinishButton = document.getElementById("learnPitchFinishButton");
const learnInfoTriggers = Array.from(document.querySelectorAll(".learn-info-trigger"));
const learnInfoOverlay = document.getElementById("learnInfoOverlay");
const learnInfoPopupTitle = document.getElementById("learnInfoPopupTitle");
const learnInfoPopupBody = document.getElementById("learnInfoPopupBody");
const previewWaveform = document.getElementById("previewWaveform");
const previewCtx = previewWaveform.getContext("2d");
const waveformCanvas = document.getElementById("waveform");
const canvasCtx = waveformCanvas.getContext("2d");
const textMeasureCanvas = document.createElement("canvas");
const textMeasureCtx = textMeasureCanvas.getContext("2d");

let stream;
let mediaRecorder;
let audioContext;
let analyser;
let dataArray;
let animationFrameId;
let timerIntervalId;
let recordingStartMs = 0;
let elapsedBeforePauseMs = 0;
let isPaused = false;
let waveformPeaks = [];
let lastPeakCaptureAt = 0;
let recordedChunks = [];
let recordedAudioBlob = null;
let recordedAudioUrl = "";
let recordedDurationMs = 0;
let isSaving = false;
let currentLearnStep = 1;
let currentLearnResult = null;

const PEAK_CAPTURE_INTERVAL_MS = 45;
const MAX_STORED_PEAKS = 50000;
const WAVEFORM_BAR_SPACING = 3;
const Recommended_Speed_Range = { startDeg: 45, endDeg: 135 };
const Historic_Average_Pace = 130;
const Fixed_avg = true;
const PITCH_HISTORY_STORAGE_KEY = "pitchVariationHistory";
const PITCH_HISTORY_DECAY = 0.82;
const PACE_BAND_LABEL_ELLIPSE = {
	cx: 273,
	cy: 273,
	radiusX: 320,
	radiusY: 300,
};
const PACE_GAUGE_GEOMETRY = {
	cx: 273,
	cy: 273,
	outerRadius: 273,
	innerRadius: 273 * 0.75,
};
const API_BASE_STORAGE_KEY = "speechApiBase";
const FILLER_HISTORY_STORAGE_KEY = "fillerPercentageHistory";
const HISTORY_DECAY = 0.85;
const queryParams = new URLSearchParams(window.location.search);
const UI_PREVIEW_MODE = queryParams.get("uiPreview") === "1"
	|| queryParams.get("design") === "1"
	|| window.self !== window.top;
const PREVIEW_SCREEN = (queryParams.get("previewScreen") || queryParams.get("screen") || "").toLowerCase();

const LEARN_INFO_POPUP_CONTENT = {
	filler: {
		title: "Filler words",
		bodyHtml: `
			<p>Words like <span class="learn-info-popup-highlight">&ldquo;um&rdquo;</span>, <span class="learn-info-popup-highlight">&ldquo;uh&rdquo;</span>, and <span class="learn-info-popup-highlight">&ldquo;like&rdquo;</span> are completely normal, but using too many of them can weaken your message. They may make you sound less <span class="learn-info-popup-highlight">confident</span>, <span class="learn-info-popup-highlight">prepared</span>, and make it harder to <span class="learn-info-popup-highlight">follow</span>.</p>
			<p>When speaking, clearer language helps your ideas land more effectively and keeps listeners focused on what you have to say.</p>
			<p><span class="learn-info-popup-highlight">Reducing filler words</span> helps you come across as more:</p>
			<ul class="learn-info-popup-list">
				<li>Confident</li>
				<li>Credible</li>
				<li>Clear</li>
				<li>Professional</li>
				<li>Engaging</li>
			</ul>
		`,
	},
	pace: {
		title: "Pace",
		bodyHtml: `
			<p>A good <span class="learn-info-popup-highlight">pace</span> shapes how your speech <span class="learn-info-popup-highlight">feels</span> to an audience.</p>
			<p>If you speak too fast, listeners may <span class="learn-info-popup-highlight">miss key points</span> or feel <span class="learn-info-popup-highlight">overwhelmed</span>. If you speak too slowly, your message can <span class="learn-info-popup-highlight">lose energy</span> and momentum.</p>
			<p>A well-paced delivery makes your speech <span class="learn-info-popup-highlight">easier to process</span> and helps your audience stay <span class="learn-info-popup-highlight">attentive</span> from start to finish.</p>
			<p>A <span class="learn-info-popup-highlight">balanced speaking pace</span> helps you sound more:</p>
			<ul class="learn-info-popup-list">
				<li>Dynamic</li>
				<li>Natural</li>
				<li>Intentional</li>
				<li>Engaging</li>
			</ul>
		`,
	},
	pitch: {
		title: "Pitch variation",
		bodyHtml: `
			<p>Your voice's ups and downs greatly affect how your speech is <span class="learn-info-popup-highlight">perceived</span>.</p>
			<p>Too little variation (<span class="learn-info-popup-highlight">close to 0</span>) can make you sound <span class="learn-info-popup-highlight">flat</span> or <span class="learn-info-popup-highlight">monotone</span>, while too much (<span class="learn-info-popup-highlight">above +/-5</span>) can sound <span class="learn-info-popup-highlight">exaggerated</span> or <span class="learn-info-popup-highlight">uncontrolled</span>.</p>
			<p>A balanced variation helps <span class="learn-info-popup-highlight">emphasize important words</span>, convey <span class="learn-info-popup-highlight">emotions</span>, and give your speech a more pleasant <span class="learn-info-popup-highlight">rhythm</span>.</p>
			<p>A good <span class="learn-info-popup-highlight">pitch variation</span> helps you sound more:</p>
			<ul class="learn-info-popup-list">
				<li>Expressive</li>
				<li>Natural</li>
				<li>Engaging</li>
				<li>Warm</li>
				<li>Dynamic</li>
			</ul>
		`,
	},
};

function openLearnInfoPopup(topic) {
	if (!learnInfoOverlay || !learnInfoPopupTitle || !learnInfoPopupBody) {
		return;
	}

	const selectedContent = LEARN_INFO_POPUP_CONTENT[topic] || LEARN_INFO_POPUP_CONTENT.filler;
	learnInfoPopupTitle.textContent = selectedContent.title;
	learnInfoPopupBody.innerHTML = selectedContent.bodyHtml;
	learnInfoOverlay.hidden = false;
}

function closeLearnInfoPopup() {
	if (!learnInfoOverlay) {
		return;
	}

	learnInfoOverlay.hidden = true;
}

function createSilentWavBlob(durationMs = 2200, sampleRate = 16000) {
	const channels = 1;
	const bytesPerSample = 2;
	const sampleCount = Math.max(1, Math.floor((durationMs / 1000) * sampleRate));
	const dataSize = sampleCount * bytesPerSample;
	const buffer = new ArrayBuffer(44 + dataSize);
	const view = new DataView(buffer);

	function writeAscii(offset, text) {
		for (let i = 0; i < text.length; i += 1) {
			view.setUint8(offset + i, text.charCodeAt(i));
		}
	}

	writeAscii(0, "RIFF");
	view.setUint32(4, 36 + dataSize, true);
	writeAscii(8, "WAVE");
	writeAscii(12, "fmt ");
	view.setUint32(16, 16, true);
	view.setUint16(20, 1, true);
	view.setUint16(22, channels, true);
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * channels * bytesPerSample, true);
	view.setUint16(32, channels * bytesPerSample, true);
	view.setUint16(34, bytesPerSample * 8, true);
	writeAscii(36, "data");
	view.setUint32(40, dataSize, true);

	return new Blob([buffer], { type: "audio/wav" });
}

function generatePreviewPeaks() {
	const points = [];
	for (let i = 0; i < 180; i += 1) {
		const t = i / 8;
		const base = 0.16 + Math.abs(Math.sin(t) * 0.4);
		const variation = (Math.sin(i * 1.7) + 1) * 0.09;
		points.push(Math.min(0.95, base + variation));
	}
	return points;
}

function buildPreviewResult() {
	const pitchSeries = generatePreviewPitchSeries();
	return {
		wpm: 127.4,
		filler_metrics: {
			total_words: 181,
			total_filler_words: 27,
			filler_percentage: 14.9,
			filler_word_counts: {
				uhm: 8,
				so: 7,
				like: 6,
				uh: 4,
				well: 2,
			},
		},
		pitch: {
			mean_pitch: 132.8,
			min_pitch: 94.2,
			max_pitch: 236.4,
			median_pitch: 131.1,
			mean_pitch_semitones: 0.0,
			min_pitch_semitones: -8.9,
			max_pitch_semitones: 9.2,
			median_pitch_semitones: 0.0,
			pitch_reference_hz: 131.1,
			voiced_ratio: 0.84,
		},
		pitch_series: pitchSeries,
	};
}

function generatePreviewPitchSeries() {
	const durationSeconds = 330;
	const sampleCount = 420;
	const referenceHz = 131.1;
	const points = [];

	for (let i = 0; i < sampleCount; i += 1) {
		const ratio = sampleCount === 1 ? 0 : i / (sampleCount - 1);
		const t = ratio * durationSeconds;
		const variation = (
			Math.sin(t / 24) * 2.9
			+ Math.cos(t / 11) * 1.5
			+ Math.sin(t / 5.7) * 0.7
			+ Math.cos(t / 48) * 0.9
		);
		const isVoiced = (i % 29 !== 0) && (i % 43 !== 0) && (i % 67 < 61);

		points.push({
			time: Number(t.toFixed(3)),
			pitch_hz: isVoiced ? Number((referenceHz * (2 ** (variation / 12))).toFixed(2)) : null,
			pitch_semitones: isVoiced ? Number(variation.toFixed(3)) : null,
			voiced: isVoiced,
		});
	}

	return points;
}

function seedPreviewRecordingState() {
	const fakeDurationMs = 2600;
	setPreviewAudio(createSilentWavBlob(fakeDurationMs));
	recordedDurationMs = fakeDurationMs;
	recordedChunks = [];
	waveformPeaks = generatePreviewPeaks();
	lastPeakCaptureAt = 0;
	renderPreviewWaveform();
}

function initializePreviewScreen() {
	if (!UI_PREVIEW_MODE) {
		return;
	}

	const target = ["idle", "record", "confirm", "learn"].includes(PREVIEW_SCREEN)
		? PREVIEW_SCREEN
		: "idle";

	if (target === "idle") {
		showIdleView();
		showPreviewModeHint();
		return;
	}

	seedPreviewRecordingState();

	if (target === "record") {
		recordingStartMs = Date.now();
		elapsedBeforePauseMs = 0;
		isPaused = false;
		clearInterval(timerIntervalId);
		timerIntervalId = setInterval(updateTimer, 200);
		updateTimer();
		showRecordingView();
		drawWaveform();
		showPreviewModeHint();
		return;
	}

	showConfirmView();

	if (target === "learn") {
		showLearnView(buildPreviewResult());
	}

	showPreviewModeHint();
}

function normalizeApiBase(value) {
	return String(value || "").trim().replace(/\/+$/, "");
}

function resolveApiBase() {
	const queryBase = normalizeApiBase(queryParams.get("apiBase"));
	if (queryBase) {
		localStorage.setItem(API_BASE_STORAGE_KEY, queryBase);
		return queryBase;
	}

	const storedBase = normalizeApiBase(localStorage.getItem(API_BASE_STORAGE_KEY));
	if (storedBase) {
		return storedBase;
	}

	if (window.location.protocol === "file:") {
		return "http://127.0.0.1:8000";
	}

	return window.location.origin;
}

const API_BASE = resolveApiBase();

function buildApiUrl(path) {
	const safePath = String(path || "").startsWith("/") ? path : `/${path}`;
	return `${API_BASE}${safePath}`;
}

function showPreviewModeHint() {
	if (!UI_PREVIEW_MODE) {
		return;
	}
	setSaveStatus("UI preview mode active: microphone and backend processing are mocked.");
}

function setStep(stepName) {
	stepRecord.classList.toggle("active", stepName === "record");
	stepConfirm.classList.toggle("active", stepName === "confirm");
	stepLearn.classList.toggle("active", stepName === "learn");
}

function setPreviewAudio(blob) {
	recordedAudioBlob = blob;

	if (recordedAudioUrl) {
		audioPreview.pause();
		URL.revokeObjectURL(recordedAudioUrl);
		recordedAudioUrl = "";
	}

	if (!blob) {
		audioPreview.removeAttribute("src");
		audioPreview.load();
		previewTime.textContent = "00:00";
		updatePreviewPlayButton(false);
		return;
	}

	recordedAudioUrl = URL.createObjectURL(blob);
	audioPreview.src = recordedAudioUrl;
	audioPreview.load();
}

function setSaveStatus(message, isError = false) {
	saveStatus.textContent = message || "";
	saveStatus.classList.toggle("error", Boolean(isError));
}

function setSavingState(isProcessing) {
	isSaving = isProcessing;
	saveButton.disabled = isProcessing;
	deleteButton.disabled = isProcessing;
	previewPlayButton.disabled = isProcessing;
}

function sanitizeFilename(value) {
	const base = (value || "recording").trim();
	const cleaned = base.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/_+/g, "_").replace(/^_+|_+$/g, "");
	return cleaned || "recording";
}

function isAutoGeneratedRecordingName(value) {
	return /^Recording_\d+$/i.test(String(value || "").trim());
}

async function syncNextDefaultRecordingName({ force = false } = {}) {
	if (UI_PREVIEW_MODE) {
		return;
	}

	const currentName = recordingName.value.trim();
	if (!force && currentName && !isAutoGeneratedRecordingName(currentName)) {
		return;
	}

	try {
		const response = await fetch(buildApiUrl("/api/next-recording-name"));
		if (!response.ok) {
			return;
		}

		const payload = await response.json();
		const nextName = String(payload && payload.recording_name ? payload.recording_name : "").trim();
		if (!nextName) {
			return;
		}

		recordingName.value = nextName;
		syncHeaderWithName();
	} catch (_error) {
		// Keep current value when backend is unreachable.
	}
}

function audioBufferToWavBlob(audioBuffer) {
	const channels = 1;
	const sampleRate = audioBuffer.sampleRate;
	const source = audioBuffer.numberOfChannels > 1
		? audioBuffer.getChannelData(0)
		: audioBuffer.getChannelData(0);

	const bytesPerSample = 2;
	const dataSize = source.length * bytesPerSample;
	const headerSize = 44;
	const buffer = new ArrayBuffer(headerSize + dataSize);
	const view = new DataView(buffer);

	function writeAscii(offset, text) {
		for (let i = 0; i < text.length; i += 1) {
			view.setUint8(offset + i, text.charCodeAt(i));
		}
	}

	writeAscii(0, "RIFF");
	view.setUint32(4, 36 + dataSize, true);
	writeAscii(8, "WAVE");
	writeAscii(12, "fmt ");
	view.setUint32(16, 16, true);
	view.setUint16(20, 1, true);
	view.setUint16(22, channels, true);
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, sampleRate * channels * bytesPerSample, true);
	view.setUint16(32, channels * bytesPerSample, true);
	view.setUint16(34, 8 * bytesPerSample, true);
	writeAscii(36, "data");
	view.setUint32(40, dataSize, true);

	let offset = 44;
	for (let i = 0; i < source.length; i += 1) {
		const s = Math.max(-1, Math.min(1, source[i]));
		view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
		offset += 2;
	}

	return new Blob([buffer], { type: "audio/wav" });
}

async function convertBlobToWav(blob) {
	if (!blob) {
		throw new Error("No recorded audio found.");
	}

	if (blob.type === "audio/wav") {
		return blob;
	}

	const arrayBuffer = await blob.arrayBuffer();
	const context = new (window.AudioContext || window.webkitAudioContext)();
	try {
		const decoded = await context.decodeAudioData(arrayBuffer.slice(0));
		return audioBufferToWavBlob(decoded);
	} finally {
		if (context.state !== "closed") {
			await context.close();
		}
	}
}

function updatePreviewPlayButton(isPlaying) {
	previewPlayButton.innerHTML = isPlaying
		? `
			<svg viewBox="0 0 24 24" aria-hidden="true">
				<rect x="6" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
				<rect x="13.8" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
			</svg>
		`
		: `
			<svg viewBox="0 0 24 24" aria-hidden="true">
				<path d="M8 5.5L18 12L8 18.5a2 2 0 0 1-3-1.7V7.2a2 2 0 0 1 3-1.7z" fill="#ffffff"></path>
			</svg>
		`;
}

function renderPreviewWaveform() {
	const { width, height } = previewWaveform;
	previewCtx.clearRect(0, 0, width, height);
	previewCtx.strokeStyle = "#21afa2";
	previewCtx.lineWidth = 2;

	if (!waveformPeaks.length) {
		previewCtx.beginPath();
		previewCtx.moveTo(0, height / 2);
		previewCtx.lineTo(width, height / 2);
		previewCtx.stroke();
		return;
	}

	const centerY = height / 2;
	const spacing = 3;
	const maxBars = Math.max(1, Math.floor((width - 2) / spacing));
	const bars = compressPeaksForBars(waveformPeaks, maxBars);

	previewCtx.beginPath();
	for (let i = 0; i < bars.length; i += 1) {
		const x = 1 + i * spacing;
		const barHeight = Math.max(2, bars[i] * (height * 0.44));
		previewCtx.moveTo(x, centerY - barHeight);
		previewCtx.lineTo(x, centerY + barHeight);
	}
	previewCtx.stroke();
}

function getFinalElapsedMs() {
	const runningMs = recordingStartMs ? Date.now() - recordingStartMs : 0;
	return elapsedBeforePauseMs + runningMs;
}

function syncHeaderWithName() {
	const cleanName = recordingName.value.trim();
	if (!confirmView.hidden && cleanName) {
		pageTitle.textContent = cleanName;
	}
}

function formatTime(ms) {
	const totalSeconds = Math.floor(ms / 1000);
	const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
	const seconds = String(totalSeconds % 60).padStart(2, "0");
	return `${minutes}:${seconds}`;
}

function updateTimer() {
	if (!recordingStartMs) {
		timerDisplay.textContent = "00:00";
		return;
	}
	const runningElapsed = Date.now() - recordingStartMs;
	timerDisplay.textContent = formatTime(elapsedBeforePauseMs + runningElapsed);
}

function drawIdleWaveformLine() {
	const { width, height } = waveformCanvas;
	canvasCtx.clearRect(0, 0, width, height);
	canvasCtx.strokeStyle = "#1e8486";
	canvasCtx.lineWidth = 2;
	canvasCtx.beginPath();
	canvasCtx.moveTo(0, height / 2);
	canvasCtx.lineTo(width, height / 2);
	canvasCtx.stroke();
}

function pushCurrentPeak() {
	if (!analyser || !dataArray) {
		return;
	}

	analyser.getByteTimeDomainData(dataArray);
	let peak = 0;
	for (let i = 0; i < dataArray.length; i += 1) {
		const normalized = Math.abs((dataArray[i] - 128) / 128);
		if (normalized > peak) {
			peak = normalized;
		}
	}

	waveformPeaks.push(peak);
	if (waveformPeaks.length > MAX_STORED_PEAKS) {
		waveformPeaks = waveformPeaks.slice(-MAX_STORED_PEAKS);
	}
}

function compressPeaksForBars(source, maxBars) {
	if (source.length <= maxBars) {
		return source;
	}

	const compressed = [];
	for (let bar = 0; bar < maxBars; bar += 1) {
		const start = Math.floor((bar * source.length) / maxBars);
		const end = Math.floor(((bar + 1) * source.length) / maxBars);
		let bucketPeak = 0;

		for (let i = start; i < end; i += 1) {
			if (source[i] > bucketPeak) {
				bucketPeak = source[i];
			}
		}

		compressed.push(bucketPeak);
	}

	return compressed;
}

function renderWaveformHistory() {
	const { width, height } = waveformCanvas;
	if (!waveformPeaks.length) {
		drawIdleWaveformLine();
		return;
	}

	canvasCtx.clearRect(0, 0, width, height);
	canvasCtx.strokeStyle = "#1e8486";
	canvasCtx.lineWidth = 2;

	const centerY = height / 2;
	const maxBars = Math.max(1, Math.floor((width - 2) / WAVEFORM_BAR_SPACING));
	const bars = compressPeaksForBars(waveformPeaks, maxBars);

	canvasCtx.beginPath();
	for (let i = 0; i < bars.length; i += 1) {
		const x = 1 + i * WAVEFORM_BAR_SPACING;
		const barHeight = Math.max(3, bars[i] * (height * 0.46));
		canvasCtx.moveTo(x, centerY - barHeight);
		canvasCtx.lineTo(x, centerY + barHeight);
	}
	canvasCtx.stroke();

	const isCompressed = waveformPeaks.length > maxBars;
	const endX = isCompressed ? width - 1 : Math.min(width - 1, 1 + (bars.length - 1) * WAVEFORM_BAR_SPACING);
	canvasCtx.beginPath();
	canvasCtx.moveTo(endX, 8);
	canvasCtx.lineTo(endX, height - 8);
	canvasCtx.stroke();
}

function drawWaveform() {
	if (!analyser) {
		renderWaveformHistory();
		animationFrameId = requestAnimationFrame(drawWaveform);
		return;
	}

	const now = performance.now();
	if (!isPaused && now - lastPeakCaptureAt >= PEAK_CAPTURE_INTERVAL_MS) {
		pushCurrentPeak();
		lastPeakCaptureAt = now;
	}

	renderWaveformHistory();
	animationFrameId = requestAnimationFrame(drawWaveform);
}

function showRecordingView() {
	closeLearnInfoPopup();
	idleView.hidden = true;
	recordingView.hidden = false;
	confirmView.hidden = true;
	learnView.hidden = true;
	if (learnViewStep3) {
		learnViewStep3.hidden = true;
	}
	pageTitle.textContent = "New recording";
	setStep("record");
	document.body.classList.add("is-recording");
}

function showIdleView() {
	closeLearnInfoPopup();
	learnView.hidden = true;
	if (learnViewStep3) {
		learnViewStep3.hidden = true;
	}
	confirmView.hidden = true;
	recordingView.hidden = true;
	idleView.hidden = false;
	pageTitle.textContent = "New recording";
	setStep("record");
	document.body.classList.remove("is-recording");
}

function showConfirmView() {
	closeLearnInfoPopup();
	idleView.hidden = true;
	recordingView.hidden = true;
	confirmView.hidden = false;
	learnView.hidden = true;
	if (learnViewStep3) {
		learnViewStep3.hidden = true;
	}
	syncHeaderWithName();
	void syncNextDefaultRecordingName();
	renderPreviewWaveform();
	previewTime.textContent = formatTime(recordedDurationMs);
	setStep("confirm");
	document.body.classList.remove("is-recording");
}

function loadFillerHistory() {
	try {
		const parsed = JSON.parse(localStorage.getItem(FILLER_HISTORY_STORAGE_KEY) || "[]");
		if (!Array.isArray(parsed)) {
			return [];
		}
		return parsed.filter((value) => typeof value === "number" && Number.isFinite(value));
	} catch (_error) {
		return [];
	}
}

function storeFillerHistory(history) {
	localStorage.setItem(FILLER_HISTORY_STORAGE_KEY, JSON.stringify(history));
}

function loadPitchVariationHistory() {
	try {
		const parsed = JSON.parse(localStorage.getItem(PITCH_HISTORY_STORAGE_KEY) || "[]");
		if (!Array.isArray(parsed)) {
			return [];
		}
		return parsed.filter((value) => typeof value === "number" && Number.isFinite(value) && value >= 0);
	} catch (_error) {
		return [];
	}
}

function storePitchVariationHistory(history) {
	localStorage.setItem(PITCH_HISTORY_STORAGE_KEY, JSON.stringify(history));
}

function quantile(sortedValues, q) {
	if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
		return NaN;
	}

	const clampedQ = Math.max(0, Math.min(1, q));
	const position = (sortedValues.length - 1) * clampedQ;
	const lowerIndex = Math.floor(position);
	const upperIndex = Math.ceil(position);
	if (lowerIndex === upperIndex) {
		return Number(sortedValues[lowerIndex]);
	}

	const lowerValue = Number(sortedValues[lowerIndex]);
	const upperValue = Number(sortedValues[upperIndex]);
	const fraction = position - lowerIndex;
	return lowerValue + ((upperValue - lowerValue) * fraction);
}

function calculatePitchVariation(pitchSeries, pitchSummary = {}) {
	const values = (Array.isArray(pitchSeries) ? pitchSeries : [])
		.map((point) => Number(point && (point.pitch_semitones ?? point.pitch_st)))
		.filter((value) => Number.isFinite(value));

	if (values.length >= 3) {
		const sorted = [...values].sort((a, b) => a - b);
		const p10 = quantile(sorted, 0.1);
		const p90 = quantile(sorted, 0.9);
		if (Number.isFinite(p10) && Number.isFinite(p90)) {
			return Math.max(0, (p90 - p10) / 2.0);
		}
	}

	const minSt = Number(pitchSummary.min_pitch_semitones);
	const maxSt = Number(pitchSummary.max_pitch_semitones);
	if (Number.isFinite(minSt) && Number.isFinite(maxSt)) {
		return Math.max(0, (maxSt - minSt) / 2.0);
	}

	return NaN;
}

function getPitchVariationLearnMessage(pitchSeries, pitchSummary = {}) {
	const values = (Array.isArray(pitchSeries) ? pitchSeries : [])
		.map((point) => Number(point && (point.pitch_semitones ?? point.pitch_st)))
		.filter((value) => Number.isFinite(value));

	const summaryMin = Number(pitchSummary.min_pitch_semitones);
	const summaryMax = Number(pitchSummary.max_pitch_semitones);
	const observedMin = Number.isFinite(summaryMin)
		? summaryMin
		: (values.length > 0 ? Math.min(...values) : NaN);
	const observedMax = Number.isFinite(summaryMax)
		? summaryMax
		: (values.length > 0 ? Math.max(...values) : NaN);

	const exceedsRecommendedBand = (Number.isFinite(observedMax) && observedMax > 5)
		|| (Number.isFinite(observedMin) && observedMin < -5);
	if (exceedsRecommendedBand) {
		return "Careful! You exceed the recommended variation, which can make your voice sound exaggerated or uncontrolled.";
	}

	const reachesRecommendedBand = values.some((value) => (
		(value >= 3 && value <= 5)
		|| (value <= -3 && value >= -5)
	));
	if (!reachesRecommendedBand) {
		return "Careful your voice doesn't become too monotone. Remember to e.g. go up on questions and down on statements.";
	}

	return "Good job! Keep variating between the upper and lower recommended band to sound perfectly engaging.";
}

function formatPitchAxisTime(seconds) {
	const safeSeconds = Math.max(0, Number(seconds) || 0);
	const minutes = Math.floor(safeSeconds / 60);
	const secondsPart = Math.round(safeSeconds % 60);
	return `${minutes}:${String(secondsPart).padStart(2, "0")}`;
}

function smoothValues(values, windowSize = 7) {
	if (!Array.isArray(values) || values.length === 0) {
		return [];
	}

	const size = Math.max(1, Math.floor(windowSize));
	const radius = Math.floor(size / 2);
	const result = [];

	for (let index = 0; index < values.length; index += 1) {
		let total = 0;
		let count = 0;
		for (let offset = -radius; offset <= radius; offset += 1) {
			const sampleIndex = index + offset;
			if (sampleIndex < 0 || sampleIndex >= values.length) {
				continue;
			}
			const value = Number(values[sampleIndex]);
			if (!Number.isFinite(value)) {
				continue;
			}
			total += value;
			count += 1;
		}
		result.push(count > 0 ? total / count : NaN);
	}

	return result;
}

function interpolateMissingPitchValues(points) {
	const interpolated = [];
	const finiteIndices = [];

	for (let index = 0; index < points.length; index += 1) {
		const value = Number(points[index].pitch);
		if (Number.isFinite(value)) {
			finiteIndices.push(index);
		}
	}

	if (finiteIndices.length === 0) {
		return points.map((point) => ({ ...point, pitch: NaN }));
	}

	for (let index = 0; index < points.length; index += 1) {
		const current = points[index];
		const value = Number(current.pitch);
		if (Number.isFinite(value)) {
			interpolated.push({ ...current, pitch: value });
			continue;
		}

		let previousIndex = index - 1;
		while (previousIndex >= 0 && !Number.isFinite(Number(points[previousIndex].pitch))) {
			previousIndex -= 1;
		}

		let nextIndex = index + 1;
		while (nextIndex < points.length && !Number.isFinite(Number(points[nextIndex].pitch))) {
			nextIndex += 1;
		}

		let replacement = NaN;
		if (previousIndex >= 0 && nextIndex < points.length) {
			const previousPoint = points[previousIndex];
			const nextPoint = points[nextIndex];
			const denominator = nextPoint.time - previousPoint.time;
			if (denominator > 0) {
				const ratio = (current.time - previousPoint.time) / denominator;
				replacement = previousPoint.pitch + ((nextPoint.pitch - previousPoint.pitch) * ratio);
			} else {
				replacement = previousPoint.pitch;
			}
		}

		interpolated.push({ ...current, pitch: replacement });
	}

	return interpolated;
}

function buildPitchVariationChartMarkup({ pitchSeries, previousAverageVariation }) {
	const points = Array.isArray(pitchSeries)
		? pitchSeries
			.map((point) => ({
				time: Number(point && point.time),
				pitch: Number(point && (point.pitch_semitones ?? point.pitch_st)),
				voiced: Boolean(point && point.voiced),
			}))
			.filter((point) => Number.isFinite(point.time))
		: [];

	if (!points.length) {
		return {
			markup: '<div class="pitch-chart-empty">No pitch data available for this recording.</div>',
			title: '',
		};
	}

	const completePoints = interpolateMissingPitchValues(points);
	const smoothedValues = smoothValues(completePoints.map((point) => point.pitch), 7);
	const chartPoints = completePoints.map((point, index) => ({
		time: point.time,
		pitch: Number.isFinite(smoothedValues[index]) ? smoothedValues[index] : null,
	}));

	const durationSeconds = Math.max(1, chartPoints[chartPoints.length - 1].time || 1);
	const width = 560;
	const height = 250;
	const margin = { top: 12, right: 18, bottom: 50, left: 52 };
	const plotWidth = width - margin.left - margin.right;
	const plotHeight = height - margin.top - margin.bottom;
	const yMin = -10;
	const yMax = 10;

	const xScale = (seconds) => margin.left + ((seconds / durationSeconds) * plotWidth);
	const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;

	const lineSegments = [];
	let currentSegment = [];
	chartPoints.forEach((point) => {
		if (!Number.isFinite(point.pitch)) {
			if (currentSegment.length > 0) {
				lineSegments.push(currentSegment.join(" "));
				currentSegment = [];
			}
			return;
		}

		const x = xScale(point.time);
		const y = yScale(point.pitch);
		currentSegment.push(`${currentSegment.length === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
	});
	if (currentSegment.length > 0) {
		lineSegments.push(currentSegment.join(" "));
	}

	const upperBandTop = yScale(5);
	const upperBandBottom = yScale(3);
	const lowerBandTop = yScale(-3);
	const lowerBandBottom = yScale(-5);
	const upperBandY = Math.min(upperBandTop, upperBandBottom);
	const upperBandHeight = Math.abs(upperBandBottom - upperBandTop);
	const lowerBandY = Math.min(lowerBandTop, lowerBandBottom);
	const lowerBandHeight = Math.abs(lowerBandBottom - lowerBandTop);

	const pathParts = [];
	chartPoints.forEach((point, index) => {
		const x = xScale(point.time);
		const y = yScale(point.pitch);
		pathParts.push(`${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
	});

	const xTickRatios = [0, 1 / 3, 2 / 3, 1];
	const xTicks = xTickRatios.map((ratio) => {
		const seconds = durationSeconds * ratio;
		return { x: xScale(seconds), label: formatPitchAxisTime(seconds) };
	});

	const yTicks = [];
	for (let tick = -10; tick <= 10; tick += 5) {
		yTicks.push({ y: yScale(tick), label: `${tick > 0 ? "+" : ""}${tick}` });
	}

	let previousLineMarkup = "";
	const hasUpper = previousAverageVariation
		&& Number.isFinite(previousAverageVariation.upper);
	const hasLower = previousAverageVariation
		&& Number.isFinite(previousAverageVariation.lower);
	if (hasUpper || hasLower) {
		const lineMarkup = [];
		if (hasUpper) {
			const previousUpper = yScale(previousAverageVariation.upper);
			lineMarkup.push(`<line x1="${margin.left}" y1="${previousUpper.toFixed(2)}" x2="${(margin.left + plotWidth).toFixed(2)}" y2="${previousUpper.toFixed(2)}" stroke="#666563" stroke-width="1.5" stroke-dasharray="5 4" />`);
		}
		if (hasLower) {
			const previousLower = yScale(previousAverageVariation.lower);
			lineMarkup.push(`<line x1="${margin.left}" y1="${previousLower.toFixed(2)}" x2="${(margin.left + plotWidth).toFixed(2)}" y2="${previousLower.toFixed(2)}" stroke="#666563" stroke-width="1.5" stroke-dasharray="5 4" />`);
		}
		previousLineMarkup = lineMarkup.join("\n");
	}

	const axisLabels = xTicks.map((tick) => `
		<text x="${tick.x.toFixed(2)}" y="${(height - 18).toFixed(2)}" text-anchor="middle" class="pitch-axis-label">${tick.label}</text>
	`).join("");
	const yAxisLabels = yTicks.map((tick) => `
		<text x="${(margin.left - 10).toFixed(2)}" y="${(tick.y + 4).toFixed(2)}" text-anchor="end" class="pitch-axis-label">${tick.label}</text>
	`).join("");

	return {
		markup: `
			<svg class="pitch-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Pitch variation line chart">
				<defs>
					<clipPath id="pitch-chart-clip">
						<rect x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}" />
					</clipPath>
				</defs>
				<rect x="0" y="0" width="${width}" height="${height}" rx="10" fill="#ffffff" />
				<g clip-path="url(#pitch-chart-clip)">
					<rect x="${margin.left}" y="${upperBandY.toFixed(2)}" width="${plotWidth}" height="${upperBandHeight.toFixed(2)}" fill="#dff0ee" opacity="0.9" />
					<rect x="${margin.left}" y="${lowerBandY.toFixed(2)}" width="${plotWidth}" height="${lowerBandHeight.toFixed(2)}" fill="#dff0ee" opacity="0.9" />
					${previousLineMarkup}
					${lineSegments.map((segment) => `<path d="${segment}" fill="none" stroke="#1e8486" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />`).join("")}
				</g>
				${xTicks.map((tick) => `<line x1="${tick.x.toFixed(2)}" y1="${(margin.top + plotHeight).toFixed(2)}" x2="${tick.x.toFixed(2)}" y2="${(margin.top + plotHeight + 4).toFixed(2)}" stroke="#666563" stroke-width="1" />`).join("")}
				${yTicks.map((tick) => `<line x1="${(margin.left - 4).toFixed(2)}" y1="${tick.y.toFixed(2)}" x2="${margin.left.toFixed(2)}" y2="${tick.y.toFixed(2)}" stroke="#666563" stroke-width="1" />`).join("")}
				${axisLabels}
				${yAxisLabels}
				<text x="${(margin.left + plotWidth / 2).toFixed(2)}" y="${(height - 4).toFixed(2)}" text-anchor="middle" class="pitch-axis-title">Time (sec)</text>
				<text x="16" y="${(margin.top + plotHeight / 2).toFixed(2)}" text-anchor="middle" class="pitch-axis-title" transform="rotate(-90 16 ${(margin.top + plotHeight / 2).toFixed(2)})">
					<tspan class="pitch-axis-title-main">Pitch</tspan>
					<tspan class="pitch-axis-title-sub"> (semitones)</tspan>
				</text>
			</svg>
		`,
	};
}

function calculateRecencyWeightedAverage(values, decay = HISTORY_DECAY) {
	if (!Array.isArray(values) || values.length === 0) {
		return NaN;
	}

	const safeDecay = Number.isFinite(decay) && decay > 0 ? decay : 0.15;
	let weightedSum = 0;
	let totalWeight = 0;
	let age = 0;

	for (let i = values.length - 1; i >= 0; i -= 1) {
		const value = Number(values[i]);
		if (!Number.isFinite(value)) {
			continue;
		}

		const weight = safeDecay ** age;
		weightedSum += value * weight;
		totalWeight += weight;
		age += 1;
	}

	if (totalWeight <= 0) {
		return NaN;
	}

	return weightedSum / totalWeight;
}

function sentenceCase(value) {
	if (!value) {
		return "";
	}
	const trimmed = String(value).trim();
	if (!trimmed) {
		return "";
	}
	return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}

function measureTextWidth(text, font = '400 14px "IBM Plex Mono"') {
	if (!textMeasureCtx) {
		return String(text || "").length * 8;
	}

	textMeasureCtx.font = font;
	return textMeasureCtx.measureText(String(text || "")).width;
}

function getSortedFillerPairs(fillerCounts) {
	if (!fillerCounts || typeof fillerCounts !== "object") {
		return [];
	}

	return Object.entries(fillerCounts)
		.filter(([, count]) => Number.isFinite(Number(count)) && Number(count) > 0)
		.sort((a, b) => Number(b[1]) - Number(a[1]));
}

function buildLearnMessage(currentPercentage, previousAverage) {
	if (!Number.isFinite(currentPercentage) || currentPercentage <= 0) {
		return "No filler words detected this session. Great control!";
	}

	if (!Number.isFinite(previousAverage)) {
		return `You used ${currentPercentage.toFixed(1)}% filler words this session.`;
	}

	if (currentPercentage < previousAverage) {
		return "You used less than your normal amount of filler words. Keep going!";
	}

	if (currentPercentage > previousAverage) {
		return "You used more filler words than your average. Keep practicing.";
	}

	return "You matched your previous filler-word average.";
}

function renderLearnWordList(sortedPairs) {
	if (!sortedPairs.length) {
		learnWordList.innerHTML = '<p class="learn-word-empty">No filler words were detected in this recording.</p>';
		return;
	}

	const topFive = sortedPairs.slice(0, 5);
	const maxBarWidth = 170;
	const gapBetweenNameAndCount = 8;
	const horizontalPadding = 16; // 8px left + 8px right 
	const longestLabelWidth = topFive.reduce((currentMax, [word]) => {
		const labelWidth = measureTextWidth(sentenceCase(word));
		return Math.max(currentMax, labelWidth);
	}, 0);
	const slimmestCountWidth = topFive.reduce((currentMax, [, count]) => {
		const countWidth = measureTextWidth(String(Number(count)));
		return Math.min(currentMax, countWidth);
	}, 100);

	const minTextBoxWidth = longestLabelWidth + 8;
	const minBarWidth = minTextBoxWidth + slimmestCountWidth + gapBetweenNameAndCount + horizontalPadding;
	const highestCount = Number(topFive[0][1]) || 1;

	function getBarWidth(countValue) {
		const count = Math.max(1, Number(countValue) || 1);
		if (highestCount <= 1) {
			return minBarWidth;
		}

		const ratio = (count - 1) / (highestCount - 1);
		return minBarWidth + (ratio * (maxBarWidth - minBarWidth));
	}

	learnWordList.innerHTML = topFive
		.map(([word, count]) => {
			const barWidth = getBarWidth(count).toFixed(2);
			const labelText = sentenceCase(word);
			const finalWidth = Math.max(Number(barWidth), minBarWidth);
			return `
				<div class="learn-word-item" style="width:${finalWidth.toFixed(2)}px;" role="listitem" aria-label="${labelText} ${Number(count)}">
					<span class="name">${labelText}</span>
					<span class="count">${Number(count)}</span>
				</div>
			`;
		})
		.join("");
}

function renderLearnJar(fillerPercentage, previousAveragePercentage) {
	const JAR_WIDTH = 77;
	const JAR_HEIGHT = 160;
	const JAR_VIEWBOX_WIDTH = 100;
	const JAR_VIEWBOX_HEIGHT = 180;
	const JAR_CORNER_RADIUS = 20;

	// Calculate scale factors for viewBox to actual pixel dimensions
	const scaleX = JAR_WIDTH / JAR_VIEWBOX_WIDTH;
	const scaleY = JAR_HEIGHT / JAR_VIEWBOX_HEIGHT;

	// Clamp percentage to 0-100
	const clampedFillerPct = Math.max(0, Math.min(100, fillerPercentage));
	const clampedPreviousPct = Math.max(0, Math.min(100, previousAveragePercentage));

	// Calculate fill height in viewBox coordinates
	// If less than 10%, use 10% height minimum for readability
	const displayFillerPct = Math.max(10, clampedFillerPct);
	const fillHeightViewBox = (displayFillerPct / 100) * JAR_VIEWBOX_HEIGHT;
	const fillY = JAR_VIEWBOX_HEIGHT - fillHeightViewBox;
	const fillRadius = Math.min(JAR_CORNER_RADIUS, fillHeightViewBox / 2, JAR_VIEWBOX_WIDTH / 2);

	// Update filler area height
	if (learnJarFill) {
		const roundedBottomFillPath = [
			`M 0 ${fillY.toFixed(2)}`,
			`H 100`,
			`V ${(JAR_VIEWBOX_HEIGHT - fillRadius).toFixed(2)}`,
			`A ${fillRadius.toFixed(2)} ${fillRadius.toFixed(2)} 0 0 1 ${(100 - fillRadius).toFixed(2)} ${JAR_VIEWBOX_HEIGHT.toFixed(2)}`,
			`H ${fillRadius.toFixed(2)}`,
			`A ${fillRadius.toFixed(2)} ${fillRadius.toFixed(2)} 0 0 1 0 ${(JAR_VIEWBOX_HEIGHT - fillRadius).toFixed(2)}`,
			"Z",
		].join(" ");
		learnJarFill.setAttribute("d", roundedBottomFillPath);
	}

	// Update current percentage text
	if (learnJarFillerText) {
		learnJarFillerText.textContent = `${Math.round(clampedFillerPct)}%`;
		// Center text vertically in the red fill area (or minimum 10% area)
		const textYViewBox = JAR_VIEWBOX_HEIGHT - (fillHeightViewBox / 2);
		learnJarFillerText.setAttribute("y", textYViewBox.toFixed(2));
	}

	// Update previous average line and label
	if (Number.isFinite(clampedPreviousPct) && clampedPreviousPct > 0) {
		// Line at y-coordinate representing the previous average percentage
		const lineYViewBox = JAR_VIEWBOX_HEIGHT - ((clampedPreviousPct / 100) * JAR_VIEWBOX_HEIGHT);

		if (learnJarPreviousLine) {
			learnJarPreviousLine.setAttribute("y1", lineYViewBox.toFixed(2));
			learnJarPreviousLine.setAttribute("y2", lineYViewBox.toFixed(2));
			learnJarPreviousLine.style.display = "block";
		}

		// Label text positioned to the left of jar at the same y-height
		if (learnJarPreviousLabel) {
			learnJarPreviousLabel.textContent = `${Math.round(clampedPreviousPct)}%`;
			// Convert viewBox y-coordinate to pixel coordinate
			const labelPixelY = lineYViewBox * scaleY;
			learnJarPreviousLabel.style.top = `${labelPixelY.toFixed(2)}px`;
			learnJarPreviousLabel.style.transform = "translate(-100%, -50%)";
			learnJarPreviousLabel.style.display = "block";
		}
	} else {
		// Hide previous average elements if no data
		if (learnJarPreviousLine) {
			learnJarPreviousLine.style.display = "none";
		}
		if (learnJarPreviousLabel) {
			learnJarPreviousLabel.style.display = "none";
		}
	}
}

function polarToSvg(cx, cy, radius, angleDeg) {
	const radians = (angleDeg * Math.PI) / 180;
	return {
		x: cx + radius * Math.cos(radians),
		y: cy - radius * Math.sin(radians),
	};
}

function buildAnnulusSegmentPath(cx, cy, outerRadius, innerRadius, startDeg, endDeg) {
	const outerStart = polarToSvg(cx, cy, outerRadius, startDeg);
	const outerEnd = polarToSvg(cx, cy, outerRadius, endDeg);
	const innerStart = polarToSvg(cx, cy, innerRadius, startDeg);
	const innerEnd = polarToSvg(cx, cy, innerRadius, endDeg);
	const span = ((endDeg - startDeg) % 360 + 360) % 360;
	const largeArcFlag = span > 180 ? 1 : 0;

	return [
		`M ${outerStart.x.toFixed(2)} ${outerStart.y.toFixed(2)}`,
		`A ${outerRadius} ${outerRadius} 0 ${largeArcFlag} 0 ${outerEnd.x.toFixed(2)} ${outerEnd.y.toFixed(2)}`,
		`L ${innerEnd.x.toFixed(2)} ${innerEnd.y.toFixed(2)}`,
		`A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 1 ${innerStart.x.toFixed(2)} ${innerStart.y.toFixed(2)}`,
		"Z",
	].join(" ");
}

function renderRecommendedSpeedRange() {
	if (!paceRecommendedRangeArc) {
		return;
	}

	const path = buildAnnulusSegmentPath(
		PACE_GAUGE_GEOMETRY.cx,
		PACE_GAUGE_GEOMETRY.cy,
		PACE_GAUGE_GEOMETRY.outerRadius,
		PACE_GAUGE_GEOMETRY.innerRadius,
		Recommended_Speed_Range.startDeg,
		Recommended_Speed_Range.endDeg
	);

	paceRecommendedRangeArc.setAttribute("d", path);
}

function paceToGaugeAngle(paceWpm) {
	const minPace = 100;
	const maxPace = 180;
	const clamped = Math.max(minPace, Math.min(maxPace, paceWpm));
	const ratio = (clamped - minPace) / (maxPace - minPace);
	return 180 - (ratio * 180);
}

function renderHistoricAverageLine(averagePaceWpm) {
	if (!paceHistoricAverageLine) {
		return;
	}

	const angle = paceToGaugeAngle(averagePaceWpm);
	const start = polarToSvg(PACE_GAUGE_GEOMETRY.cx, PACE_GAUGE_GEOMETRY.cy, PACE_GAUGE_GEOMETRY.outerRadius, angle);
	const end = polarToSvg(PACE_GAUGE_GEOMETRY.cx, PACE_GAUGE_GEOMETRY.cy, PACE_GAUGE_GEOMETRY.outerRadius - 64, angle);

	paceHistoricAverageLine.setAttribute("x1", start.x.toFixed(2));
	paceHistoricAverageLine.setAttribute("y1", start.y.toFixed(2));
	paceHistoricAverageLine.setAttribute("x2", end.x.toFixed(2));
	paceHistoricAverageLine.setAttribute("y2", end.y.toFixed(2));
	paceHistoricAverageLine.setAttribute("stroke-dasharray", "2 5");
}

function renderCurrentPaceTriangle(currentPaceWpm) {
	if (!paceCurrentTriangle) {
		return;
	}

	const angle = paceToGaugeAngle(currentPaceWpm);
	const rotationDeg = 90 - angle;
	paceCurrentTriangle.setAttribute(
		"transform",
		`rotate(${rotationDeg.toFixed(2)} ${PACE_GAUGE_GEOMETRY.cx} ${PACE_GAUGE_GEOMETRY.cy})`
	);
}

function renderPaceBandLabels() {
	const labels = [
		{ element: paceBandLabelFast, angle: 0, anchor: "middle", text: "fast" },
		{ element: paceBandLabel150, angle: 45, anchor: "middle", text: "160" },
		{ element: paceBandLabel125, angle: 90, anchor: "middle", text: "140" },
		{ element: paceBandLabel100, angle: 135, anchor: "middle", text: "120" },
		{ element: paceBandLabelSlow, angle: 180, anchor: "middle", text: "slow" },
	];

	for (const label of labels) {
		if (!label.element) {
			continue;
		}

		const radians = (label.angle * Math.PI) / 180;
		const point = {
			x: PACE_BAND_LABEL_ELLIPSE.cx + (PACE_BAND_LABEL_ELLIPSE.radiusX * Math.cos(radians)),
			y: PACE_BAND_LABEL_ELLIPSE.cy - (PACE_BAND_LABEL_ELLIPSE.radiusY * Math.sin(radians)),
		};

		label.element.setAttribute("x", point.x.toFixed(2));
		label.element.setAttribute("y", point.y.toFixed(2));
		label.element.setAttribute("text-anchor", label.anchor);
		if (label.text) {
			label.element.textContent = label.text;
		}
	}
}

function showLearnView(result) {
	currentLearnResult = result;
	currentLearnStep = 1;
	showLearnStep(1, result);
}

function showLearnStep(step, result) {
	closeLearnInfoPopup();
	const recordingTitle = recordingName.value ? recordingName.value.trim() : "Recording";
	pageTitle.textContent = recordingTitle || "Recording";

	idleView.hidden = true;
	recordingView.hidden = true;
	confirmView.hidden = true;
	document.body.classList.remove("is-recording");

	if (step === 1) {
		currentLearnStep = 1;
		showLearnStep1(result);
	} else if (step === 2) {
		currentLearnStep = 2;
		showLearnStep2(result);
	} else if (step === 3) {
		currentLearnStep = 3;
		showLearnStep3(result);
	}
	setStep("learn");
}

function showLearnStep1(result) {
	const fillerMetrics = result && result.filler_metrics ? result.filler_metrics : {};
	const fillerPercentageRaw = Number(fillerMetrics.filler_percentage);
	const fillerPercentage = Number.isFinite(fillerPercentageRaw)
		? Math.max(0, Math.min(100, fillerPercentageRaw))
		: 0;
	const nonFillerPercentage = Math.max(0, 100 - fillerPercentage);
	const sortedPairs = getSortedFillerPairs(fillerMetrics.filler_word_counts);

	const history = loadFillerHistory();
	const previousAverage = calculateRecencyWeightedAverage(history, HISTORY_DECAY);
	storeFillerHistory([...history, fillerPercentage]);
	const previousAverageClamped = Number.isFinite(previousAverage)
		? Math.max(0, Math.min(100, previousAverage))
		: 0;

	learnDonut.setAttribute(
		"aria-label",
		`Filler words ${fillerPercentage.toFixed(1)} percent, previous average ${previousAverageClamped.toFixed(1)} percent, non-filler words ${nonFillerPercentage.toFixed(1)} percent`
	);

	renderLearnJar(fillerPercentage, previousAverageClamped);

	learnSummaryText.textContent = Number.isFinite(previousAverage)
		? `Below, you can see your filler-word percentage for this session, along with your previous average.\nSee your top 5 filler words and how often you said them`
		: "Below, you can see your filler-word percentage for this session.\nSee your top 5 filler words and how often you said them";

	renderLearnWordList(sortedPairs);
	learnMessage.textContent = buildLearnMessage(fillerPercentage, previousAverage);

	learnView.hidden = false;
	learnViewStep2.hidden = true;
	if (learnViewStep3) {
		learnViewStep3.hidden = true;
	}
}

function showLearnStep2(result) {
	const wpm = Number(result && result.wpm) || 0;
	const validWpm = Number.isFinite(wpm) ? Math.max(0, Math.min(400, wpm)) : 0;
	renderPaceBandLabels();
	renderRecommendedSpeedRange();
	renderCurrentPaceTriangle(validWpm);
	renderHistoricAverageLine(Historic_Average_Pace);

	paceValue.textContent = Math.round(validWpm);

	// Build message based on pace
	let paceMessageText = "";
	if (validWpm < 100) {
		paceMessageText = "Your pace is quite slow. Try to speak a bit faster for better engagement.";
	} else if (validWpm < 120) {
		paceMessageText = "Your pace is a bit slow. Consider increasing your speaking speed slightly.";
	} else if (validWpm <= 150) {
		paceMessageText = "Your pace was just right! Keep it up!";
	} else if (validWpm <= 180) {
		paceMessageText = "Your pace is a bit fast. Try to slow down slightly for clarity.";
	} else {
		paceMessageText = "Your pace is quite fast. Slow down to ensure your audience can follow.";
	}
	paceMessage.textContent = paceMessageText;

	learnPaceSubtext.textContent = "Below, you can see your average pace (words per minute) for this session, compared to your previous sessions.";

	learnView.hidden = true;
	learnViewStep2.hidden = false;
	if (learnViewStep3) {
		learnViewStep3.hidden = true;
	}
}

function showLearnStep3(result) {
	const pitchSummary = result && result.pitch ? result.pitch : {};
	const pitchSeries = result && result.pitch_series ? result.pitch_series : [];
	const currentVariation = calculatePitchVariation(pitchSeries, pitchSummary);
	const history = loadPitchVariationHistory();
	const previousAverage = calculateRecencyWeightedAverage(history, PITCH_HISTORY_DECAY);
	const previousAverageVariation = Fixed_avg
		? { lower: -3, upper: 2 }
		: {
			lower: Number.isFinite(previousAverage) ? -previousAverage : NaN,
			upper: Number.isFinite(previousAverage) ? previousAverage : NaN,
		};

	if (Number.isFinite(currentVariation)) {
		storePitchVariationHistory([...history, currentVariation]);
	}

	if (learnPitchSubtext) {
		learnPitchSubtext.textContent = "Below, you can see your pitch variation and the recommended upper and lower band to reach.\nCompare with your average range from previous sessions.";
	}

	if (learnPitchChart) {
		learnPitchChart.innerHTML = buildPitchVariationChartMarkup({
			pitchSeries,
			previousAverageVariation,
		}).markup;
	}

	if (learnPitchMessage) {
		learnPitchMessage.textContent = Number.isFinite(currentVariation)
			? getPitchVariationLearnMessage(pitchSeries, pitchSummary)
			: "No pitch variation data was available for this recording.";
	}

	learnView.hidden = true;
	learnViewStep2.hidden = true;
	if (learnViewStep3) {
		learnViewStep3.hidden = false;
	}
}

function resetUiAfterStop() {
	isPaused = false;
	pauseButton.setAttribute("aria-label", "Pause recording");
	pauseButton.title = "Pause";
	pauseButton.innerHTML = `
		<svg viewBox="0 0 24 24" aria-hidden="true">
			<rect x="6" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
			<rect x="13.8" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
		</svg>
	`;
	timerDisplay.textContent = "00:00";
	drawIdleWaveformLine();
}

function stopTracks() {
	if (stream) {
		stream.getTracks().forEach((track) => track.stop());
		stream = null;
	}
}

async function startRecording() {
	setSaveStatus("");

	if (UI_PREVIEW_MODE) {
		recordedChunks = [];
		recordedDurationMs = 0;
		setPreviewAudio(createSilentWavBlob(2600));
		waveformPeaks = generatePreviewPeaks();
		lastPeakCaptureAt = 0;
		recordingStartMs = Date.now();
		elapsedBeforePauseMs = 0;
		isPaused = false;
		clearInterval(timerIntervalId);
		timerIntervalId = setInterval(updateTimer, 200);
		updateTimer();
		showRecordingView();
		drawWaveform();
		return;
	}

	try {
		stream = await navigator.mediaDevices.getUserMedia({ audio: true });
	} catch (error) {
		if (UI_PREVIEW_MODE) {
			setSaveStatus("Microphone unavailable in this preview. Use ?uiPreview=1 for design mode.", true);
		} else {
			alert("Microphone access is required to record.");
		}
		return;
	}

	audioContext = new (window.AudioContext || window.webkitAudioContext)();
	const source = audioContext.createMediaStreamSource(stream);
	analyser = audioContext.createAnalyser();
	analyser.fftSize = 256;
	dataArray = new Uint8Array(analyser.frequencyBinCount);
	source.connect(analyser);

	if (typeof MediaRecorder !== "undefined") {
		recordedChunks = [];
		recordedDurationMs = 0;
		setPreviewAudio(null);
		mediaRecorder = new MediaRecorder(stream);
		mediaRecorder.addEventListener("dataavailable", (event) => {
			if (event.data && event.data.size > 0) {
				recordedChunks.push(event.data);
			}
		});
		mediaRecorder.addEventListener("stop", () => {
			const mimeType = mediaRecorder && mediaRecorder.mimeType ? mediaRecorder.mimeType : "audio/webm";
			const audioBlob = recordedChunks.length ? new Blob(recordedChunks, { type: mimeType }) : null;
			setPreviewAudio(audioBlob);
		});
		mediaRecorder.start();
	}

	recordingStartMs = Date.now();
	elapsedBeforePauseMs = 0;
	waveformPeaks = [];
	lastPeakCaptureAt = 0;
	timerIntervalId = setInterval(updateTimer, 200);
	updateTimer();

	showRecordingView();
	drawWaveform();
}

async function togglePause() {
	if (UI_PREVIEW_MODE && recordingView.hidden === false) {
		if (!isPaused) {
			isPaused = true;
			elapsedBeforePauseMs += Date.now() - recordingStartMs;
			recordingStartMs = 0;
			clearInterval(timerIntervalId);
			pauseButton.setAttribute("aria-label", "Resume recording");
			pauseButton.title = "Resume";
			pauseButton.innerHTML = `
				<svg viewBox="0 0 24 24" aria-hidden="true">
					<path d="M8 5.5L18 12L8 18.5a2 2 0 0 1-3-1.7V7.2a2 2 0 0 1 3-1.7z" fill="#ffffff"></path>
				</svg>
			`;
			return;
		}

		isPaused = false;
		recordingStartMs = Date.now();
		clearInterval(timerIntervalId);
		timerIntervalId = setInterval(updateTimer, 200);
		pauseButton.setAttribute("aria-label", "Pause recording");
		pauseButton.title = "Pause";
		pauseButton.innerHTML = `
			<svg viewBox="0 0 24 24" aria-hidden="true">
				<rect x="6" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
				<rect x="13.8" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
			</svg>
		`;
		return;
	}

	if (!stream) {
		return;
	}

	if (!isPaused) {
		isPaused = true;
		elapsedBeforePauseMs += Date.now() - recordingStartMs;
		recordingStartMs = 0;
		clearInterval(timerIntervalId);

		if (audioContext && audioContext.state === "running") {
			await audioContext.suspend();
		}

		if (mediaRecorder && mediaRecorder.state === "recording") {
			mediaRecorder.pause();
		}

		pauseButton.setAttribute("aria-label", "Resume recording");
		pauseButton.title = "Resume";
		pauseButton.innerHTML = `
			<svg viewBox="0 0 24 24" aria-hidden="true">
				<path d="M8 5.5L18 12L8 18.5a2 2 0 0 1-3-1.7V7.2a2 2 0 0 1 3-1.7z" fill="#ffffff"></path>
			</svg>
		`;
		return;
	}

	isPaused = false;
	recordingStartMs = Date.now();
	timerIntervalId = setInterval(updateTimer, 200);

	if (audioContext && audioContext.state === "suspended") {
		await audioContext.resume();
	}

	if (mediaRecorder && mediaRecorder.state === "paused") {
		mediaRecorder.resume();
	}

	pauseButton.setAttribute("aria-label", "Pause recording");
	pauseButton.title = "Pause";
	pauseButton.innerHTML = `
		<svg viewBox="0 0 24 24" aria-hidden="true">
			<rect x="6" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
			<rect x="13.8" y="4" width="4.2" height="16" rx="2" fill="#ffffff"></rect>
		</svg>
	`;
}

function stopRecording() {
	if (UI_PREVIEW_MODE && !stream) {
		recordedDurationMs = getFinalElapsedMs();
		clearInterval(timerIntervalId);
		cancelAnimationFrame(animationFrameId);
		recordingStartMs = 0;
		elapsedBeforePauseMs = 0;
		isPaused = false;
		recordingName.value = "";
		showConfirmView();
		resetUiAfterStop();
		showPreviewModeHint();
		return;
	}

	if (!stream) {
		return;
	}

	recordedDurationMs = getFinalElapsedMs();

	if (mediaRecorder && mediaRecorder.state !== "inactive") {
		mediaRecorder.stop();
	}

	clearInterval(timerIntervalId);
	cancelAnimationFrame(animationFrameId);

	if (audioContext && audioContext.state !== "closed") {
		audioContext.close();
	}

	stopTracks();

	mediaRecorder = null;
	audioContext = null;
	analyser = null;
	dataArray = null;
	recordingStartMs = 0;
	elapsedBeforePauseMs = 0;
	isPaused = false;
	recordingName.value = "";

	showConfirmView();
	resetUiAfterStop();
}

function clearPreviewAndReturnToIdle() {
	clearInterval(timerIntervalId);
	cancelAnimationFrame(animationFrameId);
	recordingStartMs = 0;
	elapsedBeforePauseMs = 0;
	waveformPeaks = [];
	setPreviewAudio(null);
	recordedChunks = [];
	recordedDurationMs = 0;
	setSaveStatus("");
	resetUiAfterStop();
	showIdleView();
}

async function saveRecording() {
	if (isSaving) {
		return;
	}

	if (UI_PREVIEW_MODE) {
		setSavingState(true);
		setSaveStatus("Processing preview data...");
		try {
			showLearnView(buildPreviewResult());
			setSaveStatus("");
		} finally {
			setSavingState(false);
		}
		return;
	}

	if (!recordedAudioBlob) {
		setSaveStatus("No recording to save yet.", true);
		return;
	}

	setSavingState(true);
	setSaveStatus("Processing recording...");

	try {
		const wavBlob = await convertBlobToWav(recordedAudioBlob);
		const safeName = sanitizeFilename(recordingName.value);
		const formData = new FormData();
		formData.append("audio", wavBlob, `${safeName}.wav`);
		formData.append("recording_name", recordingName.value.trim() || "Recording");
		formData.append("category", categorySelect.value || "");

		const response = await fetch(buildApiUrl("/api/process-recording"), {
			method: "POST",
			body: formData,
		});

		let payload = null;
		try {
			payload = await response.json();
		} catch (parseError) {
			payload = null;
		}

		if (!response.ok || !payload || !payload.ok) {
			const message = payload && payload.error ? payload.error : "Processing failed on server.";
			throw new Error(message);
		}

		const result = payload.result || {};
		showLearnView(result);
	} catch (error) {
		if (error instanceof TypeError && /fetch/i.test(error.message || "")) {
			setSaveStatus(
				`Could not reach API at ${API_BASE}. Start the local app with "python main.py --serve --port 8000" instead of "python -m http.server 8000", or open with ?apiBase=http://<host>:<port>`,
				true
			);
			return;
		}

		setSaveStatus(error.message || "Could not save and process recording.", true);
	} finally {
		setSavingState(false);
	}
}

function togglePreviewPlayback() {
	if (!audioPreview.src) {
		return;
	}

	if (audioPreview.paused) {
		audioPreview.play();
		return;
	}

	audioPreview.pause();
}

for (const trigger of learnInfoTriggers) {
	trigger.addEventListener("click", () => {
		const topic = trigger.dataset.infoTopic || "filler";
		openLearnInfoPopup(topic);
	});

	trigger.addEventListener("keydown", (event) => {
		if (event.key !== "Enter" && event.key !== " ") {
			return;
		}

		event.preventDefault();
		const topic = trigger.dataset.infoTopic || "filler";
		openLearnInfoPopup(topic);
	});
}

if (learnInfoOverlay) {
	learnInfoOverlay.addEventListener("click", closeLearnInfoPopup);
}

document.addEventListener("keydown", (event) => {
	if (event.key === "Escape" && learnInfoOverlay && !learnInfoOverlay.hidden) {
		closeLearnInfoPopup();
	}
});

startButton.addEventListener("click", startRecording);
pauseButton.addEventListener("click", togglePause);
stopButton.addEventListener("click", stopRecording);
deleteButton.addEventListener("click", clearPreviewAndReturnToIdle);
saveButton.addEventListener("click", saveRecording);
learnNextButton.addEventListener("click", () => showLearnStep(2, currentLearnResult));
learnBackButton.addEventListener("click", () => showLearnStep(1, currentLearnResult));
learnStep2NextButton.addEventListener("click", () => showLearnStep(3, currentLearnResult));
if (learnPitchBackButton) {
	learnPitchBackButton.addEventListener("click", () => showLearnStep(2, currentLearnResult));
}
if (learnPitchFinishButton) {
	learnPitchFinishButton.addEventListener("click", clearPreviewAndReturnToIdle);
}
recordingName.addEventListener("input", syncHeaderWithName);
recordingName.addEventListener("focus", () => {
	if (isAutoGeneratedRecordingName(recordingName.value)) {
		recordingName.select();
	}
});
recordingName.addEventListener("click", () => {
	if (isAutoGeneratedRecordingName(recordingName.value)) {
		recordingName.select();
	}
});
editNameButton.addEventListener("click", () => {
	recordingName.focus();
	recordingName.select();
});
previewPlayButton.addEventListener("click", togglePreviewPlayback);
audioPreview.addEventListener("play", () => updatePreviewPlayButton(true));
audioPreview.addEventListener("pause", () => updatePreviewPlayButton(false));
audioPreview.addEventListener("ended", () => updatePreviewPlayButton(false));
audioPreview.addEventListener("loadedmetadata", () => {
	if (Number.isFinite(audioPreview.duration)) {
		previewTime.textContent = formatTime(audioPreview.duration * 1000);
	}
});

drawIdleWaveformLine();
renderPreviewWaveform();
initializePreviewScreen();
void syncNextDefaultRecordingName({ force: true });
// Bottom nav: divide into 5 equal zones and handle navigation clicks
(function setupBottomNavZones() {
	const bottomNav = document.querySelector('.bottom-nav');
	if (!bottomNav) return;

	const isInsightsPage = window.location.pathname.includes('insights.html');

	bottomNav.addEventListener('click', (ev) => {
		const rect = bottomNav.getBoundingClientRect();
		const x = ev.clientX - rect.left;
		if (rect.width <= 0) return;
		const zone = Math.floor((x / rect.width) * 5);

		// zone indexes: 0..4 (left to right)
		if (zone === 1) {
			// second icon (record) -> go back to recording page/view
			if (isInsightsPage) {
				window.location.href = '../index.html';
			} else {
				showRecordingView();
			}
		} else if (zone === 3) {
			// second from right (folder/insights) -> open Insights page
			if (!isInsightsPage) {
				window.location.href = 'frontend/insights.html';
			}
		}
	});
})();
