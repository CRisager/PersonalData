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
const previewWaveform = document.getElementById("previewWaveform");
const previewCtx = previewWaveform.getContext("2d");
const waveformCanvas = document.getElementById("waveform");
const canvasCtx = waveformCanvas.getContext("2d");

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
let recordingIndex = 132;
let isSaving = false;

const PEAK_CAPTURE_INTERVAL_MS = 45;
const MAX_STORED_PEAKS = 50000;
const WAVEFORM_BAR_SPACING = 3;
const API_BASE_STORAGE_KEY = "speechApiBase";

function normalizeApiBase(value) {
	return String(value || "").trim().replace(/\/+$/, "");
}

function resolveApiBase() {
	const params = new URLSearchParams(window.location.search);
	const queryBase = normalizeApiBase(params.get("apiBase"));
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
	idleView.hidden = true;
	recordingView.hidden = false;
	confirmView.hidden = true;
	pageTitle.textContent = "New recording";
	setStep("record");
	document.body.classList.add("is-recording");
}

function showIdleView() {
	confirmView.hidden = true;
	recordingView.hidden = true;
	idleView.hidden = false;
	pageTitle.textContent = "New recording";
	setStep("record");
	document.body.classList.remove("is-recording");
}

function showConfirmView() {
	idleView.hidden = true;
	recordingView.hidden = true;
	confirmView.hidden = false;
	syncHeaderWithName();
	renderPreviewWaveform();
	previewTime.textContent = formatTime(recordedDurationMs);
	setStep("confirm");
	document.body.classList.remove("is-recording");
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

	try {
		stream = await navigator.mediaDevices.getUserMedia({ audio: true });
	} catch (error) {
		alert("Microphone access is required to record.");
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
	recordingIndex += 1;
	recordingName.value = `Recording_${recordingIndex}`;

	showConfirmView();
	resetUiAfterStop();
}

function clearPreviewAndReturnToIdle() {
	setPreviewAudio(null);
	recordedChunks = [];
	recordedDurationMs = 0;
	setSaveStatus("");
	showIdleView();
}

async function saveRecording() {
	if (isSaving) {
		return;
	}

	if (!recordedAudioBlob) {
		setSaveStatus("No recording to save yet.", true);
		return;
	}

	setSavingState(true);
	setSaveStatus("Uploading and processing recording...");

	try {
		const wavBlob = await convertBlobToWav(recordedAudioBlob);
		const safeName = sanitizeFilename(recordingName.value);
		const formData = new FormData();
		formData.append("audio", wavBlob, `${safeName}.wav`);
		formData.append("recording_name", recordingName.value.trim() || "Recording");
		formData.append("category", document.getElementById("categorySelect").value || "");

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
		const wpmText = Number.isFinite(result.wpm) ? result.wpm.toFixed(2) : "n/a";
		setStep("learn");
		setSaveStatus(`Saved and processed. WPM: ${wpmText}`);
		clearPreviewAndReturnToIdle();
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

startButton.addEventListener("click", startRecording);
pauseButton.addEventListener("click", togglePause);
stopButton.addEventListener("click", stopRecording);
deleteButton.addEventListener("click", clearPreviewAndReturnToIdle);
saveButton.addEventListener("click", saveRecording);
recordingName.addEventListener("input", syncHeaderWithName);
editNameButton.addEventListener("click", () => recordingName.focus());
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
