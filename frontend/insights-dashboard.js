(function () {
	const API_BASE_STORAGE_KEY = "speechApiBase";
	const INSIGHTS_CACHE_STORAGE_KEY = "insightsDataCache";
	const COUNT_OPTIONS = [
		{ label: "Last 5", value: 5, selected: true },
		{ label: "Last 10", value: 10 },
		{ label: "Last 15", value: 15 },
		{ label: "Last 20", value: 20 },
		{ label: "All recordings", value: "all" },
	];

	const PALETTE = ['#C8E6C9', '#B8E2D4', '#A6DDE8', '#C5D9F2', '#D6C7EE', '#9FD8F7', '#B5EAD7', '#FFDAB9', '#FFD4B4', '#E2C7EE', '#D4E6F1', '#F0E2C3', '#E8D5B7', '#C9E4CA'];

	const PLOT_DEFINITIONS = {
		pace: {
			key: "pace",
			title: "Words per minute over time",
			subtitle: "Compare speaking pace across recent recordings.",
			yAxisTitle: "Words per minute",
			kind: "line",
			getValue: (recording) => Number(recording.wpm),
			formatValue: (value) => `${value.toFixed(1)} WPM`,
		},
		fillerTrend: {
			key: "fillerTrend",
			title: "Filler words over time",
			subtitle: "Compare filler-word percentage across recordings.",
			yAxisTitle: "Filler words (%)",
			kind: "line",
			getValue: (recording) => Number(recording.filler_percentage),
			formatValue: (value) => `${value.toFixed(1)}%`,
		},
		pitch: {
			key: "pitch",
			title: "Pitch range over time",
			subtitle: "Compare pitch variation across recordings.",
			yAxisTitle: "Pitch variation (semitones)",
			kind: "line",
			getValue: (recording) => Number(recording.pitch_range),
			formatValue: (value) => `${value.toFixed(1)} st`,
		},
		fillerRecording: {
			key: "fillerRecording",
			title: "Filler words per recording",
			subtitle: "Compare filler-word percentage within individual recordings.",
			yAxisTitle: "Filler words (%)",
			kind: "bar",
			getValue: (recording) => Number(recording.filler_percentage),
			formatValue: (value) => `${value.toFixed(1)}%`,
		},
	};

	function collectFillerWords(recordings) {
		const set = new Set();
		for (const r of recordings || []) {
			const fc = r && r.filler_counts;
			if (fc && typeof fc === 'object') {
				for (const k of Object.keys(fc)) set.add(String(k));
			}
		}
		return Array.from(set).sort();
	}

	const OVERVIEW_PLOTS = [
		PLOT_DEFINITIONS.pace,
		PLOT_DEFINITIONS.fillerTrend,
		PLOT_DEFINITIONS.pitch,
		PLOT_DEFINITIONS.fillerRecording,
	];

	const phoneStage = document.getElementById("phoneStage");
	const BASE_PHONE_WIDTH = 404;
	const BASE_PHONE_HEIGHT = 873.7301635742188;
	const VIEWPORT_PAD = 16;

	function fitPhoneToViewport() {
		if (!phoneStage) {
			return;
		}

		const availableWidth = Math.max(window.innerWidth - VIEWPORT_PAD * 2, 1);
		const availableHeight = Math.max(window.innerHeight - VIEWPORT_PAD * 2, 1);
		const widthScale = availableWidth / BASE_PHONE_WIDTH;
		const heightScale = availableHeight / BASE_PHONE_HEIGHT;
		const scale = Math.min(1, widthScale, heightScale);

		phoneStage.style.setProperty("--phone-scale", String(scale));
	}

	fitPhoneToViewport();
	window.addEventListener("resize", fitPhoneToViewport);

	function normalizeApiBase(value) {
		return String(value || "").trim().replace(/\/+$/, "");
	}

	function resolveApiBase() {
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

	function formatRecordingLabel(value) {
		if (!value) {
			return "Unknown date";
		}

		const date = new Date(value);
		if (Number.isNaN(date.getTime())) {
			return String(value);
		}

		return new Intl.DateTimeFormat("en", {
			month: "short",
			day: "numeric",
			year: "numeric",
		}).format(date);
	}

	function extractDateFromRecordingName(recordingName) {
		const match = String(recordingName || "").match(/(\d{8})_(\d{6})$/);
		if (!match) {
			return null;
		}

		const datePart = match[1];
		const timePart = match[2];
		const isoValue = `${datePart.slice(0, 4)}-${datePart.slice(4, 6)}-${datePart.slice(6, 8)}T${timePart.slice(0, 2)}:${timePart.slice(2, 4)}:${timePart.slice(4, 6)}`;
		const parsed = new Date(isoValue);
		return Number.isNaN(parsed.getTime()) ? null : parsed;
	}

	function getRecordingDate(recording) {
		const fromName = extractDateFromRecordingName(recording.recording_name);
		if (fromName) {
			return fromName;
		}

		const fromCreatedAt = new Date(recording.created_at);
		if (!Number.isNaN(fromCreatedAt.getTime())) {
			return fromCreatedAt;
		}

		return null;
	}

	function toChartDateValue(recording, fallbackIndex) {
		const date = getRecordingDate(recording);
		if (!date) {
			return new Date(Date.UTC(2000, 0, 1 + fallbackIndex)).toISOString();
		}

		return date.toISOString();
	}

	function formatRecordingWithDateLabel(recording, fallbackIndex) {
		const date = getRecordingDate(recording);
		const timeLabel = !date || Number.isNaN(date.getTime())
			? `#${fallbackIndex + 1}`
			: new Intl.DateTimeFormat("en", {
				month: "short",
				day: "numeric",
				hour: "2-digit",
				minute: "2-digit",
				hour12: false,
			}).format(date);

		return `${recording.recording_name} - ${timeLabel}`;
	}

	function parseCountSelection(value) {
		if (value === "all") {
			return "all";
		}

		const count = Number(value);
		if (Number.isFinite(count) && count > 0) {
			return Math.floor(count);
		}

		return 10;
	}

	function getVisibleRecordings(recordings, countValue) {
		if (!Array.isArray(recordings) || recordings.length === 0) {
			return [];
		}

		const selection = parseCountSelection(countValue);
		if (selection === "all") {
			return recordings;
		}

		return recordings.slice(Math.max(0, recordings.length - selection));
	}

	function sortRecordingsByDate(recordings) {
		return [...recordings].sort((left, right) => {
			const leftDate = getRecordingDate(left);
			const rightDate = getRecordingDate(right);

			if (leftDate && rightDate) {
				const dateDelta = leftDate.getTime() - rightDate.getTime();
				if (dateDelta !== 0) {
					return dateDelta;
				}
			}

			if (leftDate && !rightDate) {
				return -1;
			}

			if (!leftDate && rightDate) {
				return 1;
			}

			return String(left.recording_name || "").localeCompare(String(right.recording_name || ""));
		});
	}

	function loadCachedInsightsData() {
		try {
			const parsed = JSON.parse(sessionStorage.getItem(INSIGHTS_CACHE_STORAGE_KEY) || "[]");
			return Array.isArray(parsed) ? parsed : [];
		} catch (_error) {
			return [];
		}
	}

	function saveCachedInsightsData(recordings) {
		try {
			sessionStorage.setItem(INSIGHTS_CACHE_STORAGE_KEY, JSON.stringify(Array.isArray(recordings) ? recordings : []));
		} catch (_error) {
			// Ignore storage limits or disabled storage.
		}
	}

	async function loadStaticInsightsData() {
		const response = await fetch(buildApiUrl("/frontend/insights-data.json"));
		if (!response.ok) {
			throw new Error("Unable to load insights data.");
		}

		const payload = await response.json();
		return Array.isArray(payload.recordings) ? payload.recordings : [];
	}

	async function loadInsightsData() {
		try {
			const response = await fetch(buildApiUrl("/api/insights-data"));
			if (!response.ok) {
				throw new Error("Unable to load insights data.");
			}

			const payload = await response.json();
			const recordings = Array.isArray(payload.recordings) ? payload.recordings : [];
			saveCachedInsightsData(recordings);
			return recordings;
		} catch (error) {
			try {
				const staticRecordings = await loadStaticInsightsData();
				saveCachedInsightsData(staticRecordings);
				return staticRecordings;
			} catch (_staticError) {
				// Fall through to session cache.
			}

			const cachedRecordings = loadCachedInsightsData();
			if (cachedRecordings.length > 0) {
				return cachedRecordings;
			}

			throw error;
		}
	}

	function createCountSelect(selectedValue) {
		const select = document.createElement("select");
		select.className = "insight-count-select";
		for (const option of COUNT_OPTIONS) {
			const item = document.createElement("option");
			item.value = String(option.value);
			item.textContent = option.label;
			if (String(option.value) === String(selectedValue)) {
				item.selected = true;
			}
			select.appendChild(item);
		}
		return select;
	}

	function buildLineTrace(definition, recordings) {
		const subset = sortRecordingsByDate(getVisibleRecordings(recordings, definition.selectedCount));
		return {
			type: "scatter",
			mode: "lines+markers",
			x: subset.map((recording, index) => toChartDateValue(recording, index)),
			y: subset.map((recording) => definition.getValue(recording)),
			customdata: subset.map((recording, index) => ({
				recordingName: recording.recording_name,
				label: formatRecordingLabel(toChartDateValue(recording, index)),
			})),
			line: {
				color: "#1e8486",
				width: 2.5,
			},
			marker: {
				color: "#21afa2",
				size: 6,
			},
			hovertemplate: `%{customdata.recordingName}<br>%{customdata.label}<br>${definition.yAxisTitle}: %{y:.1f}<extra></extra>`,
		};
	}

	function buildWpmAndPitchTraces(definition, recordings) {
		const ordered = sortRecordingsByDate(recordings);
		const dd = getVisibleRecordings(ordered, definition.selectedCount);
		if (definition.key === 'pace') {
			// Build WPM traces: points, trend, avg and target zone
			const dates = dd.map((r, i) => toChartDateValue(r, i));
			const wpm_v = dd.map((r) => Number(r.wpm));
			const avg = wpm_v.length ? wpm_v.reduce((a,b) => a+b,0)/wpm_v.length : NaN;
			let trend = wpm_v.slice();
			if (wpm_v.length >= 2) {
				// simple linear fit (least squares) in JS using numeric approach
				const xs = dates.map((d) => new Date(d).getTime());
				const n = xs.length;
				const sx = xs.reduce((a,b)=>a+b,0);
				const sy = wpm_v.reduce((a,b)=>a+b,0);
				const sx2 = xs.reduce((a,b)=>a+b*b,0);
				const sxy = xs.reduce((a,b,i)=>a + b * wpm_v[i],0);
				const denom = n * sx2 - sx * sx;
				if (denom !== 0) {
					const m = (n * sxy - sx * sy) / denom;
					const c = (sy - m * sx) / n;
					trend = xs.map((x) => m * x + c).map((v) => v);
				}
			}
			if (trend.length !== wpm_v.length) trend = wpm_v.slice();
			return {
				traces: [
					{
						type: 'scatter', mode: 'markers', name: 'WPM', x: dates, y: wpm_v,
						marker: { size: 8, color: PALETTE[2] },
						customdata: dd.map((r,i)=>[r.recording_name, r.wpm]),
						hovertemplate: '<b>%{customdata[0]}</b><br>Date: %{x|%b %d, %Y}<br>WPM: %{customdata[1]:.1f}<extra></extra>'
					},
					{
						type: 'scatter', mode: 'lines', name: 'Trend', x: dates, y: trend,
						line: { color: PALETTE[4], width: 2.5, dash: 'dash' }
					},
					{
						type: 'scatter', mode: 'lines', name: `Average WPM (${isNaN(avg)?'':avg.toFixed(1)})`, x: dates, y: dates.map(()=>avg),
						line: { color: '#4E5A67', width: 2, dash: 'dot' }
					}
				],
				layoutExtras: {
					shapes: [
						{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 130, y1: 150, fillcolor: 'rgba(197,217,242,0.25)', line: { width:0 } }
					],
					yRange: [0, Math.max(180, (Math.max(...wpm_v, 0) || 150) * 1.18)]
				}
			};
		} else if (definition.key === 'pitch') {
			const orderedPitch = dd.filter((r)=>Number.isFinite(Number(r.pitch_mean_semitones)));
			const dates = orderedPitch.map((r,i)=>toChartDateValue(r,i));
			const avg = orderedPitch.map((r)=>Number(r.pitch_mean_semitones));
			const minv = orderedPitch.map((r)=>Number(r.pitch_min_semitones));
			const maxv = orderedPitch.map((r)=>Number(r.pitch_max_semitones));
			const overallAvg = avg.length ? avg.reduce((a,b)=>a+b,0)/avg.length : NaN;
			const upper = Math.max(7, ...maxv.map((v)=>v*1.15));
			return {
				traces: [
					{
						type: 'scatter', mode: 'lines', name: 'Min-max band',
						x: dates.concat(dates.slice().reverse()),
						y: maxv.concat(minv.slice().reverse()),
						fill: 'toself', fillcolor: PALETTE[1], opacity: 0.38, line: { color: 'rgba(0,0,0,0)' }, hoverinfo: 'skip'
					},
					{
						type: 'scatter', mode: 'lines+markers', name: 'Average variation', x: dates, y: avg,
						line: { color: PALETTE[2], width: 2.5 }, marker: { size: 6 },
						hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Avg: %{y:.2f} st<extra></extra>'
					},
					{
						type: 'scatter', mode: 'lines', name: `Overall average (${isNaN(overallAvg)?'':overallAvg.toFixed(2)})`, x: dates, y: dates.map(()=>overallAvg),
						line: { color: '#4E5A67', width: 2, dash: 'dash' }
					}
				],
				layoutExtras: {
					shapes: [ { type: 'rect', xref: 'x', x0: dates[0]||null, x1: dates[dates.length-1]||null, yref: 'y', y0: 3.0, y1: 5.0, fillcolor: PALETTE[0], opacity: 0.2, line: { width: 0 } } ],
					yRange: [0, upper]
				}
			};
		}
		return null;
	}

	function buildBarTrace(definition, recordings) {
		const subset = sortRecordingsByDate(getVisibleRecordings(recordings, definition.selectedCount));
		const xLabels = subset.map((recording, index) => formatRecordingWithDateLabel(recording, index));
		return {
			type: "bar",
			x: xLabels,
			y: subset.map((recording) => definition.getValue(recording)),
			customdata: subset.map((recording, index) => ({
				recordingName: recording.recording_name,
				label: formatRecordingLabel(toChartDateValue(recording, index)),
				fullLabel: xLabels[index],
			})),
			marker: {
				color: "#21afa2",
			},
			text: subset.map((recording) => definition.formatValue(definition.getValue(recording))),
			textposition: "outside",
			hovertemplate: "%{customdata.recordingName}<br>%{customdata.label}<br>%{y:.1f}<extra></extra>",
		};
	}

	function buildLayout(definition, isPreview) {
		const titleFontSize = isPreview ? 12 : 14;
		const tickFontSize = isPreview ? 8 : 10;
		const margin = isPreview
			? { l: 42, r: 12, t: 8, b: 34 }
			: { l: 52, r: 20, t: 16, b: 50 };

		return {
			margin,
			paper_bgcolor: "rgba(255,255,255,0)",
			plot_bgcolor: "#ffffff",
			showlegend: false,
			font: {
				family: '"IBM Plex Mono", "Space Mono", Courier, monospace',
				color: "#666563",
				size: tickFontSize,
			},
			xaxis: {
				automargin: true,
				title: isPreview ? "" : { text: "Recording", font: { size: titleFontSize, color: "#666563" } },
				tickfont: { size: tickFontSize, color: "#666563" },
				gridcolor: "#ececec",
				zeroline: false,
			},
			yaxis: {
				automargin: true,
				title: isPreview ? "" : { text: definition.yAxisTitle, font: { size: titleFontSize, color: "#666563" } },
				tickfont: { size: tickFontSize, color: "#666563" },
				gridcolor: "#ececec",
				zeroline: false,
			},
			transition: {
				duration: 180,
				easing: "cubic-in-out",
			},
		};
	}

	function renderPlotInto(container, definition, recordings, selectedCount, isPreview) {
		if (!container) {
			return;
		}

		if (!Array.isArray(recordings) || recordings.length === 0) {
			container.innerHTML = '<div class="dashboard-empty">No recordings found yet.</div>';
			return;
		}

		definition.selectedCount = selectedCount;
		const trace = definition.kind === "bar"
			? buildBarTrace(definition, recordings)
			: buildLineTrace(definition, recordings);

		const visibleRecordings = getVisibleRecordings(recordings, selectedCount);
		const visibleRecordingsByDate = sortRecordingsByDate(visibleRecordings);
		const layout = buildLayout(definition, isPreview);
		const containerWidth = Math.max(Math.floor(container.getBoundingClientRect().width || container.clientWidth || 0), 1);
		layout.height = isPreview ? 170 : 360;
		layout.width = isPreview ? containerWidth : undefined;
		layout.autosize = true;
		layout.xaxis.tickangle = isPreview ? -20 : 0;
		layout.xaxis.nticks = isPreview ? 4 : undefined;
		layout.xaxis.autorange = true;
		layout.xaxis.range = undefined;
		layout.yaxis.nticks = isPreview ? 3 : undefined;
		if (isPreview && visibleRecordingsByDate.length > 0) {
			const visibleValues = visibleRecordingsByDate.map((recording) => Number(definition.getValue(recording)) || 0);
			const minValue = Math.min(...visibleValues);
			const maxValue = Math.max(...visibleValues);
			const valueSpan = Math.max(1, maxValue - minValue);
			const valuePad = Math.max(1, valueSpan * 0.18);
			layout.yaxis.range = [Math.min(0, minValue - valuePad), maxValue + valuePad];
		} else {
			layout.yaxis.range = undefined;
		}

		if (definition.kind === "line") {
			layout.xaxis.type = "date";
			layout.xaxis.tickformat = isPreview ? "%b %d" : "%b %d, %Y";
			if (visibleRecordingsByDate.length > 0) {
				const firstX = new Date(toChartDateValue(visibleRecordingsByDate[0], 0)).getTime();
				const lastX = new Date(toChartDateValue(visibleRecordingsByDate[visibleRecordingsByDate.length - 1], visibleRecordingsByDate.length - 1)).getTime();
				const span = Math.max(1, lastX - firstX);
				const pad = Math.max(isPreview ? 30 * 60 * 1000 : 60 * 1000, Math.round(span * (isPreview ? 0.18 : 0.06)));
				layout.xaxis.range = [new Date(firstX - pad).toISOString(), new Date(lastX + pad).toISOString()];
			}
		}

		if (isPreview) {
			layout.dragmode = false;
			layout.xaxis.fixedrange = true;
			layout.yaxis.fixedrange = true;
		}

		trace.cliponaxis = false;

		// Special-case detailed definitions to produce multiple traces and extras
		let traces = [trace];
		let extraLayout = {};
		if (!isPreview) {
			if (definition.key === 'pace' || definition.key === 'pitch') {
				const result = buildWpmAndPitchTraces(definition, recordings);
				if (result) {
					traces = result.traces;
					extraLayout.shapes = result.layoutExtras && result.layoutExtras.shapes;
					if (result.layoutExtras && result.layoutExtras.yRange) {
						layout.yaxis.range = result.layoutExtras.yRange;
					}
				}
			} else if (definition.key === 'fillerRecording') {
				// Build stacked horizontal bars using filler_counts
				const fillerWords = collectFillerWords(recordings);
				const selected = sortRecordingsByDate(getVisibleRecordings(recordings, selectedCount));
				const labels = selected.map((r,i)=>formatRecordingWithDateLabel(r,i));
				traces = fillerWords.map((word, idx) => {
					const values = selected.map((r)=> {
						const tot = Number(r.total_words) || 1;
						const count = r.filler_counts && r.filler_counts[word] ? Number(r.filler_counts[word]) : 0;
						return (count / Math.max(1, tot)) * 100.0;
					});
					return {
						type: 'bar', orientation: 'h', name: word, y: labels, x: values, marker: { color: PALETTE[idx % PALETTE.length] },
						customdata: selected.map((r)=>[word, r.recording_name, r.total_words]),
						hovertemplate: '<b>%{customdata[1]}</b><br>Filler: %{customdata[0]}<br>Share: %{x:.2f}%<extra></extra>',
					};
				});
				layout.barmode = 'stack';
				layout.margin = { l: 240, r: 35, t: 115, b: 110 };
				layout.yaxis = layout.yaxis || {};
				layout.yaxis.categoryorder = 'array';
				layout.yaxis.categoryarray = labels;
				layout.xaxis = layout.xaxis || {};
				layout.xaxis.ticksuffix = '%';
				layout.title = { text: `Filler words as percentages per recording`, x:0.0, xanchor:'left' };
			}
		}

		Plotly.react(container, traces, Object.assign({}, layout, extraLayout), {
			displayModeBar: false,
			responsive: true,
		});

		if (isPreview) {
			window.requestAnimationFrame(() => {
				Plotly.Plots.resize(container);
			});
		}
	}

	function createCard(definition, recordings, initialSelection) {
		const card = document.createElement("section");
		card.className = "insight-card";
		card.tabIndex = 0;
		card.setAttribute("role", "button");
		card.setAttribute("aria-label", `${definition.title} details`);

		const header = document.createElement("div");
		header.className = "insight-card-header";

		const titleWrap = document.createElement("div");
		titleWrap.className = "insight-card-title-wrap";

		const title = document.createElement("h2");
		title.className = "insight-card-title";
		title.textContent = definition.title;

		const subtitle = document.createElement("p");
		subtitle.className = "insight-card-subtitle";
		subtitle.textContent = definition.subtitle;

		titleWrap.append(title, subtitle);

		const controls = document.createElement("div");
		controls.className = "insight-card-controls";
		const select = createCountSelect(initialSelection);
		controls.appendChild(select);

		header.append(titleWrap, controls);

		const chart = document.createElement("div");
		chart.className = "insight-card-chart";

		card.append(header, chart);

		const rerender = () => {
			renderPlotInto(chart, definition, recordings, select.value, true);
		};

		const scheduleInitialRerender = () => {
			window.requestAnimationFrame(() => {
				window.requestAnimationFrame(rerender);
			});
		};

		select.addEventListener("click", (event) => event.stopPropagation());
		select.addEventListener("change", (event) => {
			event.stopPropagation();
			rerender();
		});

		card.addEventListener("click", () => {
			window.location.href = `plot.html?plot=${encodeURIComponent(definition.key)}&count=${encodeURIComponent(select.value)}`;
		});
		card.addEventListener("keydown", (event) => {
			if (event.key !== "Enter" && event.key !== " ") {
				return;
			}

			event.preventDefault();
			window.location.href = `plot.html?plot=${encodeURIComponent(definition.key)}&count=${encodeURIComponent(select.value)}`;
		});

		scheduleInitialRerender();
		return card;
	}

	function setNavState(viewName) {
		const recordIcon = document.querySelector(".bottom-nav .record-icon");
		const searchIcon = document.querySelector(".bottom-nav .icon.search");
		const bottomNav = document.querySelector(".bottom-nav");

		if (recordIcon) {
			recordIcon.style.cursor = "pointer";
			recordIcon.addEventListener("click", (event) => {
				event.stopPropagation();
				window.location.href = "../index.html";
			});
		}

		if (searchIcon) {
			searchIcon.style.cursor = "pointer";
			searchIcon.addEventListener("click", (event) => {
				event.stopPropagation();
				if (viewName === "detail") {
					window.location.href = "insights.html";
					return;
				}

				window.location.reload();
			});
		}

		if (bottomNav && viewName === "overview") {
			bottomNav.setAttribute("aria-label", "Primary navigation");
		}
	}

	function applyActiveNavStyles(viewName) {
		document.body.dataset.view = viewName;
	}

	function getPageContext() {
		const params = new URLSearchParams(window.location.search);
		const plot = String(params.get("plot") || "pace");
		const count = params.get("count") || "10";
		const view = document.body.dataset.view || "overview";
		return { plot, count, view };
	}

	function selectPlotDefinition(plotKey) {
		return PLOT_DEFINITIONS[plotKey] || PLOT_DEFINITIONS.pace;
	}

	async function initializeOverview(recordings) {
		const app = document.getElementById("insightsApp");
		if (!app) {
			return;
		}

		app.innerHTML = "";
		for (const definition of OVERVIEW_PLOTS) {
			app.appendChild(createCard(definition, recordings, 5));
		}
	}

	function buildDetailShell(recordings, definition, selectedCount) {
		const app = document.getElementById("plotApp");
		const pageTitle = document.getElementById("pageTitle");
		const backButton = document.getElementById("plotBackButton");
		if (!app || !pageTitle) {
			return;
		}

		document.title = `Personal Data - ${definition.title}`;
		if (backButton) {
			backButton.addEventListener("click", () => {
				window.location.href = "insights.html";
			});
		}

		app.innerHTML = "";

		const panel = document.createElement("section");
		panel.className = "plot-panel";

		const toolbar = document.createElement("div");
		toolbar.className = "plot-toolbar";

		const titleWrap = document.createElement("div");
		titleWrap.className = "plot-toolbar-title-wrap";

		const title = document.createElement("h2");
		title.className = "plot-toolbar-title";
		title.textContent = definition.title;

		const subtitle = document.createElement("p");
		subtitle.className = "plot-toolbar-subtitle";
		subtitle.textContent = definition.subtitle;

		titleWrap.append(title, subtitle);

		const intro = document.createElement("p");
		intro.className = "plot-intro";
		intro.textContent = "Use the dropdown to adjust how much data you want to see. \n Hover over the plot to see specifics regarding each data point.";

		const controls = document.createElement("div");
		controls.className = "plot-toolbar-controls";
		const select = createCountSelect(selectedCount);
		controls.appendChild(select);

		toolbar.append(titleWrap, controls);

		const chartWrap = document.createElement("div");
		chartWrap.className = "plot-chart-wrap";

		panel.append(toolbar, chartWrap);

		const detailStack = document.createElement("div");
		detailStack.className = "plot-detail-stack";
		detailStack.append(intro, panel);

		app.appendChild(detailStack);

		const rerender = () => {
			renderPlotInto(chartWrap, definition, recordings, select.value, false);
		};

		select.addEventListener("change", rerender);
		select.addEventListener("click", (event) => event.stopPropagation());
		rerender();
	}

	function initializeLoadingState(viewName) {
		const app = document.getElementById(viewName === "detail" ? "plotApp" : "insightsApp");
		if (app) {
			app.innerHTML = '<div class="dashboard-loading">Loading insights...</div>';
		}
	}

	function initializeErrorState(viewName, message) {
		const app = document.getElementById(viewName === "detail" ? "plotApp" : "insightsApp");
		if (app) {
			app.innerHTML = `<div class="dashboard-error">${message}</div>`;
		}
	}

	async function start() {
		const context = getPageContext();
		applyActiveNavStyles(context.view);
		setNavState(context.view);
		initializeLoadingState(context.view);

		try {
			const recordings = await loadInsightsData();
			if (context.view === "detail") {
				const definition = selectPlotDefinition(context.plot);
				buildDetailShell(recordings, definition, context.count);
			} else {
				await initializeOverview(recordings);
			}
		} catch (error) {
			initializeErrorState(context.view, error.message || "Unable to load insights.");
		}
	}

	if (typeof Plotly === "undefined") {
		initializeErrorState(document.body.dataset.view || "overview", "Plotly could not be loaded.");
		return;
	}

	start();
})();