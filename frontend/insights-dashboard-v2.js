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

	const TIMEFRAME_OPTIONS = [
		{ label: "This week", value: "this_week" },
		{ label: "Month", value: "month" },
		{ label: "6 months", value: "six_months" },
		{ label: "Year", value: "year" },
		{ label: "All time", value: "all_time" },
	];

	const PALETTE = ['#C8E6C9', '#B8E2D4', '#A6DDE8', '#C5D9F2', '#D6C7EE', '#9FD8F7', '#B5EAD7', '#FFDAB9', '#FFD4B4', '#E2C7EE', '#D4E6F1', '#F0E2C3', '#E8D5B7', '#C9E4CA'];
	const FILLER_PALETTE = [
		'#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
		'#a65628', '#f781bf', '#999999', '#1b9e77', '#d95f02',
		'#7570b3', '#e7298a', '#66a61e', '#a6761d',
	];

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
			title: "Pitch variation over time",
			subtitle: "Compare pitch variation across recordings.",
			yAxisTitle: "Pitch variation (semitones from session median)",
			kind: "line",
			getValue: (recording) => Number(recording.pitch_range),
			formatValue: (value) => `${value.toFixed(1)} st`,
		},
		fillerRecording: {
			key: "fillerRecording",
			title: "Filler words per recording",
			subtitle: "Compare filler-word percentage within individual recordings.",
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
		PLOT_DEFINITIONS.fillerTrend,
		PLOT_DEFINITIONS.fillerRecording,
		PLOT_DEFINITIONS.pace,
		PLOT_DEFINITIONS.pitch,
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

	function createTimeframeSelect(selectedValue) {
		const select = document.createElement("select");
		select.className = "insight-count-select";
		for (const option of TIMEFRAME_OPTIONS) {
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

	function getDateRangeForTimeframe(timeframeKey) {
		const today = new Date();
		const dayStart = new Date(today.getFullYear(), today.getMonth(), today.getDate());
		const weekday = dayStart.getDay();
		const mondayOffset = (weekday + 6) % 7;
		const mondayThisWeek = new Date(dayStart);
		mondayThisWeek.setDate(dayStart.getDate() - mondayOffset);
		const sundayThisWeek = new Date(mondayThisWeek);
		sundayThisWeek.setDate(mondayThisWeek.getDate() + 6);

		const mondayLastWeek = new Date(mondayThisWeek);
		mondayLastWeek.setDate(mondayThisWeek.getDate() - 7);
		const sundayLastWeek = new Date(mondayLastWeek);
		sundayLastWeek.setDate(mondayLastWeek.getDate() + 6);

		switch (String(timeframeKey || "all_time")) {
			case "this_week":
				return { start: mondayThisWeek, end: sundayThisWeek };
			case "last_week":
				return { start: mondayLastWeek, end: sundayLastWeek };
			case "month":
				return { start: new Date(dayStart.getFullYear(), dayStart.getMonth(), 1), end: dayStart };
			case "six_months": {
				const start = new Date(dayStart);
				start.setMonth(start.getMonth() - 6);
				return { start, end: dayStart };
			}
			case "year":
				return { start: new Date(dayStart.getFullYear(), 0, 1), end: dayStart };
			case "all_time":
			default:
				return { start: null, end: null };
		}
	}

	function applyTimeframe(recordings, timeframeKey) {
		const ordered = sortRecordingsByDate(recordings || []);
		const { start, end } = getDateRangeForTimeframe(timeframeKey);
		if (!start || !end) {
			return ordered;
		}
		return ordered.filter((r) => {
			const d = getRecordingDate(r);
			if (!d) return false;
			return d >= start && d <= new Date(end.getFullYear(), end.getMonth(), end.getDate(), 23, 59, 59, 999);
		});
	}

	function getTimeframeLabel(timeframeKey) {
		const found = TIMEFRAME_OPTIONS.find((opt) => opt.value === timeframeKey);
		return found ? found.label : "All time";
	}

	function getTimeframeTickConfig(timeframeKey) {
		switch (String(timeframeKey || "all_time")) {
			case "this_week":
			case "last_week":
				return { tickformat: "%a", dtick: 86400000 };
			case "month":
				return { tickformat: "%b %d", dtick: null };
			case "six_months":
				return { tickformat: "%b", dtick: "M1" };
			case "year":
				return { tickformat: "%b", dtick: "M1" };
			case "all_time":
			default:
				return { tickformat: "%b %Y", dtick: "M1" };
		}
	}

	function getTimeframeXRange(allRecords, filteredRecords, timeframeKey) {
		const allDates = (allRecords || []).map((r) => getRecordingDate(r)).filter(Boolean);
		if (!allDates.length) {
			return [null, null];
		}

		if (String(timeframeKey || "all_time") === "all_time") {
			const minDate = new Date(Math.min(...allDates.map((d) => d.getTime())));
			const maxDate = new Date(Math.max(...allDates.map((d) => d.getTime())));
			const padMs = 20 * 24 * 60 * 60 * 1000;
			return [new Date(minDate.getTime() - padMs).toISOString(), new Date(maxDate.getTime() + padMs).toISOString()];
		}

		const { start, end } = getDateRangeForTimeframe(timeframeKey);
		if (!start || !end) {
			return [null, null];
		}
		const padMs = 24 * 60 * 60 * 1000;
		return [new Date(start.getTime() - padMs).toISOString(), new Date(end.getTime() + padMs).toISOString()];
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
						type: 'scatter', mode: 'lines', name: 'Trend line', x: dates, y: trend,
						line: { color: PALETTE[4], width: 2.5, dash: 'dash' }
					},
					{
						type: 'scatter', mode: 'lines', name: 'Recommended band',
						x: [null], y: [null],
						line: { color: PALETTE[3], width: 3 },
						hoverinfo: 'skip', showlegend: true,
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
			const rangeCandidates = [...avg, ...minv, ...maxv].filter((value) => Number.isFinite(value));
			const maxAbsVariation = rangeCandidates.length
				? Math.max(5, ...rangeCandidates.map((value) => Math.abs(value)))
				: 5;
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
						customdata: dates.map((d, i) => [minv[i], maxv[i]]),
						hovertemplate: '<b>%{x|%b %d, %Y}</b><br>Avg: %{y:.2f} st<br>Min: %{customdata[0]:.2f} st<br>Max: %{customdata[1]:.2f} st<extra></extra>'
					},
					{
						type: 'scatter', mode: 'lines', name: `Overall average (${isNaN(overallAvg)?'':overallAvg.toFixed(2)})`, x: dates, y: dates.map(()=>overallAvg),
						line: { color: '#4E5A67', width: 2, dash: 'dash' }, hoverinfo: 'skip'
					}
				],
				layoutExtras: {
					shapes: [
						{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0, y1: 0, line: { color: '#666563', width: 1.5, dash: 'dot' } },
						{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 3.0, y1: 5.0, fillcolor: PALETTE[0], opacity: 0.22, line: { width: 0 } },
						{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: -5.0, y1: -3.0, fillcolor: PALETTE[0], opacity: 0.22, line: { width: 0 } },
					],
					yRange: [-Math.max(10, maxAbsVariation * 1.1), Math.max(10, maxAbsVariation * 1.1)]
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

	function renderPlotInto(container, definition, recordings, selectedCount, isPreview, timeframeKey, isOverviewCard = false) {
		if (!container) {
			return;
		}

		const isOverview = Boolean(isOverviewCard);
		const usePreviewLayout = isPreview && !isOverview;

		if (!Array.isArray(recordings) || recordings.length === 0) {
			container.innerHTML = '<div class="dashboard-empty">No recordings found yet.</div>';
			return;
		}

		definition.selectedCount = selectedCount;
		const trace = definition.kind === "bar"
			? buildBarTrace(definition, recordings)
			: buildLineTrace(definition, recordings);

		let visibleRecordings;
		if (usePreviewLayout) {
			const base = timeframeKey ? applyTimeframe(recordings, timeframeKey) : recordings;
			visibleRecordings = getVisibleRecordings(base, selectedCount);
		} else {
			visibleRecordings = applyTimeframe(recordings, timeframeKey);
		}
		const visibleRecordingsByDate = sortRecordingsByDate(visibleRecordings);
		const layout = buildLayout(definition, usePreviewLayout);
		const containerWidth = Math.max(Math.floor(container.getBoundingClientRect().width || container.clientWidth || 0), 1);
		layout.height = usePreviewLayout ? 260 : isOverview ? 300 : 360;
		layout.width = usePreviewLayout ? containerWidth : undefined;
		layout.autosize = true;
		layout.xaxis.tickangle = usePreviewLayout ? -20 : isOverview ? -10 : 0;
		layout.xaxis.nticks = usePreviewLayout ? 4 : isOverview ? 5 : undefined;
		layout.xaxis.autorange = true;
		layout.xaxis.range = undefined;
		layout.yaxis.nticks = usePreviewLayout ? 3 : isOverview ? 4 : undefined;
		if (usePreviewLayout && visibleRecordingsByDate.length > 0) {
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
			layout.xaxis.tickformat = usePreviewLayout ? "%b %d" : isOverview ? "%b %d" : "%b %d, %Y";
			if (visibleRecordingsByDate.length > 0) {
				const firstX = new Date(toChartDateValue(visibleRecordingsByDate[0], 0)).getTime();
				const lastX = new Date(toChartDateValue(visibleRecordingsByDate[visibleRecordingsByDate.length - 1], visibleRecordingsByDate.length - 1)).getTime();
				const span = Math.max(1, lastX - firstX);
				const pad = Math.max(usePreviewLayout ? 30 * 60 * 1000 : isOverview ? 12 * 60 * 60 * 1000 : 60 * 1000, Math.round(span * (usePreviewLayout ? 0.18 : isOverview ? 0.12 : 0.06)));
				layout.xaxis.range = [new Date(firstX - pad).toISOString(), new Date(lastX + pad).toISOString()];
			}
		}

		if (usePreviewLayout || isOverview) {
			layout.dragmode = false;
			layout.xaxis.fixedrange = true;
			layout.yaxis.fixedrange = true;
			layout.showlegend = false;
		}

		trace.cliponaxis = false;

		// Special-case detailed definitions to produce multiple traces and extras
		let traces = [trace];
		let extraLayout = {};
		let fillerLegendItems = null;

		// Preview: for fillerRecording, build the stacked breakdown (compact)
		if (usePreviewLayout && definition.key === 'fillerRecording') {
			const detailSetPreview = sortRecordingsByDate(getVisibleRecordings(recordings, definition.selectedCount));
			const selectedPreview = [...detailSetPreview].sort((a, b) => (Number(b.filler_percentage) || 0) - (Number(a.filler_percentage) || 0));
			const displayedPreview = selectedPreview.slice(0, 7);
			const labels = displayedPreview.map((r) => String(r.recording_name || "Unknown recording"));
			const dates = displayedPreview.map((r, i) => formatRecordingLabel(toChartDateValue(r, i)));
			const totals = displayedPreview.map((r) => Number(r.filler_percentage) || 0);
			const fillerWords = collectFillerWords(displayedPreview.length ? displayedPreview : recordings);
			traces = fillerWords.map((word, idx) => {
				const values = displayedPreview.map((r) => {
					const tot = Number(r.total_words) || 1;
					const count = r.filler_counts && r.filler_counts[word] ? Number(r.filler_counts[word]) : 0;
					return (count / Math.max(1, tot)) * 100.0;
				});
				return {
					type: 'bar', orientation: 'h', name: word, y: labels, x: values, marker: { color: PALETTE[idx % PALETTE.length] },
					showlegend: false,
					customdata: displayedPreview.map((r, i) => [word, labels[i], dates[i], totals[i], Number(r.total_words) || 0]),
					hovertemplate: '<b>%{customdata[1]}</b><br>Date: %{customdata[2]}<br>Filler word: %{customdata[0]}<br>Word share: %{x:.2f}%<br>Total filler: %{customdata[3]:.2f}%<br>Total words: %{customdata[4]:.0f}<extra></extra>',
				};
			});
			fillerLegendItems = traces.map((t, idx) => ({ word: t.name, color: t.marker ? t.marker.color : '#999', traceIndex: idx }));
			// Preview layout: compact
			layout.barmode = 'stack';
			layout.margin = { l: 0, r: 25, t: 20, b: 30 };
			layout.template = "plotly_white";
			// compute height so up to 7 bars fit
			const maxVisibleBars = 7;
			const fixedViewportHeight = 220;
			const availableHeightForBars = fixedViewportHeight - layout.margin.b - layout.margin.t;
			const barHeightPerRecording = availableHeightForBars / maxVisibleBars;
			layout.height = Math.max(120, layout.margin.t + displayedPreview.length * barHeightPerRecording + layout.margin.b);
			layout.showlegend = false;
		}
		// Build a detailSet for detailed views or a visible subset for previews
		const detailSet = usePreviewLayout
			? sortRecordingsByDate(getVisibleRecordings(recordings, definition.selectedCount))
			: applyTimeframe(recordings, timeframeKey);
		const timeframeLabel = getTimeframeLabel(timeframeKey);
		const tickConfig = getTimeframeTickConfig(timeframeKey);
		const xRange = getTimeframeXRange(recordings, detailSet, timeframeKey);

		// For all non-fillerRecording-preview cases, run the detailed trace builder
		if (!(usePreviewLayout && definition.key === 'fillerRecording')) {
			layout.template = "plotly_white";
			if (!usePreviewLayout) layout.height = isOverview ? 300 : 600;

			if (definition.key === 'pace' || definition.key === 'pitch') {
				const result = buildWpmAndPitchTraces(definition, detailSet);
				if (result) {
					traces = result.traces;
					extraLayout.shapes = result.layoutExtras && result.layoutExtras.shapes;
					if (result.layoutExtras && result.layoutExtras.yRange) {
						layout.yaxis.range = result.layoutExtras.yRange;
					}
					if (!traces.length || !traces[0].x || traces[0].x.length === 0) {
						extraLayout.annotations = [{
							text: "No recordings in this timeframe",
							xref: "paper", yref: "paper", x: 0.5, y: 0.5,
							showarrow: false, font: { size: 17, color: "#aaaaaa" },
						}];
					}
				}

				if (definition.key === "pace") {
					if (!usePreviewLayout) layout.height = isOverview ? 300 : 480;
					layout.title = { text: "" };
					layout.xaxis.type = "date";
					layout.xaxis.autorange = false;
					layout.xaxis.tickformat = tickConfig.tickformat;
					layout.xaxis.dtick = tickConfig.dtick;
					layout.xaxis.range = xRange;
					layout.xaxis.title = "Session date";
					layout.yaxis.title = "Words per minute (WPM)";
					layout.hovermode = "closest";
					layout.legend = { orientation: "h", yanchor: "top", y: isOverview ? -0.18 : -0.22, xanchor: "left", x: 0, font: { size: 11 } };
					layout.showlegend = !usePreviewLayout && !isOverview;
					layout.margin = usePreviewLayout ? { b: 24, t: 10, l: 34, r: 12 } : isOverview ? { b: 72, t: 20, l: 46, r: 18 } : { b: 130, t: 40, l: 60, r: 30 };
				} else {
					layout.xaxis.type = "date";
					layout.xaxis.tickformat = tickConfig.tickformat;
					layout.xaxis.dtick = tickConfig.dtick;
					layout.xaxis.range = xRange;
					layout.xaxis.title = "Session date";
					layout.yaxis.title = "Pitch variation (semitones from session median)";
					layout.hovermode = usePreviewLayout ? "closest" : "x unified";
					// For overview cards, use white hoverlabel background for readability
					if (isOverview) {
						layout.hoverlabel = { bgcolor: '#ffffff', font: { color: '#122a2c' } };
					}
					layout.xaxis.showspikes = !usePreviewLayout && !isOverview;
					layout.xaxis.spikemode = usePreviewLayout || isOverview ? false : "across";
					layout.xaxis.spikesnap = usePreviewLayout || isOverview ? false : "cursor";
					layout.xaxis.spikedash = usePreviewLayout || isOverview ? false : "dot";
					layout.xaxis.spikecolor = usePreviewLayout || isOverview ? false : "rgba(0,0,0,0.25)";
					layout.xaxis.spikethickness = usePreviewLayout || isOverview ? false : 1;
					layout.legend = { orientation: "v", yanchor: "top", y: -0.22, xanchor: "left", x: 0, font: { size: 11 } };
					layout.showlegend = !usePreviewLayout && !isOverview;
					layout.margin = usePreviewLayout ? { b: 24, t: 10, l: 34, r: 12 } : isOverview ? { b: 72, t: 20, l: 46, r: 18 } : { b: 250, t: 50, l: 60, r: 30 };
				}
			} else if (definition.key === 'fillerTrend') {
				const filtered = detailSet;
				const fillerWords = collectFillerWords(filtered.length ? filtered : recordings);
				let maxY = 5;
				traces = fillerWords.map((word, idx) => {
					const byDay = new Map();
					for (const r of filtered) {
						const date = getRecordingDate(r);
						if (!date) continue;
						const dayKey = date.toISOString().slice(0, 10);
						const totalWords = Math.max(1, Number(r.total_words) || 1);
						const count = r.filler_counts && r.filler_counts[word] ? Number(r.filler_counts[word]) : 0;
						const val = (100.0 * count) / totalWords;
						if (!byDay.has(dayKey)) byDay.set(dayKey, { sum: 0, n: 0 });
						const entry = byDay.get(dayKey);
						entry.sum += val;
						entry.n += 1;
					}
					const days = Array.from(byDay.keys()).sort();
					const x = days.map(d => d + "T12:00:00.000Z");
					const y = days.map(d => byDay.get(d).sum / byDay.get(d).n);
					const hasData = y.some((v) => v > 0);
					maxY = Math.max(maxY, ...y, maxY);
						const color = PALETTE[idx % PALETTE.length];
						return {
							type: "scatter",
							mode: "lines+markers",
							name: word,
							x,
							y,
							visible: hasData,
							showlegend: !usePreviewLayout,
							line: { color, width: 2.5, shape: "spline" },
							marker: { size: 6, color },
							hovertemplate: `<b>%{x|%b %d, %Y}</b><br>${word}: %{y:.2f}%<extra></extra>`,
						};
				});
				traces.unshift({
					type: "scatter",
					mode: "none",
					name: "Disable all",
					x: [],
					y: [],
					showlegend: !usePreviewLayout,
					hoverinfo: "none",
					line: { color: "#999" },
					marker: { color: "#999" },
				});
				layout.showlegend = !usePreviewLayout && !isOverview;
				layout.title = { text: "" };
				layout.height = isOverview ? 280 : undefined;
				layout.xaxis.type = "date";
				layout.xaxis.tickformat = tickConfig.tickformat;
				layout.xaxis.dtick = tickConfig.dtick;
				layout.xaxis.range = xRange;
				layout.xaxis.title = "Recording date";
				layout.yaxis.title = "Percentage of spoken words (%)";
				layout.hovermode = "closest";
				layout.legend = {
					orientation: "h",
					yanchor: "top",
					y: -0.22,
					xanchor: "left",
					x: 0,
					font: { size: 11 },
				};
				layout.margin = usePreviewLayout ? { b: 24, t: 10, l: 34, r: 12 } : isOverview ? { b: 72, t: 20, l: 46, r: 18 } : { b: 130, t: 40, l: 60, r: 30 };
				layout.yaxis.range = [0, Math.max(5, maxY * 1.2)];
				if (!filtered.length) {
					extraLayout.annotations = [{
						text: "No recordings in this timeframe",
						xref: "paper", yref: "paper", x: 0.5, y: 0.5,
						showarrow: false, font: { size: 17, color: "#aaaaaa" },
					}];
				}
			} else if (definition.key === 'fillerRecording') {
				// Build stacked horizontal bars using filler_counts
				const fillerWords = collectFillerWords(detailSet.length ? detailSet : recordings);
				const selected = [...detailSet].sort((a, b) => (Number(b.filler_percentage) || 0) - (Number(a.filler_percentage) || 0));
				
				// For preview, limit to top 7 bars; for detail, show all
				const displayedRecordings = usePreviewLayout ? selected.slice(0, 7) : selected;
				const labels = displayedRecordings.map((r) => String(r.recording_name || "Unknown recording"));
				const dates = displayedRecordings.map((r, i) => formatRecordingLabel(toChartDateValue(r, i)));
				const totals = displayedRecordings.map((r) => Number(r.filler_percentage) || 0);
				
				traces = fillerWords.map((word, idx) => {
					const values = displayedRecordings.map((r)=> {
						const tot = Number(r.total_words) || 1;
						const count = r.filler_counts && r.filler_counts[word] ? Number(r.filler_counts[word]) : 0;
						return (count / Math.max(1, tot)) * 100.0;
					});
					const hasData = values.some((v) => v > 0);
					return {
						type: 'bar', orientation: 'h', name: word, y: labels, x: values, marker: { color: PALETTE[idx % PALETTE.length] },
						showlegend: false,
						customdata: displayedRecordings.map((r, i)=>[word, labels[i], dates[i], totals[i], Number(r.total_words) || 0]),
						hovertemplate: '<b>%{customdata[1]}</b><br>Date: %{customdata[2]}<br>Filler word: %{customdata[0]}<br>Word share: %{x:.2f}%<br>Total filler: %{customdata[3]:.2f}%<br>Total words: %{customdata[4]:.0f}<extra></extra>',
					};
				});
				fillerLegendItems = traces.map((t, idx) => ({
					word: t.name,
					color: t.marker ? t.marker.color : '#999',
					traceIndex: idx,
				}));
				
				if (usePreviewLayout) {
					// Preview: show top 7 bars in scrollable container
					const maxVisibleBars = 7;
					const marginBot = 30;
					const marginTop = 20;
					const marginLeft = 0;
					const marginRight = 25;
					const fixedViewportHeight = 220;
					const availableHeightForBars = fixedViewportHeight - marginBot - marginTop;
					const barHeightPerRecording = availableHeightForBars / maxVisibleBars;
					const totalChartHeight = marginTop + (displayedRecordings.length * barHeightPerRecording) + marginBot;
					
					layout.barmode = 'stack';
					layout.margin = { l: marginLeft, r: marginRight, t: marginTop, b: marginBot };
					layout.template = "plotly_white";
					layout.height = totalChartHeight;
					layout.showlegend = false;
				} else {
					// Detail view: show all bars in scrollable container
					const maxVisibleBars = isOverview ? 6 : 15;
					const marginBot = isOverview ? 28 : 50;
					const marginTop = isOverview ? 18 : 30;
					const marginLeft = 0;
					const marginRight = isOverview ? 20 : 35;
					const fixedViewportHeight = isOverview ? 210 : 400;
					const availableHeightForBars = fixedViewportHeight - marginBot - marginTop;
					const barHeightPerRecording = availableHeightForBars / maxVisibleBars;
					const totalChartHeight = marginTop + (displayedRecordings.length * barHeightPerRecording) + marginBot;
					
					layout.barmode = 'stack';
					layout.margin = { l: marginLeft, r: marginRight, t: marginTop, b: marginBot };
					layout.template = "plotly_white";
					layout.height = totalChartHeight;
					layout.showlegend = !isOverview;
					layout.legend = {
						orientation: "h",
						yanchor: "bottom",
						y: isOverview ? -0.08 : -0.12,
						xanchor: "left",
						x: 0,
						font: { size: 11 },
					};
				}
				
				layout.yaxis = layout.yaxis || {};
				layout.yaxis.categoryorder = 'array';
				layout.yaxis.categoryarray = labels;
				layout.yaxis.autorange = 'reversed';
				layout.xaxis = layout.xaxis || {};
				layout.xaxis.ticksuffix = '%';
				const maxFiller = Math.max(...(detailSet.length ? detailSet : recordings).map((r) => Number(r.filler_percentage) || 0), 10);
				layout.xaxis.range = [0, Math.min(100, maxFiller * 1.2 + 3)];
				layout.xaxis.title = usePreviewLayout
					? { text: "% of spoken words", standoff: 10 }
					: { text: "% of all spoken words", standoff: 12 };
				if (usePreviewLayout) {
					layout.yaxis.title = "Recording";
				}
				layout.title = { text: "" };
				if (!displayedRecordings.length) {
					extraLayout.annotations = [{
						text: "No recordings in this timeframe",
						xref: "paper", yref: "paper", x: 0.5, y: 0.5,
						showarrow: false, font: { size: 17, color: "#aaaaaa" },
					}];
				}
			}
		}

		if (!usePreviewLayout) {
			layout.annotations = extraLayout.annotations || [];
		}

		let plotTarget = container;
		if (definition.key === 'fillerRecording') {
			let scrollPane = container.querySelector('.filler-chart-scroll');
			if (!scrollPane) {
				container.style.cssText += ';display:flex;flex-direction:column;overflow:hidden;';
				scrollPane = document.createElement('div');
				scrollPane.className = 'filler-chart-scroll';
				container.appendChild(scrollPane);
				if (!usePreviewLayout && !isOverview) {
					const legendPane = document.createElement('div');
					legendPane.className = 'filler-chart-legend';
					container.appendChild(legendPane);
				}
			}

			// Constrain overview cards: set scroll pane height to the chart container's
			// computed height so the card remains fixed and extra bars become scrollable.
			if (scrollPane) {
				if (isOverview) {
					// Compute a fixed chart area height based on the card's square size.
					// Use the card's layout (width == height due to aspect-ratio) and reserve
					// space for the header so the chart area becomes a stable box.
					const cardEl = container.closest('.insight-card');
					let chartAreaH = 220;
					if (cardEl) {
						const cardRect = cardEl.getBoundingClientRect();
						const headerEl = cardEl.querySelector('.insight-card-header');
						const headerH = headerEl ? headerEl.getBoundingClientRect().height : 0;
						// aim for square card: use card height minus header as available chart area
						chartAreaH = Math.max(80, Math.floor(cardRect.height - headerH - 8));
					}
					scrollPane.style.height = `${chartAreaH}px`;
					scrollPane.style.overflowY = 'auto';
					scrollPane.style.maxHeight = '100%';
					scrollPane.style.minHeight = '0';
				} else if (usePreviewLayout) {
					// For compact previews, allow Plotly-driven heights but keep overflow available
					scrollPane.style.overflowY = 'auto';
				} else {
					// For detail views, leave scrolling to CSS/legend pane
					scrollPane.style.height = '';
					scrollPane.style.overflowY = '';
				}
			}

			plotTarget = scrollPane;
		}

		Plotly.react(plotTarget, traces, Object.assign({}, layout, extraLayout), {
			displayModeBar: false,
			responsive: false,
		});

		if (!usePreviewLayout && !isOverview && definition.key === 'fillerRecording' && fillerLegendItems) {
			const legendPane = container.querySelector('.filler-chart-legend');
			if (legendPane) {
				legendPane.innerHTML = '';
				
				// Add "Disable all" item
				const disableAllItem = document.createElement('span');
				disableAllItem.className = 'filler-legend-item filler-legend-disable-all';
				disableAllItem.style.cursor = 'pointer';
				disableAllItem.textContent = 'Disable all';
				disableAllItem.addEventListener('click', () => {
					const anyVisible = fillerLegendItems.some((item) => {
						const trace = plotTarget.data && plotTarget.data[item.traceIndex];
						const v = trace ? trace.visible : true;
						return v !== "legendonly" && v !== false;
					});
					const updates = fillerLegendItems.map(() => anyVisible ? "legendonly" : true);
					Plotly.restyle(plotTarget, "visible", updates, fillerLegendItems.map((item) => item.traceIndex));
					updateLegendItemVisibility();
				});
				legendPane.appendChild(disableAllItem);
				
				// Add individual filler word items
				fillerLegendItems.forEach((item) => {
					const span = document.createElement('span');
					span.className = 'filler-legend-item';
					span.style.cursor = 'pointer';
					const swatch = document.createElement('span');
					swatch.className = 'filler-legend-swatch';
					swatch.style.background = item.color;
					const label = document.createElement('span');
					label.textContent = item.word;
					span.appendChild(swatch);
					span.appendChild(label);
					span.addEventListener('click', () => {
						const trace = plotTarget.data && plotTarget.data[item.traceIndex];
						const currentVisible = trace ? trace.visible : true;
						const newVisible = currentVisible === "legendonly" || currentVisible === false ? true : "legendonly";
						Plotly.restyle(plotTarget, "visible", newVisible, [item.traceIndex]);
						updateLegendItemVisibility();
					});
					legendPane.appendChild(span);
				});
				
				function updateLegendItemVisibility() {
					if (!plotTarget.data) return;
					fillerLegendItems.forEach((item) => {
						const span = legendPane.querySelectorAll('.filler-legend-item')[item.traceIndex + 1];
						const trace = plotTarget.data[item.traceIndex];
						const isVisible = trace && trace.visible && trace.visible !== "legendonly";
						if (span) {
							span.style.opacity = isVisible ? '1' : '0.4';
						}
					});
				}
				updateLegendItemVisibility();
			}
		}

		if (!usePreviewLayout && !isOverview && definition.key === "fillerTrend" && !container._fillerLegendBound) {
			container._fillerLegendBound = true;
			container.on("plotly_legendclick", function(data) {
				if (data.data[data.curveNumber].name !== "Disable all") return;
				const realIndices = data.data
					.map((t, i) => t.name !== "Disable all" ? i : null)
					.filter(i => i !== null);
				const anyVisible = realIndices.some(i => {
					const v = data.data[i].visible;
					return v !== "legendonly" && v !== false;
				});
				Plotly.restyle(container, "visible", realIndices.map(() => anyVisible ? "legendonly" : true), realIndices);
				return false;
			});
		}

		if (usePreviewLayout) {
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
		let timeframeSelect = null;
		if (definition.key === "pace" || definition.key === "pitch" || definition.key === "fillerTrend" || definition.key === "fillerRecording") {
			timeframeSelect = createTimeframeSelect("all_time");
			controls.appendChild(timeframeSelect);
		}

		header.append(titleWrap, controls);

		const chart = document.createElement("div");
		chart.className = "insight-card-chart";

		card.append(header, chart);

		const rerender = () => {
			// Overview cards always render the full dataset.
			renderPlotInto(chart, definition, recordings, initialSelection, true, timeframeSelect ? timeframeSelect.value : undefined, true);
		};

		const scheduleInitialRerender = () => {
			// Use a ResizeObserver to keep the card square by matching height to width.
			// This reacts to the phone-stage scaling and window resizes.
			try {
				const applySquare = () => {
					const w = Math.floor(card.offsetWidth || card.clientWidth || 0);
					if (w > 0) {
						card.style.height = w + 'px';
						chart.style.maxHeight = '100%';
						chart.style.minHeight = '0';
					}
				};
				const ro = new ResizeObserver(() => {
					applySquare();
					// after sizing, render the chart inside the constrained area
					window.requestAnimationFrame(rerender);
				});
				ro.observe(card);
				// apply once immediately
				applySquare();
			} catch (e) {
				// fallback: schedule a single render
				window.requestAnimationFrame(rerender);
			}
		};

		if (timeframeSelect) {
			timeframeSelect.addEventListener("click", (event) => event.stopPropagation());
			timeframeSelect.addEventListener("change", (event) => {
				event.stopPropagation();
				rerender();
			});
		}

		card.addEventListener("click", () => {
			let url = `plot.html?plot=${encodeURIComponent(definition.key)}&count=${encodeURIComponent(initialSelection)}`;
			if (timeframeSelect) url += `&timeframe=${encodeURIComponent(timeframeSelect.value)}`;
			window.location.href = url;
		});
		card.addEventListener("keydown", (event) => {
			if (event.key !== "Enter" && event.key !== " ") {
				return;
			}

			event.preventDefault();
			let url = `plot.html?plot=${encodeURIComponent(definition.key)}&count=${encodeURIComponent(initialSelection)}`;
			if (timeframeSelect) url += `&timeframe=${encodeURIComponent(timeframeSelect.value)}`;
			window.location.href = url;
		});

		scheduleInitialRerender();
		return card;
	}

	function setNavState(viewName) {
		const recordIcon = document.querySelector(".bottom-nav .record-icon");
		const searchIcon = document.querySelector(".bottom-nav .icon.search");
		const folderIcon = document.querySelector(".bottom-nav .icon.folder");
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

		if (folderIcon) {
			folderIcon.style.cursor = "pointer";
			folderIcon.addEventListener("click", (event) => {
				event.stopPropagation();
				window.location.href = "insights.html";
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
			app.appendChild(createCard(definition, recordings, "all"));
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
		intro.textContent = "Use the dropdown to adjust the visible time frame. \n Hover over data points to see specifics.";

		const controls = document.createElement("div");
		controls.className = "plot-toolbar-controls";
		const select = createCountSelect(selectedCount);
		let timeframeSelect = null;
		if (definition.key === "pace" || definition.key === "pitch" || definition.key === "fillerTrend" || definition.key === "fillerRecording") {
			timeframeSelect = createTimeframeSelect("all_time");
			controls.appendChild(timeframeSelect);
		} else {
			// Fallback (currently unused): keep count selector for non-timeframe plots.
			controls.appendChild(select);
		}

		toolbar.append(titleWrap, controls);

		const chartWrap = document.createElement("div");
		chartWrap.className = "plot-chart-wrap";

		panel.append(toolbar, chartWrap);

		const detailStack = document.createElement("div");
		detailStack.className = "plot-detail-stack";
		detailStack.append(intro, panel);

		app.appendChild(detailStack);

		const rerender = () => {
			const effectiveCount = timeframeSelect ? "all" : select.value;
			renderPlotInto(chartWrap, definition, recordings, effectiveCount, false, timeframeSelect ? timeframeSelect.value : "all_time");
		};

		if (!timeframeSelect) {
			select.addEventListener("change", rerender);
			select.addEventListener("click", (event) => event.stopPropagation());
		}
		if (timeframeSelect) {
			timeframeSelect.addEventListener("change", rerender);
			timeframeSelect.addEventListener("click", (event) => event.stopPropagation());
		}
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