"""Build an interactive HTML dashboard from synthetic speech-analysis data.

The dashboard includes:
- filler-word position density, per recording
- filler-word counts over time
- rolling words-per-minute over time
- clickable filler-word cloud with drill-down info
- pitch variation views
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PALETTE = ["#f97316", "#06b6d4", "#10b981", "#8b5cf6", "#ef4444", "#0f766e"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML dashboard from fake speech data.")
    parser.add_argument(
        "--input",
        default="recordings/fake_speech_dataset.json",
        help="Path to the synthetic dataset JSON.",
    )
    parser.add_argument(
        "--output",
        default="recordings/fake_speech_plots.html",
        help="Path to the output HTML dashboard.",
    )
    return parser.parse_args()


def load_data(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def js_data(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    payload = payload.replace("<", "\\u003c")
    return payload


def create_html(data: dict[str, Any]) -> str:
    dataset_json = js_data(data)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fake Speech Analysis Dashboard</title>
  <style>
    :root {{
      --bg: #f6f1e7;
      --panel: rgba(255, 255, 255, 0.78);
      --panel-strong: rgba(255, 255, 255, 0.94);
      --text: #16202a;
      --muted: #5a6a76;
      --line: rgba(22, 32, 42, 0.12);
      --shadow: 0 18px 60px rgba(15, 23, 42, 0.10);
      --accent-1: #f97316;
      --accent-2: #06b6d4;
      --accent-3: #10b981;
      --accent-4: #8b5cf6;
      --accent-5: #ef4444;
      --accent-6: #0f766e;
      font-family: Georgia, "Times New Roman", serif;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(249,115,22,0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(6,182,212,0.14), transparent 28%),
        linear-gradient(180deg, #f9f3e8 0%, #f6f1e7 100%);
    }}

    header {{
      padding: 40px 28px 18px;
      max-width: 1400px;
      margin: 0 auto;
    }}

    h1 {{
      font-size: clamp(2.1rem, 5vw, 4.2rem);
      line-height: 0.96;
      margin: 0 0 12px;
      letter-spacing: -0.04em;
    }}

    .subtitle {{
      margin: 0;
      max-width: 920px;
      color: var(--muted);
      font-size: 1.02rem;
      line-height: 1.5;
    }}

    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 18px 18px 44px;
    }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 16px 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      margin-bottom: 18px;
    }}

    .toolbar label {{
      font-size: 0.95rem;
      color: var(--muted);
      display: block;
      margin-bottom: 6px;
    }}

    .toolbar select {{
      min-width: 240px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: white;
      font: inherit;
      color: var(--text);
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}

    .metric {{
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }}

    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}

    .metric .value {{
      font-size: 1.55rem;
      font-weight: 700;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}

    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-7 {{ grid-column: span 7; }}
    .span-6 {{ grid-column: span 6; }}
    .span-5 {{ grid-column: span 5; }}

    .panel h2 {{
      margin: 0 0 8px;
      font-size: 1.3rem;
      letter-spacing: -0.02em;
    }}

    .panel p.desc {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.45;
    }}

    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 14px;
    }}

    .mini-card {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px;
    }}

    .mini-card header {{
      padding: 0;
      margin-bottom: 10px;
    }}

    .mini-card .title {{
      font-size: 0.98rem;
      font-weight: 700;
      margin: 0;
    }}

    .mini-card .sub {{
      color: var(--muted);
      font-size: 0.88rem;
      margin-top: 4px;
    }}

    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}

    .cloud-wrap {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 16px;
      align-items: start;
    }}

    .cloud {{
      min-height: 360px;
      background: rgba(255,255,255,0.66);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-content: flex-start;
    }}

    .cloud button {{
      appearance: none;
      border: 0;
      background: transparent;
      color: var(--text);
      font: inherit;
      line-height: 1;
      cursor: pointer;
      padding: 4px 0;
      border-radius: 999px;
      transition: transform 140ms ease, opacity 140ms ease;
    }}

    .cloud button:hover {{
      transform: translateY(-2px) scale(1.03);
      opacity: 0.88;
    }}

    .detail {{
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      position: sticky;
      top: 16px;
    }}

    .detail h3 {{ margin: 0 0 10px; }}
    .detail .word {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 10px; }}
    .detail .stat {{ margin: 8px 0; color: var(--muted); }}

    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}

    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 99px;
      display: inline-block;
    }}

    @media (max-width: 980px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .span-8, .span-7, .span-6, .span-5 {{ grid-column: span 12; }}
      .cloud-wrap {{ grid-template-columns: 1fr; }}
    }}

    @media (max-width: 700px) {{
      header {{ padding: 26px 18px 12px; }}
      .shell {{ padding: 12px 12px 36px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .toolbar {{ align-items: stretch; }}
      .toolbar select {{ width: 100%; min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Fake Speech Analysis Dashboard</h1>
    <p class="subtitle">Interactive views built from the synthetic dataset in recordings/fake_speech_dataset.json. The dashboard is designed to answer where filler words appear, how the pace shifts, how pitch behaves, and which words dominate the filler vocabulary.</p>
  </header>

  <main class="shell">
    <div class="toolbar">
      <div>
        <label for="recordingSelect">Selected recording</label>
        <select id="recordingSelect"></select>
      </div>
      <div>
        <label>Dataset</label>
        <div id="datasetSummary"></div>
      </div>
    </div>

    <section class="metrics" id="metrics"></section>

    <section class="grid">
      <article class="panel span-12">
        <h2>1) Filler-word position density</h2>
        <p class="desc">Each card shows where filler words land within a recording. Peaks near the left edge indicate nervous openers or early hesitation; broader coverage suggests fillers are scattered through the whole recording.</p>
        <div class="mini-grid" id="densityGrid"></div>
      </article>

      <article class="panel span-12">
        <h2>2) Filler words over time</h2>
        <p class="desc">Line chart of filler counts over time for the selected recording. It shows the main filler types and a total line so you can see whether hesitations cluster in bursts or fade as the recording continues.</p>
        <div id="fillerTimeChart"></div>
        <div class="legend" id="fillerTimeLegend"></div>
      </article>

      <article class="panel span-12">
        <h2>3) Pace over time</h2>
        <p class="desc">Rolling words-per-minute over a 30-second window. This is useful for spotting acceleration at the end, slow sections mid-thought, or a steady delivery.</p>
        <div id="paceChart"></div>
      </article>

      <article class="panel span-12">
        <h2>4) Filler-word cloud</h2>
        <p class="desc">Click a word to see how many times it appears in the dataset and in how many recordings it appears. This replaces a static word cloud with a clickable drill-down.</p>
        <div class="cloud-wrap">
          <div class="cloud" id="wordCloud"></div>
          <aside class="detail" id="wordDetail">
            <h3>Word details</h3>
            <div class="word">Select a word</div>
            <div class="stat">Click any word in the cloud to inspect its totals.</div>
          </aside>
        </div>
      </article>

      <article class="panel span-12">
        <h2>5) Pitch variation</h2>
        <p class="desc">The best pitch views for this dataset are a smoothed contour for the selected recording and an overview of median pitch plus range across all recordings. Together, they reveal within-recording movement and between-recording differences.</p>
        <div id="pitchContourChart"></div>
        <div style="height: 14px;"></div>
        <div id="pitchOverviewChart"></div>
      </article>
    </section>
  </main>

  <script>
    const DATA = __DATASET_JSON__;
    const COLORS = __PALETTE_JSON__;
    const recordings = DATA.recordings;
    const fillerSummary = DATA.filler_word_summary || [];

    const state = {{ selectedIndex: 0 }};

    function esc(text) {{
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    function formatNumber(value, digits = 0) {{
      return Number(value).toLocaleString(undefined, {{ maximumFractionDigits: digits, minimumFractionDigits: digits }});
    }}

    function sum(values) {{
      return values.reduce((total, value) => total + value, 0);
    }}

    function mean(values) {{
      return values.length ? sum(values) / values.length : 0;
    }}

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}

    function interpolateSeries(series) {{
      const filled = [];
      let last = null;
      for (const point of series) {{
        if (point == null || Number.isNaN(point)) {{
          filled.push(null);
        }} else {{
          filled.push(point);
          last = point;
        }}
      }}
      let next = null;
      for (let i = filled.length - 1; i >= 0; i -= 1) {{
        if (filled[i] == null && next != null && last != null) {{
          filled[i] = next;
        }} else if (filled[i] != null) {{
          next = filled[i];
        }}
      }}
      return filled.map(value => value == null ? 0 : value);
    }}

    function smoothSeries(values, window = 5) {{
      if (window <= 1) return values.slice();
      const radius = Math.floor(window / 2);
      return values.map((_, index) => {{
        let total = 0;
        let count = 0;
        for (let offset = -radius; offset <= radius; offset += 1) {{
          const candidate = values[index + offset];
          if (candidate != null) {{
            total += candidate;
            count += 1;
          }}
        }}
        return count ? total / count : values[index];
      }});
    }}

    function createSVG(width, height, viewBoxContent) {{
      return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">${{viewBoxContent}}</svg>`;
    }}

    function chartFrame(width, height, title, subtitle) {{
      return `
        <div style="padding: 6px 2px 0;">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px;">
            <strong style="font-size:1rem;">${{esc(title)}}</strong>
            <span style="color:var(--muted);font-size:0.88rem;">${{esc(subtitle || '')}}</span>
          </div>
        </div>
      `;
    }}

    function niceTicks(minValue, maxValue, count) {{
      if (count <= 1 || minValue === maxValue) return [minValue, maxValue];
      const step = (maxValue - minValue) / (count - 1);
      return Array.from({{ length: count }}, (_, index) => minValue + index * step);
    }}

    function lineChartSVG(seriesList, options) {{
      const width = options.width || 980;
      const height = options.height || 300;
      const margin = {{ top: 18, right: 18, bottom: 34, left: 48 }};
      const xMin = options.xMin;
      const xMax = options.xMax;
      const yMin = options.yMin != null ? options.yMin : 0;
      const yMax = options.yMax != null ? options.yMax : 1;
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;

      const xScale = value => margin.left + ((value - xMin) / (xMax - xMin || 1)) * plotWidth;
      const yScale = value => margin.top + plotHeight - ((value - yMin) / (yMax - yMin || 1)) * plotHeight;

      const grid = [];
      const yTicks = niceTicks(yMin, yMax, 5);
      const xTicks = niceTicks(xMin, xMax, 6);

      for (const tick of yTicks) {{
        const y = yScale(tick);
        grid.push(`<line x1="${{margin.left}}" y1="${{y}}" x2="${{width - margin.right}}" y2="${{y}}" stroke="rgba(22,32,42,0.09)" />`);
        grid.push(`<text x="${{margin.left - 8}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#5a6a76">${{options.yFormat ? options.yFormat(tick) : formatNumber(tick, 0)}}</text>`);
      }}

      for (const tick of xTicks) {{
        const x = xScale(tick);
        grid.push(`<line x1="${{x}}" y1="${{margin.top}}" x2="${{x}}" y2="${{height - margin.bottom}}" stroke="rgba(22,32,42,0.06)" />`);
        grid.push(`<text x="${{x}}" y="${{height - 10}}" text-anchor="middle" font-size="11" fill="#5a6a76">${{options.xFormat ? options.xFormat(tick) : formatNumber(tick, 0)}}</text>`);
      }}

      const paths = seriesList.map((series, index) => {{
        const points = series.data.map(point => [xScale(point.x), yScale(point.y)]);
        const path = points.map((point, pointIndex) => `${{pointIndex === 0 ? 'M' : 'L'}}${{point[0].toFixed(2)}},${{point[1].toFixed(2)}}`).join(' ');
        const color = series.color || COLORS[index % COLORS.length];
        const areaPath = options.area
          ? `${{path}} L ${{xScale(series.data[series.data.length - 1].x).toFixed(2)}},${{yScale(yMin).toFixed(2)}} L ${{xScale(series.data[0].x).toFixed(2)}},${{yScale(yMin).toFixed(2)}} Z`
          : '';
        return `
          ${{options.area ? `<path d="${{areaPath}}" fill="${{color}}" fill-opacity="0.12" stroke="none"></path>` : ''}}
          <path d="${{path}}" fill="none" stroke="${{color}}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"></path>
        `;
      }}).join('');

      return createSVG(width, height, `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="18" fill="rgba(255,255,255,0.65)" stroke="rgba(22,32,42,0.10)" />
        ${{grid.join('')}}
        ${{paths}}
        <text x="${{margin.left}}" y="${{15}}" font-size="12" fill="#5a6a76">${{esc(options.caption || '')}}</text>
      `);
    }}

    function densityCard(recording, index) {{
      const positions = (recording.filler_events || []).map(event => event.position_ratio).filter(value => typeof value === 'number');
      const bins = 20;
      const counts = Array.from({{ length: bins }}, () => 0);
      for (const position of positions) {{
        const binIndex = clamp(Math.floor(position * bins), 0, bins - 1);
        counts[binIndex] += 1;
      }}
      const maxCount = Math.max(...counts, 1);
      const width = 260;
      const height = 150;
      const margin = {{ top: 14, right: 12, bottom: 26, left: 30 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xScale = value => margin.left + value * plotWidth;
      const yScale = value => margin.top + plotHeight - (value / maxCount) * plotHeight;
      const points = counts.map((count, binIndex) => [xScale((binIndex + 0.5) / bins), yScale(count)]);
      const line = points.map((point, pointIndex) => `${{pointIndex === 0 ? 'M' : 'L'}}${{point[0].toFixed(2)}},${{point[1].toFixed(2)}}`).join(' ');
      const area = `${{line}} L ${{xScale(1).toFixed(2)}},${{yScale(0).toFixed(2)}} L ${{xScale(0).toFixed(2)}},${{yScale(0).toFixed(2)}} Z`;
      const grid = [0, 0.25, 0.5, 0.75, 1].map(tick => `
        <line x1="${{xScale(tick)}}" y1="${{margin.top}}" x2="${{xScale(tick)}}" y2="${{height - margin.bottom}}" stroke="rgba(22,32,42,0.07)" />
        <text x="${{xScale(tick)}}" y="${{height - 9}}" text-anchor="middle" font-size="10" fill="#6b7280">${{tick.toFixed(2)}}</text>
      `).join('');

      return `
        <div class="mini-card">
          <header>
            <p class="title">${{esc(recording.recording_id)}} <span style="color:var(--muted);font-weight:400;">${{esc(recording.pace_profile)}} / ${{esc(recording.pitch_profile)}}</span></p>
            <div class="sub">${{positions.length}} filler events · ${{recording.duration_seconds.toFixed(1)}} s</div>
          </header>
          ${createSVG(width, height, `
            <rect x="0" y="0" width="${width}" height="${height}" rx="16" fill="rgba(255,255,255,0.75)" stroke="rgba(22,32,42,0.08)" />
            ${grid}
            <path d="${area}" fill="${COLORS[index % COLORS.length]}" fill-opacity="0.16"></path>
            <path d="${line}" fill="none" stroke="${COLORS[index % COLORS.length]}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"></path>
            <text x="12" y="16" font-size="11" fill="#5a6a76">start</text>
            <text x="${width - 36}" y="16" font-size="11" fill="#5a6a76">end</text>
          `)}
        </div>
      `;
    }}

    function updateMetricTiles() {{
      const totalRecordings = recordings.length;
      const totalWords = DATA.total_words || 0;
      const totalFillers = DATA.total_filler_words || 0;
      const fillerPct = DATA.overall_filler_percentage || 0;
      document.getElementById('metrics').innerHTML = `
        <div class="metric"><span class="label">Recordings</span><span class="value">${{formatNumber(totalRecordings)}}</span></div>
        <div class="metric"><span class="label">Total words</span><span class="value">${{formatNumber(totalWords)}}</span></div>
        <div class="metric"><span class="label">Filler words</span><span class="value">${{formatNumber(totalFillers)}}</span></div>
        <div class="metric"><span class="label">Filler rate</span><span class="value">${{formatNumber(fillerPct, 1)}}%</span></div>
      `;
      document.getElementById('datasetSummary').textContent = `${{formatNumber(totalRecordings)}} recordings · ${{formatNumber(totalFillers)}} fillers`;
    }}

    function renderRecordingSelect() {{
      const select = document.getElementById('recordingSelect');
      select.innerHTML = recordings.map((recording, index) => `<option value="${{index}}">${{esc(recording.recording_id)}} · ${{formatNumber(recording.actual_wpm, 1)}} WPM · ${{formatNumber(recording.filler_percentage, 1)}}% fillers</option>`).join('');
      select.addEventListener('change', event => {{
        state.selectedIndex = Number(event.target.value);
        renderSelectedViews();
      }});
    }}

    function renderDensityGrid() {{
      const grid = document.getElementById('densityGrid');
      grid.innerHTML = recordings.map((recording, index) => densityCard(recording, index)).join('');
    }}

    function renderFillerTimeChart() {{
      const recording = recordings[state.selectedIndex];
      const bins = recording.filler_time_bins || [];
      const xValues = bins.map(row => (row.time_start + row.time_end) / 2);
      const seriesNames = ['um', 'uh', 'you know', 'like', 'total'];
      const seriesList = seriesNames.map((name, index) => {{
        return {{
          name,
          color: COLORS[index % COLORS.length],
          data: bins.map((row, rowIndex) => ({{
            x: xValues[rowIndex],
            y: row[name] ?? row.total ?? 0,
          }})),
        }};
      }});
      const maxY = Math.max(...seriesList.flatMap(series => series.data.map(point => point.y)), 1);
      document.getElementById('fillerTimeChart').innerHTML = chartFrame(980, 300, recording.recording_id, `10-second bins · ${{recording.filler_events.length}} fillers`)
        + lineChartSVG(seriesList, {{
          width: 980,
          height: 300,
          xMin: bins.length ? bins[0].time_start : 0,
          xMax: bins.length ? bins[bins.length - 1].time_end : 1,
          yMin: 0,
          yMax: maxY,
          area: false,
          xFormat: value => `${{Math.round(value)}}s`,
          yFormat: value => formatNumber(value, 0),
          caption: 'Filler count lines'
        }});
      document.getElementById('fillerTimeLegend').innerHTML = seriesList.map((series, index) => `<span><i class="swatch" style="background:${{series.color}}"></i>${{esc(series.name)}}</span>`).join('');
    }}

    function renderPaceChart() {{
      const recording = recordings[state.selectedIndex];
      const series = recording.rolling_wpm_30s || [];
      const cleaned = series.map(point => ({{ x: point.time, y: point.wpm }}));
      const maxY = Math.max(...cleaned.map(point => point.y), 1);
      document.getElementById('paceChart').innerHTML = chartFrame(980, 300, recording.recording_id, 'Rolling 30-second average')
        + lineChartSVG([{{ name: 'WPM', color: '#10b981', data: cleaned }}], {{
          width: 980,
          height: 300,
          xMin: cleaned.length ? cleaned[0].x : 0,
          xMax: cleaned.length ? cleaned[cleaned.length - 1].x : 1,
          yMin: 0,
          yMax: maxY,
          area: true,
          xFormat: value => `${{Math.round(value)}}s`,
          yFormat: value => formatNumber(value, 0),
          caption: 'Words per minute over time'
        }});
    }}

    function renderWordCloud() {{
      const cloud = document.getElementById('wordCloud');
      const detail = document.getElementById('wordDetail');
      const summary = fillerSummary.slice().sort((a, b) => b.count - a.count);
      const maxCount = Math.max(...summary.map(item => item.count), 1);

      cloud.innerHTML = summary.map((item, index) => {{
        const size = 18 + (item.count / maxCount) * 26;
        const color = COLORS[index % COLORS.length];
        const rotate = (index % 5 === 0) ? -5 : (index % 7 === 0) ? 6 : 0;
        return `<button data-word="${{esc(item.word)}}" data-count="${{item.count}}" data-recordings="${{item.recording_count}}" data-size="${{size.toFixed(1)}}" style="font-size:${{size.toFixed(1)}}px; color:${{color}}; transform: rotate(${{rotate}}deg);">${{esc(item.word)}}</button>`;
      }}).join(' ');

      function showDetails(item) {{
        detail.innerHTML = `
          <h3>Word details</h3>
          <div class="word" style="color: var(--accent-4);">${{esc(item.word)}}</div>
          <div class="stat"><strong>${{formatNumber(item.count)}}</strong> total uses</div>
          <div class="stat"><strong>${{formatNumber(item.recording_count)}}</strong> recordings contain this word</div>
          <div class="stat">This makes it a good candidate for a drill-down or tooltip in a more interactive dashboard.</div>
        `;
      }}

      cloud.querySelectorAll('button').forEach(button => {{
        button.addEventListener('click', () => {{
          const word = button.dataset.word;
          const item = summary.find(entry => entry.word === word);
          if (item) showDetails(item);
        }});
      }});

      if (summary.length) showDetails(summary[0]);
    }}

    function contourPathFromTrack(track) {{
      const voiced = track.filter(point => point.pitch_hz != null);
      const points = voiced.map(point => point.pitch_hz);
      const times = voiced.map(point => point.time);
      const interpolated = [];
      for (let i = 0; i < track.length; i += 1) {{
        if (track[i].pitch_hz == null) {{
          let left = i - 1;
          while (left >= 0 && track[left].pitch_hz == null) left -= 1;
          let right = i + 1;
          while (right < track.length && track[right].pitch_hz == null) right += 1;
          if (left >= 0 && right < track.length) {{
            const t = (track[i].time - track[left].time) / (track[right].time - track[left].time || 1);
            interpolated.push(track[left].pitch_hz + t * (track[right].pitch_hz - track[left].pitch_hz));
          }} else if (left >= 0) {{
            interpolated.push(track[left].pitch_hz);
          }} else if (right < track.length) {{
            interpolated.push(track[right].pitch_hz);
          }} else {{
            interpolated.push(0);
          }}
        }} else {{
          interpolated.push(track[i].pitch_hz);
        }}
      }}
      return smoothSeries(interpolateSeries(interpolated), 7).map((pitch, index) => ({{ x: track[index].time, y: pitch }}));
    }}

    function renderPitchContour() {{
      const recording = recordings[state.selectedIndex];
      const track = recording.pitch_track || [];
      const contour = contourPathFromTrack(track);
      const maxY = Math.max(...contour.map(point => point.y), 1);
      const minY = Math.min(...contour.map(point => point.y), 0);
      document.getElementById('pitchContourChart').innerHTML = chartFrame(980, 300, recording.recording_id, `Smoothed contour · ${formatNumber(recording.pitch_summary?.voiced_ratio * 100 || 0, 1)}% voiced`)
        + lineChartSVG([{{ name: 'Pitch', color: '#06b6d4', data: contour }}], {{
          width: 980,
          height: 300,
          xMin: contour.length ? contour[0].x : 0,
          xMax: contour.length ? contour[contour.length - 1].x : 1,
          yMin: Math.max(60, minY - 10),
          yMax: maxY + 10,
          area: false,
          xFormat: value => `${{Math.round(value)}}s`,
          yFormat: value => `${{Math.round(value)}} Hz`,
          caption: 'Pitch contour over time'
        }});
    }}

    function renderPitchOverview() {{
      const points = recordings.map((recording, index) => ({{
        x: index + 1,
        median: recording.pitch_summary?.median_hz || 0,
        min: recording.pitch_summary?.min_hz || 0,
        max: recording.pitch_summary?.max_hz || 0,
        range: recording.pitch_summary?.range_hz || 0,
      }}));
      const width = 980;
      const height = 270;
      const margin = {{ top: 20, right: 24, bottom: 42, left: 56 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xScale = value => margin.left + ((value - 1) / ((points.length - 1) || 1)) * plotWidth;
      const allValues = points.flatMap(point => [point.min, point.median, point.max]).filter(Boolean);
      const yMin = Math.max(55, Math.min(...allValues) - 10);
      const yMax = Math.max(...allValues) + 10;
      const yScale = value => margin.top + plotHeight - ((value - yMin) / (yMax - yMin || 1)) * plotHeight;
      const xTicks = points.map(point => point.x);
      const yTicks = niceTicks(yMin, yMax, 5);
      const bars = points.map((point, index) => {
        const x = xScale(point.x);
        return `
          <line x1="${x}" y1="${yScale(point.min)}" x2="${x}" y2="${yScale(point.max)}" stroke="rgba(6,182,212,0.7)" stroke-width="4" stroke-linecap="round"></line>
          <circle cx="${x}" cy="${yScale(point.median)}" r="5" fill="${COLORS[index % COLORS.length]}" stroke="white" stroke-width="2"></circle>
          <text x="${x}" y="${height - 10}" text-anchor="middle" font-size="10" fill="#5a6a76">${esc(recordings[index].recording_id)}</text>
        `;
      }).join('');
      const gridY = yTicks.map(tick => {
        const y = yScale(tick);
        return `
          <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="rgba(22,32,42,0.09)"></line>
          <text x="${margin.left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#5a6a76">${Math.round(tick)} Hz</text>
        `;
      }).join('');
      document.getElementById('pitchOverviewChart').innerHTML = `
        <div style="padding: 6px 2px 0;">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px;">
            <strong style="font-size:1rem;">Pitch summary across recordings</strong>
            <span style="color:var(--muted);font-size:0.88rem;">Median with min-max range</span>
          </div>
        </div>
      ` + createSVG(width, height, `
        <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="rgba(255,255,255,0.65)" stroke="rgba(22,32,42,0.10)" />
        ${gridY}
        ${bars}
      `);
    }}

    function renderSelectedViews() {{
      document.getElementById('recordingSelect').value = String(state.selectedIndex);
      renderFillerTimeChart();
      renderPaceChart();
      renderPitchContour();
    }}

    updateMetricTiles();
    renderRecordingSelect();
    renderDensityGrid();
    renderWordCloud();
    renderSelectedViews();
    renderPitchOverview();
  </script>
</body>
</html>
"""
    return (
        template.replace("__DATASET_JSON__", dataset_json)
        .replace("__PALETTE_JSON__", json.dumps(PALETTE))
        .replace("{{", "{")
        .replace("}}", "}")
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {input_path}")

    data = load_data(input_path)
    html_output = create_html(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_output, encoding="utf-8")

    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    main()