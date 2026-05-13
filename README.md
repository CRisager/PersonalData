# Personal Data Project: Speakly

A speech analysis app styled as a mobile phone UI. Record a speech, get instant feedback on filler words, speaking pace, and pitch variation, then review trends over time in a performance insights dashboard.

## How it works

The browser renders a phone-shell UI. Recording and analysis run through a Flask backend:

1. **Record** — capture audio in the browser
2. **Confirm** — preview the recording, name it, and choose a category
3. **Learn** — step through three feedback screens:
   - **Filler words** — jar visualization showing your ratio vs. previous average
   - **Pace** — semicircle gauge showing words per minute vs. recommended range
   - **Pitch variation** — chart comparing your pitch range to previous recordings
4. **Insights** — a separate dashboard with trend charts across all recordings

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
python main.py --serve --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--host 0.0.0.0` | `127.0.0.1` | Expose to local network |
| `--model small` | `base.en` | Whisper model size |
| `--plot-pitch` | off | Save a pitch PNG alongside each recording |

Health check: `http://127.0.0.1:8000/api/health`

## App screens

| Screen | URL | Status |
|--------|-----|--------|
| New recording | `/` | Live |
| Performance Insights | `/frontend/insights.html` | Live |
| Profile | `/frontend/profile.html` | Live (static) |
| Home | `/frontend/home.html` | Coming soon |
| Folder | `/frontend/folder.html` | Coming soon |

## Output files

Each saved recording produces two files, both using the same stem:

- `Sound_recordings/<stem>.wav` — the audio file
- `sound_analysis/<stem>.json` — consolidated analysis (transcript, filler metrics, pace, pitch summary + series, metadata)

The stem is built from the recording name you enter, sanitized for cross-platform filenames, and suffixed with a timestamp (`YYYYMMDD_HHMMSS`). If a conflict exists, a numeric suffix is appended automatically.

Example for a recording named `Class Presentation 1`:

```
Sound_recordings/Class_Presentation_1_20260513_143022.wav
sound_analysis/Class_Presentation_1_20260513_143022.json
```

The JSON file is the single source of truth for each recording and is what the Insights dashboard reads.

## Performance Insights dashboard

The dashboard at `/frontend/insights.html` reads all recordings via `/api/insights-data` and renders interactive Plotly charts:

- Words per minute over time
- Filler word percentage over time
- Pitch variation over time
- Filler word count breakdown per recording

Results can be filtered by count (last 5 / 10 / 15 / 20 / all) or by timeframe (this week / month / 6 months / year / all time). Clicking a chart opens a detail view.

To regenerate the static `frontend/insights-data.json` file (used as a fallback when the API is not running):

```bash
python scripts/regenerate_insights_data.py
```

## UI-only design mode

To tweak the frontend without a running backend or microphone access:

```
http://127.0.0.1:8000/?uiPreview=1
```

Recording and processing are mocked so you can click through the full Record → Confirm → Learn flow. To jump directly to a specific screen, add `previewScreen`:

| Screen | URL |
|--------|-----|
| Idle | `?uiPreview=1&previewScreen=idle` |
| Record | `?uiPreview=1&previewScreen=record` |
| Confirm | `?uiPreview=1&previewScreen=confirm` |
| Learn | `?uiPreview=1&previewScreen=learn` |
