# PersonalData

## Run locally

The browser UI uses a Flask API for uploading and processing recordings, so `python -m http.server 8000` will only serve the page and will not handle the save/process action.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the local app:

```bash
python main.py --serve --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

## UI-only design mode

When you want to tweak frontend UI without microphone access or backend processing, open with:

```bash
http://127.0.0.1:8000/?uiPreview=1
```

In this mode, recording and processing are mocked so you can click through Record -> Confirm -> Learn while styling.

To jump directly to one screen while designing, add `previewScreen`:

- Learn screen: `http://127.0.0.1:8000/?uiPreview=1&previewScreen=learn`
- Confirm screen: `http://127.0.0.1:8000/?uiPreview=1&previewScreen=confirm`
- Record screen: `http://127.0.0.1:8000/?uiPreview=1&previewScreen=record`
- Idle screen: `http://127.0.0.1:8000/?uiPreview=1&previewScreen=idle`

Optional:

- Change host: `python main.py --serve --host 0.0.0.0 --port 8000`
- Health check: `http://127.0.0.1:8000/api/health`

## Output naming

When you save a recording from the UI, the recording name is used as the filename stem for generated files.

Example:

- Input name: `Class Presentation 1`
- Sanitized stem: `Class_Presentation_1`

Generated outputs use that stem:

- `Sound_recordings/Class_Presentation_1.wav`
- `Transcripts/Class_Presentation_1.txt`
- `Filler_analysis/Class_Presentation_1.json` (or `.csv`)
- `Speed/Class_Presentation_1.json`
- `Sound_recordings/Class_Presentation_1_pitch.png` (if pitch plotting is enabled)

If a file with the same stem already exists, a numeric suffix is added automatically to avoid overwrite (for example `Class_Presentation_1_2`).

## Synthetic data for plots

To generate fake speech-analysis data that matches the plots in this project, run:

```bash
python generate_fake_speech_data.py --output recordings/fake_speech_dataset.json --recordings 8 --seed 42
```

To build the interactive dashboard from that dataset, run:

```bash
python build_fake_speech_plots.py --input recordings/fake_speech_dataset.json --output recordings/fake_speech_plots.html
```

The generated JSON includes per-recording word timings, filler positions, 10-second filler bins, rolling 30-second WPM, and pitch tracks so you can build the following plots directly. The dashboard renders them as an interactive HTML page with no extra plotting packages required:

- Filler-word position density, using `filler_events[].position_ratio`
- Filler-word counts over time, using `filler_time_bins` or `filler_time_bins_cumulative`
- Rolling pace over time, using `rolling_wpm_30s`
- Filler-word word cloud, using `filler_word_summary`
- Pitch variation, using `pitch_track` and `pitch_summary`

For pitch, the most useful views are a smoothed pitch contour over time and an overview of median pitch plus min-max range across recordings. If you want a PNG-only export later, I can add a matplotlib-based version once the plotting dependencies are installed.