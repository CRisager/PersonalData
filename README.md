# PersonalData

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