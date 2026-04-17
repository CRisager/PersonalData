"""End-to-end speech analysis pipeline.

This script orchestrates the existing modules to:
1. Record audio
2. Transcribe speech
3. Detect pitch statistics
4. Compute speaking speed (WPM)
5. Analyze filler-word usage
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DURATION = 30
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TRIM_SILENCE = True
DEFAULT_TRIM_TOP_DB = 30
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
DEFAULT_DATASET_PATH = BACKEND_DIR / "FillerWordData.json"
SPEED_OUTPUT_DIR = Path("Speed")
SPEED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PITCH_OUTPUT_DIR = Path("Pitch")
PITCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Record audio and run transcription + pitch + speed + filler analysis."
	)
	parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Recording duration in seconds")
	parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio sample rate in Hz")
	parser.add_argument("--model", default="base.en", help="Whisper model size (e.g., tiny, base.en, small)")

	parser.add_argument("--fmin", type=float, default=80.0, help="Min pitch in Hz")
	parser.add_argument("--fmax", type=float, default=400.0, help="Max pitch in Hz")
	parser.add_argument("--backend", choices=["librosa"], default="librosa", help="Pitch backend")
	parser.add_argument("--no-trim-silence", action="store_true", help="Disable silence trimming")
	parser.add_argument("--trim-top-db", type=float, default=DEFAULT_TRIM_TOP_DB, help="Silence threshold in dB")
	parser.add_argument(
		"--no-trim-unvoiced-edges",
		action="store_true",
		help="Keep leading/trailing unvoiced frames in pitch output",
	)
	parser.add_argument("--plot-pitch", action="store_true", help="Save pitch plot PNG to Sound_recordings/")
	parser.add_argument(
		"--pitch-plot-output",
		default=None,
		help="Optional custom path for pitch plot (used only with --plot-pitch)",
	)

	parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to filler-word dataset JSON")
	parser.add_argument(
		"--filler-output-format",
		choices=["json", "csv"],
		default="json",
		help="Export format for filler analysis",
	)
	parser.add_argument("--filler-output", default=None, help="Optional custom output file for filler analysis")
	parser.add_argument("--serve", action="store_true", help="Run web server for index.html + API")
	parser.add_argument("--host", default=DEFAULT_HOST, help="Host for web server")
	parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for web server")
	return parser.parse_args()


def sanitize_filename_component(value: str | None, fallback: str = "recording") -> str:
	"""Sanitize a string so it is safe for cross-platform filenames."""
	if not value:
		return fallback
	cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
	cleaned = re.sub(r"_+", "_", cleaned).strip("_")
	return cleaned or fallback


def process_audio_array(
	audio,
	*,
	args: argparse.Namespace,
	recording_name: str | None = None,
	category: str | None = None,
) -> dict[str, Any]:
	"""Run the analysis pipeline for a mono float array and return saved outputs + metrics."""
	trim_silence = not args.no_trim_silence

	try:
		import numpy as np

		from backend.FillerWords import analyze_transcript, export_results
		from backend.pitch_detector import detect_pitch, format_pitch_stats, save_pitch_plot
		from backend.speed import calculate_wpm, count_words, get_wav_duration_seconds
		from backend.transcriber import (
			AUDIO_OUTPUT_DIR,
			TRANSCRIPT_OUTPUT_DIR,
			save_audio,
			save_transcript,
			transcribe_audio,
			trim_saved_audio_file,
		)
	except ImportError as exc:
		missing = str(exc)
		raise SystemExit(
			"Missing dependency while loading pipeline modules. "
			"Install required packages (e.g. sounddevice, soundfile, openai-whisper, librosa, numpy, scipy, matplotlib) "
			f"and retry. Details: {missing}"
		)

	audio = np.asarray(audio, dtype=np.float32).squeeze()
	if audio.ndim != 1:
		raise ValueError("Expected mono audio input.")
	if len(audio) == 0:
		raise ValueError("No audio provided.")

	# Use one timestamp so generated files are tied to the same run.
	run_time = datetime.now()
	timestamp = run_time.strftime("%Y%m%d_%H%M%S")
	preferred_stem = sanitize_filename_component(recording_name, fallback=f"recording_{timestamp}")

	def select_available_output_stem(stem: str) -> str:
		"""Pick a shared filename stem that does not overwrite existing outputs."""
		index = 1
		while True:
			candidate = stem if index == 1 else f"{stem}_{index}"
			conflicts = [
				AUDIO_OUTPUT_DIR / f"{candidate}.wav",
				TRANSCRIPT_OUTPUT_DIR / f"{candidate}.txt",
				Path("Filler_analysis") / f"{candidate}.{args.filler_output_format}",
				SPEED_OUTPUT_DIR / f"{candidate}.json",
				PITCH_OUTPUT_DIR / f"{candidate}.json",
			]
			if args.plot_pitch and args.pitch_plot_output is None:
				conflicts.append(AUDIO_OUTPUT_DIR / f"{candidate}_pitch.png")

			if all(not path.exists() for path in conflicts):
				return candidate
			index += 1

	output_stem = select_available_output_stem(preferred_stem)

	# 2) Save audio
	audio_filename = f"{output_stem}.wav"
	audio_path = save_audio(audio, filename=audio_filename, sample_rate=args.sample_rate)

	if trim_silence:
		trim_info = trim_saved_audio_file(audio_path, top_db=args.trim_top_db)
		if trim_info["trimmed"]:
			print(
				f"Trimmed saved audio: start {trim_info['start_seconds']:.2f}s, "
				f"end {trim_info['end_seconds']:.2f}s "
				f"(kept {trim_info['kept_seconds']:.2f}s)"
			)
		elif trim_info["removed_all"]:
			print("Silence trimming would remove all audio; keeping saved recording unchanged.")

	# 3) Transcribe
	transcript = transcribe_audio(audio_path, model_size=args.model)
	transcript_path = save_transcript(transcript, audio_path, run_time, filename_stem=output_stem)

	# 4) Pitch detection
	pitch_data = detect_pitch(
		audio_path,
		fmin=args.fmin,
		fmax=args.fmax,
		backend=args.backend,
		trim_silence=trim_silence,
		trim_top_db=args.trim_top_db,
		trim_unvoiced_edges=not args.no_trim_unvoiced_edges,
	)

	pitch_plot_path = None
	if args.plot_pitch:
		pitch_plot_path = Path(args.pitch_plot_output) if args.pitch_plot_output else AUDIO_OUTPUT_DIR / f"{output_stem}_pitch.png"
		save_pitch_plot(
			pitch_data,
			pitch_plot_path,
			title=f"Pitch: {audio_path.name}",
			smooth_seconds=0.2,
		)

	# 5) Speed (WPM)
	duration_seconds = get_wav_duration_seconds(audio_path)
	total_words = count_words(transcript)
	wpm = calculate_wpm(total_words, duration_seconds)

	# 6) Filler words
	dataset_path = Path(args.dataset)
	filler_results = analyze_transcript(transcript_path, dataset_path)
	filler_output = (
		Path(args.filler_output)
		if args.filler_output
		else Path("Filler_analysis") / f"{output_stem}.{args.filler_output_format}"
	)
	export_results(filler_results, filler_output, args.filler_output_format)

	# 7) Save speed metrics
	speed_output = SPEED_OUTPUT_DIR / f"{output_stem}.json"
	speed_metrics = {
		"recording_name": recording_name or output_stem,
		"category": category or "",
		"audio_path": str(audio_path),
		"transcript_path": str(transcript_path),
		"duration_seconds": round(duration_seconds, 3),
		"total_words": total_words,
		"wpm": round(wpm, 2),
		"created_at": run_time.isoformat(timespec="seconds"),
	}
	speed_output.write_text(json.dumps(speed_metrics, indent=2), encoding="utf-8")

	pitch_summary = {
		"mean_pitch": round(float(pitch_data.get("mean_pitch", 0.0)), 2),
		"min_pitch": round(float(pitch_data.get("min_pitch", 0.0)), 2),
		"max_pitch": round(float(pitch_data.get("max_pitch", 0.0)), 2),
		"median_pitch": round(float(pitch_data.get("median_pitch", 0.0)), 2),
		"mean_pitch_semitones": round(float(pitch_data.get("mean_pitch_semitones", 0.0)), 2),
		"min_pitch_semitones": round(float(pitch_data.get("min_pitch_semitones", 0.0)), 2),
		"max_pitch_semitones": round(float(pitch_data.get("max_pitch_semitones", 0.0)), 2),
		"median_pitch_semitones": round(float(pitch_data.get("median_pitch_semitones", 0.0)), 2),
		"pitch_reference_hz": round(float(pitch_data.get("pitch_reference_hz", 0.0)), 2),
		"voiced_ratio": round(float(pitch_data.get("voiced_ratio", 0.0)), 4),
	}

	pitch_series = [
		{
			"time": round(float(time_value), 3),
			"pitch_hz": None if not np.isfinite(frequency_value) else round(float(frequency_value), 2),
			"pitch_semitones": None if not np.isfinite(semitone_value) else round(float(semitone_value), 3),
			"voiced": bool(voiced_value),
		}
		for time_value, frequency_value, semitone_value, voiced_value in zip(
			pitch_data.get("time", []),
			pitch_data.get("frequency", []),
			pitch_data.get("pitch_semitones", []),
			pitch_data.get("voiced_flag", []),
		)
	]

	pitch_output = PITCH_OUTPUT_DIR / f"{output_stem}.json"
	pitch_export = {
		"recording_name": recording_name or output_stem,
		"category": category or "",
		"audio_path": str(audio_path),
		"transcript_path": str(transcript_path),
		"pitch": pitch_summary,
		"pitch_series": pitch_series,
		"detector_settings": {
			"backend": args.backend,
			"fmin": float(args.fmin),
			"fmax": float(args.fmax),
			"trim_silence": bool(trim_silence),
			"trim_top_db": float(args.trim_top_db),
			"trim_unvoiced_edges": bool(not args.no_trim_unvoiced_edges),
		},
		"created_at": run_time.isoformat(timespec="seconds"),
	}
	pitch_output.write_text(json.dumps(pitch_export, indent=2), encoding="utf-8")

	return {
		"audio_path": str(audio_path),
		"transcript_path": str(transcript_path),
		"filler_output": str(filler_output),
		"speed_output": str(speed_output),
		"pitch_output": str(pitch_output),
		"pitch_plot_path": str(pitch_plot_path) if pitch_plot_path else None,
		"transcript": transcript,
		"wpm": round(wpm, 2),
		"pitch": pitch_summary,
		"pitch_series": pitch_series,
		"filler_metrics": filler_results,
		"pitch_text": format_pitch_stats(pitch_data).rstrip(),
	}


def process_audio_file(
	audio_file_path: Path,
	*,
	args: argparse.Namespace,
	recording_name: str | None = None,
	category: str | None = None,
) -> dict[str, Any]:
	"""Load an uploaded audio file, normalize to mono float, and run the pipeline."""
	try:
		import numpy as np
		import soundfile as sf
		from scipy import signal
	except ImportError as exc:
		raise SystemExit(f"Missing dependency while loading uploaded audio: {exc}")

	audio_data, sample_rate = sf.read(str(audio_file_path))
	audio_data = np.asarray(audio_data, dtype=np.float32)
	if audio_data.ndim > 1:
		audio_data = np.mean(audio_data, axis=1)

	if sample_rate != args.sample_rate:
		target_samples = int(len(audio_data) * args.sample_rate / sample_rate)
		audio_data = signal.resample(audio_data, target_samples).astype(np.float32)

	return process_audio_array(
		audio_data,
		args=args,
		recording_name=recording_name,
		category=category,
	)


def print_summary(summary: dict[str, Any]) -> None:
	"""Print a consistent summary for CLI mode."""
	print("\n" + "=" * 60)
	print("PIPELINE SUMMARY")
	print("=" * 60)
	print(f"Audio: {summary['audio_path']}")
	print(f"Transcript file: {summary['transcript_path']}")
	print(f"Speed metrics exported to: {summary['speed_output']}")
	print(f"Pitch analysis exported to: {summary['pitch_output']}")
	print(f"Filler analysis exported to: {summary['filler_output']}")
	if summary["pitch_plot_path"] is not None:
		print(f"Pitch plot saved to: {summary['pitch_plot_path']}")

	print("\nTranscript:")
	print("-" * 60)
	transcript = str(summary.get("transcript", "")).strip()
	print(transcript if transcript else "[No text detected]")
	print("-" * 60)

	print("\n" + str(summary["pitch_text"]))
	print(f"Speaking speed: {summary['wpm']:.2f} WPM")

	filler = summary["filler_metrics"]
	print("\nFiller metrics:")
	print(f"- Total words: {filler['total_words']}")
	print(f"- Total filler words: {filler['total_filler_words']}")
	print(f"- Filler percentage: {filler['filler_percentage']}%")
	print(f"- Filler counts: {filler['filler_word_counts']}")


def create_app(args: argparse.Namespace):
	"""Create Flask app that serves index.html and an audio processing endpoint."""
	from flask import Flask, jsonify, request, send_from_directory

	app = Flask(__name__, static_folder=".", static_url_path="")

	@app.after_request
	def add_cors_headers(response):
		response.headers["Access-Control-Allow-Origin"] = "*"
		response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
		response.headers["Access-Control-Allow-Headers"] = "Content-Type"
		return response

	@app.get("/")
	def index():
		return send_from_directory(Path(".").resolve(), "index.html")

	@app.get("/api/health")
	def health():
		return jsonify({"ok": True})

	@app.route("/api/process-recording", methods=["POST", "OPTIONS"])
	def process_recording():
		if request.method == "OPTIONS":
			return ("", 204)

		uploaded = request.files.get("audio")
		if uploaded is None or uploaded.filename is None:
			return jsonify({"error": "Missing audio file in form-data field 'audio'."}), 400

		recording_name = request.form.get("recording_name", "").strip() or None
		category = request.form.get("category", "").strip() or None

		tmp_path: Path | None = None
		try:
			suffix = Path(uploaded.filename).suffix.lower() or ".wav"
			with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
				tmp_path = Path(tmp_file.name)
			uploaded.save(tmp_path)

			summary = process_audio_file(
				tmp_path,
				args=args,
				recording_name=recording_name,
				category=category,
			)
			return jsonify({"ok": True, "result": summary})
		except Exception as exc:
			return jsonify({"ok": False, "error": str(exc)}), 500
		finally:
			if tmp_path is not None and tmp_path.exists():
				tmp_path.unlink(missing_ok=True)

	return app


def main() -> None:
	args = parse_args()

	if args.serve:
		try:
			import flask  # noqa: F401
		except ImportError as exc:
			raise SystemExit(
				"Flask is required for --serve mode. Install with: pip install flask"
			) from exc

		from backend.transcriber import preload_whisper_model

		preload_whisper_model(args.model)

		app = create_app(args)
		print(f"Serving UI at http://{args.host}:{args.port}")
		app.run(host=args.host, port=args.port, debug=False)
		return

	try:
		from backend.transcriber import record_audio
	except ImportError as exc:
		raise SystemExit(f"Could not import recording module: {exc}")

	print("=" * 60)
	print("SPEECH ANALYSIS PIPELINE")
	print("=" * 60)

	audio = record_audio(duration=args.duration, sample_rate=args.sample_rate)
	if audio is None:
		print("No audio recorded. Exiting.")
		return

	summary = process_audio_array(audio, args=args)
	print_summary(summary)


if __name__ == "__main__":
	main()

# Run by doing this in terminal:
# python main.py --serve --port 8000