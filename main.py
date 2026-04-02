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
from datetime import datetime
from pathlib import Path

DEFAULT_DURATION = 30
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TRIM_SILENCE = True
DEFAULT_TRIM_TOP_DB = 30


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
	parser.add_argument("--plot-pitch", action="store_true", help="Save pitch plot PNG to recordings/")
	parser.add_argument(
		"--pitch-plot-output",
		default=None,
		help="Optional custom path for pitch plot (used only with --plot-pitch)",
	)

	parser.add_argument("--dataset", default="FillerWordData.json", help="Path to filler-word dataset JSON")
	parser.add_argument(
		"--filler-output-format",
		choices=["json", "csv"],
		default="json",
		help="Export format for filler analysis",
	)
	parser.add_argument("--filler-output", default=None, help="Optional custom output file for filler analysis")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	trim_silence = not args.no_trim_silence

	try:
		from FillerWords import analyze_transcript, export_results
		from pitch_detector import detect_pitch, format_pitch_stats, save_pitch_plot
		from speed import calculate_wpm, count_words, get_wav_duration_seconds
		from transcriber import (
			OUTPUT_DIR,
			record_audio,
			save_audio,
			save_transcript,
			transcribe_audio,
			trim_audio_edges,
		)
	except ImportError as exc:
		missing = str(exc)
		raise SystemExit(
			"Missing dependency while loading pipeline modules. "
			"Install required packages (e.g. sounddevice, soundfile, openai-whisper, librosa, numpy, scipy, matplotlib) "
			f"and retry. Details: {missing}"
		)

	print("=" * 60)
	print("SPEECH ANALYSIS PIPELINE")
	print("=" * 60)

	# 1) Record
	audio = record_audio(duration=args.duration, sample_rate=args.sample_rate)
	if audio is None:
		print("No audio recorded. Exiting.")
		return

	if trim_silence:
		trimmed_audio, cut_start, cut_end = trim_audio_edges(
			audio,
			sample_rate=args.sample_rate,
			top_db=args.trim_top_db,
		)
		if len(trimmed_audio) > 0:
			audio = trimmed_audio
			kept_seconds = len(audio) / float(args.sample_rate)
			print(
				f"Trimmed silence: start {cut_start:.2f}s, end {cut_end:.2f}s "
				f"(kept {kept_seconds:.2f}s)"
			)
		else:
			print("Silence trimming removed all audio; keeping original recording.")

	# Use one timestamp so generated files are tied to the same run.
	run_time = datetime.now()
	timestamp = run_time.strftime("%Y%m%d_%H%M%S")

	# 2) Save audio
	audio_filename = f"recording_{timestamp}.wav"
	audio_path = save_audio(audio, filename=audio_filename, sample_rate=args.sample_rate)

	# 3) Transcribe
	transcript = transcribe_audio(audio_path, model_size=args.model)
	transcript_path = save_transcript(transcript, audio_path, run_time)

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
		pitch_plot_path = Path(args.pitch_plot_output) if args.pitch_plot_output else OUTPUT_DIR / f"pitch_{timestamp}.png"
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
		else OUTPUT_DIR / f"filler_analysis_{timestamp}.{args.filler_output_format}"
	)
	export_results(filler_results, filler_output, args.filler_output_format)

	# Final summary
	print("\n" + "=" * 60)
	print("PIPELINE SUMMARY")
	print("=" * 60)
	print(f"Audio: {audio_path}")
	print(f"Transcript file: {transcript_path}")
	print("\nTranscript:")
	print("-" * 60)
	print(transcript.strip() if transcript.strip() else "[No text detected]")
	print("-" * 60)

	print("\n" + format_pitch_stats(pitch_data).rstrip())
	print(f"Speaking speed: {wpm:.2f} WPM")
	print(f"Filler analysis exported to: {filler_output}")
	if pitch_plot_path is not None:
		print(f"Pitch plot saved to: {pitch_plot_path}")

	print("\nFiller metrics:")
	print(f"- Total words: {filler_results['total_words']}")
	print(f"- Total filler words: {filler_results['total_filler_words']}")
	print(f"- Filler percentage: {filler_results['filler_percentage']}%")
	print(f"- Filler counts: {filler_results['filler_word_counts']}")


if __name__ == "__main__":
	main()

# Run by doing this in terminal:
# python.exe main.py --duration 20 --model base.en --plot-pitch --filler-output-format json