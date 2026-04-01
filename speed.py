"""Calculate words-per-minute (WPM) from a transcript file."""

from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path


def extract_transcript_text(file_text: str) -> str:
	"""Return the spoken transcript section when using saved transcript format."""
	marker = "Transcript:"
	if marker not in file_text:
		return file_text.strip()

	_, after_marker = file_text.split(marker, 1)
	lines = [line for line in after_marker.splitlines() if set(line.strip()) != {"-"}]
	return "\n".join(lines).strip()


def extract_audio_filename(file_text: str) -> str | None:
	"""Extract the source audio filename from transcript metadata if present."""
	match = re.search(r"^Audio file:\s*(.+)$", file_text, flags=re.MULTILINE)
	return match.group(1).strip() if match else None


def infer_audio_path_from_transcript(transcript_path: Path, file_text: str) -> Path:
	"""Resolve the audio path from transcript metadata or matching timestamp naming."""
	audio_filename = extract_audio_filename(file_text)
	if audio_filename:
		return transcript_path.parent / audio_filename

	# Fallback for transcript_YYYYMMDD_HHMMSS.txt -> recording_YYYYMMDD_HHMMSS.wav
	if transcript_path.stem.startswith("transcript_"):
		timestamp = transcript_path.stem.replace("transcript_", "", 1)
		return transcript_path.parent / f"recording_{timestamp}.wav"

	raise ValueError(
		"Could not determine audio file from transcript. Add 'Audio file:' header or use --audio."
	)


def get_wav_duration_seconds(audio_path: Path) -> float:
	"""Return WAV duration in seconds using file header metadata."""
	with wave.open(str(audio_path), "rb") as wav_file:
		frame_count = wav_file.getnframes()
		frame_rate = wav_file.getframerate()
		if frame_rate <= 0:
			raise ValueError(f"Invalid WAV frame rate in file: {audio_path}")
		return frame_count / float(frame_rate)


def count_words(text: str) -> int:
	"""Count words while keeping contractions like "don't" as one word."""
	return len(re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text))


def calculate_wpm(total_words: int, duration_seconds: float) -> float:
	"""Convert word count over a duration to words per minute."""
	if duration_seconds <= 0:
		raise ValueError("duration_seconds must be greater than 0")
	return total_words / (duration_seconds / 60.0)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Calculate words per minute from a transcript.")
	parser.add_argument(
		"transcript",
		nargs="?",
		default="recordings/transcript_20260401_145518.txt",
		help="Path to transcript .txt file",
	)
	parser.add_argument(
		"--audio",
		default=None,
		help="Optional path to audio .wav file. If omitted, read from transcript metadata.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	transcript_path = Path(args.transcript)

	if not transcript_path.exists():
		raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

	file_text = transcript_path.read_text(encoding="utf-8")
	transcript_text = extract_transcript_text(file_text)
	total_words = count_words(transcript_text)
	audio_path = Path(args.audio) if args.audio else infer_audio_path_from_transcript(transcript_path, file_text)

	if not audio_path.exists():
		raise FileNotFoundError(f"Audio file not found: {audio_path}")

	duration_seconds = get_wav_duration_seconds(audio_path)
	wpm = calculate_wpm(total_words, duration_seconds)

	# For now, print only the resulting speed.
	print(f"{wpm:.2f}")


if __name__ == "__main__":
	main()

# Run by doing this in terminal:
# python.exe speed.py recordings/transcript_20260401_145518.txt