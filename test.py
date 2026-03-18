from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import av
import numpy as np
import whisper


def configure_runtime() -> None:
	"""Configure runtime for Windows/Conda Whisper execution."""
	# Workaround for duplicate OpenMP runtimes commonly seen on Windows.
	os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

	# Ensure Conda runtime DLL folders are available when env is not activated.
	env_root = Path(sys.executable).parent
	dll_paths = [
		env_root / "Library" / "bin",
		env_root / "Library" / "usr" / "bin",
		env_root / "DLLs",
		env_root / "bin",
	]
	prepend = os.pathsep.join(str(p) for p in dll_paths if p.exists())
	if prepend:
		os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")


def load_audio_with_av(audio_path: Path) -> np.ndarray:
	"""Decode audio to 16kHz mono float32 samples using PyAV."""
	container = av.open(str(audio_path))
	resampler = av.audio.resampler.AudioResampler(
		format="s16",
		layout="mono",
		rate=16000,
	)

	chunks: list[np.ndarray] = []
	for frame in container.decode(audio=0):
		resampled_frames = resampler.resample(frame)
		if not isinstance(resampled_frames, list):
			resampled_frames = [resampled_frames]
		for resampled in resampled_frames:
			if resampled is not None:
				chunks.append(resampled.to_ndarray().reshape(-1))

	flush_frames = resampler.resample(None)
	if flush_frames is not None:
		if not isinstance(flush_frames, list):
			flush_frames = [flush_frames]
		for frame in flush_frames:
			if frame is not None:
				chunks.append(frame.to_ndarray().reshape(-1))

	container.close()

	if not chunks:
		raise RuntimeError(f"Failed to decode audio data from: {audio_path}")

	audio = np.concatenate(chunks).astype(np.float32) / 32768.0
	return audio


def transcribe_audio(
	audio_path: Path,
	model_name: str = "base",
	language: str | None = None,
) -> dict:
	"""Load an audio file (e.g., mp3/m4a/wav) and transcribe it with Whisper."""
	if not audio_path.exists():
		raise FileNotFoundError(f"Audio file not found: {audio_path}")

	configure_runtime()
	model = whisper.load_model(model_name)
	audio = load_audio_with_av(audio_path)
	result = model.transcribe(audio, language=language)
	return result


def format_timestamp(seconds: float) -> str:
	total_ms = int(seconds * 1000)
	hours = total_ms // 3_600_000
	minutes = (total_ms % 3_600_000) // 60_000
	secs = (total_ms % 60_000) // 1000
	millis = total_ms % 1000
	return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def build_segment_text(result: dict) -> str:
	segments = result.get("segments", [])
	if not segments:
		return result.get("text", "").strip()

	lines: list[str] = []
	for segment in segments:
		start = format_timestamp(float(segment.get("start", 0.0)))
		end = format_timestamp(float(segment.get("end", 0.0)))
		text = str(segment.get("text", "")).strip()
		if text:
			lines.append(f"[{start} - {end}] {text}")

	return "\n".join(lines)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Transcribe an audio file (mp3, m4a, wav, etc.) using Whisper"
	)
	parser.add_argument("audio_file", type=Path, help="Path to the audio file")
	parser.add_argument(
		"--model",
		default="base",
		help="Whisper model name (tiny, base, small, medium, large)",
	)
	parser.add_argument(
		"--language",
		default=None,
		help="Language code like 'en', 'da', etc. (optional)",
	)
	parser.add_argument(
		"--out",
		type=Path,
		default=None,
		help="Optional output text file path",
	)

	args = parser.parse_args()
	result = transcribe_audio(
		audio_path=args.audio_file,
		model_name=args.model,
		language=args.language,
	)
	transcript = result.get("text", "").strip()

	print(transcript)

	if args.out is not None:
		args.out.write_text(build_segment_text(result), encoding="utf-8")
		print(f"\nSaved transcript to: {args.out}")


if __name__ == "__main__":
	main()