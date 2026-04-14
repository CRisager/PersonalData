"""Analyze transcript files for filler word usage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_FILLERS = ["uh", "um", "you know", "like"]


def normalize_phrase(value: str) -> str:
	"""Lowercase and normalize whitespace/punctuation for phrase matching."""
	value = value.lower().replace("-", " ")
	value = re.sub(r"[^a-z' ]+", " ", value)
	return " ".join(value.split())


def extract_words(text: str) -> List[str]:
	"""Tokenize words while keeping contractions like "don't"."""
	return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())


def extract_transcript_text(file_text: str) -> str:
	"""Return only the spoken transcript section when using saved transcript format."""
	marker = "Transcript:"
	if marker not in file_text:
		return file_text.strip()

	_, after_marker = file_text.split(marker, 1)
	lines = [line for line in after_marker.splitlines() if set(line.strip()) != {"-"}]
	return "\n".join(lines).strip()


def _extract_from_description(description: str) -> List[str]:
	"""Extract fillers listed in metadata description text (HTML/plain text)."""
	text = re.sub(r"<[^>]+>", " ", description)
	text = text.replace("&ldquo;", '"').replace("&rdquo;", '"').replace("&nbsp;", " ")
	text = re.sub(r"\s+", " ", text)
	found: set[str] = set()

	fillers_block = re.search(
		r"Fillers\s+(.*?)\s+Non-fillers",
		text,
		flags=re.IGNORECASE,
	)
	if fillers_block:
		for label in re.findall(r"-\s*([A-Za-z ]+?)\s*:\s*\d+", fillers_block.group(1)):
			phrase = normalize_phrase(label)
			if phrase:
				found.add(phrase)

	for label in ["uh", "um", "you know", "like"]:
		if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
			found.add(label)

	return sorted(found)


def load_filler_words(dataset_path: Path) -> List[str]:
	"""Load filler words from JSON data with several fallback parsing strategies."""
	with dataset_path.open("r", encoding="utf-8") as file:
		data = json.load(file)

	candidates: set[str] = set()

	def add_value(value: str) -> None:
		phrase = normalize_phrase(value)
		if phrase and phrase not in {"other", "none", "words", "repetitions"}:
			if 1 <= len(phrase.split()) <= 3:
				candidates.add(phrase)

	if isinstance(data, list):
		for item in data:
			if isinstance(item, str):
				add_value(item)
			elif isinstance(item, dict):
				for key in ("word", "label", "text", "token", "filler"):
					value = item.get(key)
					if isinstance(value, str):
						add_value(value)

	if isinstance(data, dict):
		for key in ("fillers", "filler_words", "fillerWords", "labels"):
			value = data.get(key)
			if isinstance(value, list):
				for item in value:
					if isinstance(item, str):
						add_value(item)
					elif isinstance(item, dict):
						for sub_key in ("word", "label", "text", "token", "filler"):
							sub_value = item.get(sub_key)
							if isinstance(sub_value, str):
								add_value(sub_value)

		description = (
			data.get("metadata", {}).get("description")
			if isinstance(data.get("metadata"), dict)
			else None
		)
		if isinstance(description, str):
			for phrase in _extract_from_description(description):
				add_value(phrase)

	if not candidates:
		candidates.update(DEFAULT_FILLERS)

	return sorted(candidates)


def count_filler_usage(transcript_text: str, filler_words: Iterable[str]) -> Dict[str, int]:
	"""Count occurrences for both single-word and multi-word fillers."""
	normalized_text = normalize_phrase(transcript_text)
	tokens = normalized_text.split()
	counts: Counter[str] = Counter()

	for filler in filler_words:
		filler_tokens = filler.split()
		if not filler_tokens:
			continue
		n = len(filler_tokens)
		if n == 1:
			counts[filler] = sum(1 for token in tokens if token == filler)
		else:
			counts[filler] = sum(
				1
				for i in range(len(tokens) - n + 1)
				if tokens[i : i + n] == filler_tokens
			)

	return {word: amount for word, amount in counts.items() if amount > 0}


def analyze_transcript(transcript_path: Path, dataset_path: Path) -> Dict[str, object]:
	"""Analyze one transcript and return filler metrics."""
	transcript_raw = transcript_path.read_text(encoding="utf-8")
	transcript_text = extract_transcript_text(transcript_raw)

	all_words = extract_words(transcript_text)
	total_words = len(all_words)

	filler_words = load_filler_words(dataset_path)
	filler_counts = count_filler_usage(transcript_text, filler_words)
	total_fillers = sum(filler_counts.values())

	filler_percentage = (total_fillers / total_words * 100.0) if total_words else 0.0

	return {
		"transcript_path": str(transcript_path),
		"dataset_path": str(dataset_path),
		"total_words": total_words,
		"total_filler_words": total_fillers,
		"filler_percentage": round(filler_percentage, 2),
		"filler_word_counts": dict(sorted(filler_counts.items())),
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Count filler words in a transcript.")
	parser.add_argument(
		"transcript",
		nargs="?",
		default="Transcripts/transcript_20260401_143433.txt",
		help="Path to transcript .txt file",
	)
	parser.add_argument(
		"--dataset",
		default="FillerWordData.json",
		help="Path to filler-word JSON dataset",
	)
	parser.add_argument(
		"--output-format",
		choices=["json", "csv"],
		default="json",
		help="Export format for analysis results (default: json)",
	)
	parser.add_argument(
		"--output",
		default=None,
		help="Optional output file path. Defaults to Filler_analysis/filler_analysis_<timestamp>.<ext>",
	)
	return parser.parse_args()


def export_results(results: Dict[str, object], output_path: Path, output_format: str) -> None:
	"""Export analysis results as JSON or CSV."""
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_format == "json":
		with output_path.open("w", encoding="utf-8") as file:
			json.dump(results, file, indent=2, ensure_ascii=False)
		return

	# CSV export uses one row per filler to avoid nested structures in a single cell.
	filler_counts = results.get("filler_word_counts", {})
	with output_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["transcript_path", results.get("transcript_path", "")])
		writer.writerow(["dataset_path", results.get("dataset_path", "")])
		writer.writerow(["total_words", results.get("total_words", 0)])
		writer.writerow(["total_filler_words", results.get("total_filler_words", 0)])
		writer.writerow(["filler_percentage", results.get("filler_percentage", 0.0)])
		writer.writerow([])
		writer.writerow(["filler_word", "count"])
		for word, count in filler_counts.items():
			writer.writerow([word, count])


def main() -> None:
	args = parse_args()
	transcript_path = Path(args.transcript)
	dataset_path = Path(args.dataset)

	if not transcript_path.exists():
		raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

	results = analyze_transcript(transcript_path, dataset_path)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	default_output = Path("Filler_analysis") / f"filler_analysis_{timestamp}.{args.output_format}"
	output_path = Path(args.output) if args.output else default_output

	export_results(results, output_path, args.output_format)

	print("Filler Word Analysis")
	print("=" * 40)
	print(f"Transcript: {results['transcript_path']}")
	print(f"Total words: {results['total_words']}")
	print(f"Total filler words: {results['total_filler_words']}")
	print(f"Filler percentage: {results['filler_percentage']}%")
	print("Filler word counts:")
	print(results["filler_word_counts"])
	print(f"Results exported to: {output_path}")


if __name__ == "__main__":
	main()

# Run by doing this in terminal:
# python.exe FillerWords.py Transcripts/transcript_20260401_143433.txt 