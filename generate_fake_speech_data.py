"""Generate synthetic speech-analysis data for plotting and demos.

The output is designed to support the plots requested in this project:
- filler-word position density
- filler-word counts over time
- rolling words-per-minute over time
- filler-word word cloud summaries
- pitch variation plots
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FILLER_WORDS = ["um", "uh", "you know", "like"]

CONTENT_WORDS = [
    "project",
    "plan",
    "timeline",
    "team",
    "budget",
    "meeting",
    "design",
    "data",
    "results",
    "question",
    "idea",
    "change",
    "update",
    "research",
    "analysis",
    "feature",
    "context",
    "example",
    "system",
    "workflow",
    "decision",
    "signal",
    "trend",
    "pattern",
    "speaker",
    "recording",
    "visual",
    "chart",
    "plot",
    "voice",
    "pitch",
    "pause",
    "sentence",
    "response",
    "process",
    "summary",
    "detail",
    "moment",
    "progress",
    "focus",
    "reason",
    "topic",
    "question",
    "answer",
    "thought",
    "direction",
    "context",
    "sequence",
    "section",
    "insight",
    "variation",
    "shift",
    "momentum",
    "opening",
    "conclusion",
    "middle",
    "start",
    "end",
    "review",
    "compare",
    "clarify",
    "measure",
    "observe",
    "track",
    "listen",
    "speak",
    "think",
    "move",
    "grow",
    "reduce",
    "repeat",
    "linger",
    "steady",
    "quick",
    "slow",
    "strong",
    "light",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic speech-analysis data.")
    parser.add_argument(
        "--output",
        default="recordings/fake_speech_dataset.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--recordings",
        type=int,
        default=8,
        help="Number of synthetic recordings to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible output.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=120.0,
        help="Minimum recording length in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=210.0,
        help="Maximum recording length in seconds.",
    )
    parser.add_argument(
        "--pitch-step",
        type=float,
        default=0.2,
        help="Step size in seconds for pitch samples.",
    )
    parser.add_argument(
        "--bin-seconds",
        type=float,
        default=10.0,
        help="Bin width in seconds for filler-time summaries.",
    )
    return parser.parse_args()


def make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def weighted_choice(rng: random.Random, items: dict[str, float]) -> str:
    labels = list(items.keys())
    weights = list(items.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def choose_content_word(rng: random.Random) -> str:
    return rng.choice(CONTENT_WORDS)


def pace_multiplier(profile: str, normalized_time: float) -> float:
    normalized_time = max(0.0, min(1.0, normalized_time))
    if profile == "accelerating":
        return 0.88 + 0.34 * normalized_time
    if profile == "decelerating":
        return 1.18 - 0.30 * normalized_time
    if profile == "slow_middle":
        return 1.06 - 0.28 * math.exp(-((normalized_time - 0.5) ** 2) / 0.02)
    if profile == "burst_then_settle":
        return 1.22 if normalized_time < 0.18 else 0.95 + 0.12 * normalized_time
    return 1.0 + 0.05 * math.sin(2.0 * math.pi * normalized_time * 2.0)


def pitch_target(profile: str, normalized_time: float) -> float:
    if profile == "rising":
        return 1.0 + 0.28 * normalized_time
    if profile == "falling":
        return 1.18 - 0.24 * normalized_time
    if profile == "wave":
        return 1.05 + 0.12 * math.sin(2.0 * math.pi * normalized_time * 3.0)
    if profile == "unstable":
        return 1.0 + 0.10 * math.sin(2.0 * math.pi * normalized_time * 7.0)
    return 1.0


def sentence_boundary_count(rng: random.Random) -> int:
    return rng.randint(8, 15)


def build_transcript(words: Iterable[dict[str, Any]], rng: random.Random) -> str:
    sentences: list[str] = []
    current: list[str] = []

    for entry in words:
        current.append(entry["word"])
        if entry["sentence_break"]:
            sentence = " ".join(current).strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                sentence += rng.choice([".", ".", ".", "!", "?"])
                sentences.append(sentence)
            current = []

    if current:
        sentence = " ".join(current).strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            sentence += "."
            sentences.append(sentence)

    return " ".join(sentences)


def generate_recording(
    rng: random.Random,
    recording_index: int,
    duration_seconds: float,
    pitch_step: float,
    bin_seconds: float,
) -> dict[str, Any]:
    speaker_id = f"speaker_{rng.randint(1, 8):02d}"
    pace_profile = rng.choice(["accelerating", "decelerating", "slow_middle", "burst_then_settle", "steady"])
    pitch_profile = rng.choice(["rising", "falling", "wave", "unstable", "flat"])

    base_wpm = rng.uniform(115.0, 168.0)
    base_pitch_hz = rng.uniform(118.0, 205.0)
    pitch_span_hz = rng.uniform(12.0, 28.0)

    target_words = max(60, int(duration_seconds * base_wpm / 60.0))
    word_events: list[dict[str, Any]] = []
    filler_events: list[dict[str, Any]] = []
    transcript_tokens: list[dict[str, Any]] = []

    current_time = 0.0
    words_created = 0
    sentence_limit = sentence_boundary_count(rng)
    sentence_progress = 0
    filler_bias = rng.uniform(0.08, 0.16)
    early_opening_window = min(0.18 * duration_seconds, 24.0)
    local_pause_bias = rng.uniform(0.02, 0.10)

    while words_created < target_words and current_time < duration_seconds - 0.25:
        normalized_time = current_time / duration_seconds if duration_seconds > 0 else 0.0
        pace = pace_multiplier(pace_profile, normalized_time)

        filler_probability = filler_bias
        if current_time <= early_opening_window:
            filler_probability += 0.10
        if words_created < 10:
            filler_probability += 0.06
        if word_events and word_events[-1]["word"] in {"and", "so", "because", "but"}:
            filler_probability -= 0.01

        is_filler = rng.random() < filler_probability
        if is_filler:
            word = weighted_choice(
                rng,
                {
                    "um": 0.38,
                    "uh": 0.32,
                    "you know": 0.18,
                    "like": 0.12,
                },
            )
        else:
            word = choose_content_word(rng)

        base_duration = 0.09 + 0.014 * len(word.replace(" ", ""))
        if is_filler:
            base_duration *= 0.68
        duration_word = max(0.07, base_duration / pace)

        start_time = current_time
        end_time = min(duration_seconds, start_time + duration_word)

        sentence_progress += 1
        boundary_trigger = sentence_progress >= sentence_limit
        pause_after = rng.uniform(0.03, 0.10) + local_pause_bias
        if is_filler:
            pause_after += rng.uniform(0.01, 0.06)
        if boundary_trigger:
            pause_after += rng.uniform(0.18, 0.75)

        sentence_break = boundary_trigger or current_time + duration_word + pause_after >= duration_seconds - 0.2

        word_entry = {
            "word": word,
            "kind": "filler" if is_filler else "content",
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "duration": round(end_time - start_time, 3),
            "position_ratio": round(start_time / duration_seconds, 6) if duration_seconds else 0.0,
            "sentence_break": sentence_break,
        }
        word_events.append(word_entry)
        transcript_tokens.append(word_entry)

        if is_filler:
            filler_events.append(
                {
                    "word": word,
                    "start": word_entry["start"],
                    "end": word_entry["end"],
                    "position_ratio": word_entry["position_ratio"],
                }
            )

        words_created += 1
        current_time = end_time + pause_after

        if boundary_trigger:
            sentence_limit = sentence_boundary_count(rng)
            sentence_progress = 0

    transcript_text = build_transcript(transcript_tokens, rng)

    filler_counts = Counter(event["word"] for event in filler_events)
    total_fillers = sum(filler_counts.values())
    total_words = len(word_events)
    wpm = total_words / (duration_seconds / 60.0) if duration_seconds > 0 else 0.0

    filler_bin_counts: list[dict[str, Any]] = []
    cumulative_by_word = defaultdict(int)
    cumulative_total = 0
    cumulative_series: list[dict[str, Any]] = []
    time_bins = int(math.ceil(duration_seconds / bin_seconds)) if bin_seconds > 0 else 0
    for bin_index in range(time_bins):
        bin_start = bin_index * bin_seconds
        bin_end = min(duration_seconds, bin_start + bin_seconds)
        bin_events = [event for event in filler_events if bin_start <= event["start"] < bin_end]
        counts = Counter(event["word"] for event in bin_events)
        row = {
            "time_start": round(bin_start, 3),
            "time_end": round(bin_end, 3),
            "total": sum(counts.values()),
        }
        for filler_word in FILLER_WORDS:
            row[filler_word] = counts.get(filler_word, 0)
        filler_bin_counts.append(row)

        cumulative_total += row["total"]
        for filler_word in FILLER_WORDS:
            cumulative_by_word[filler_word] += row[filler_word]
        cumulative_row = {
            "time_start": round(bin_start, 3),
            "time_end": round(bin_end, 3),
            "total": cumulative_total,
        }
        for filler_word in FILLER_WORDS:
            cumulative_row[filler_word] = cumulative_by_word[filler_word]
        cumulative_series.append(cumulative_row)

    rolling_wpm: list[dict[str, Any]] = []
    sample_time = 0.0
    while sample_time <= duration_seconds + 1e-9:
        window_start = max(0.0, sample_time - 30.0)
        window_end = sample_time
        window_word_count = sum(
            1
            for event in word_events
            if window_start <= event["start"] <= window_end
        )
        window_seconds = max(5.0, window_end - window_start)
        rolling_wpm.append(
            {
                "time": round(sample_time, 3),
                "window_seconds": round(window_seconds, 3),
                "wpm": round(window_word_count * 60.0 / window_seconds, 2),
            }
        )
        sample_time += 1.0

    pitch_track: list[dict[str, Any]] = []
    timestamp = 0.0
    word_index = 0
    while timestamp <= duration_seconds + 1e-9:
        while word_index < len(word_events) and word_events[word_index]["end"] < timestamp:
            word_index += 1

        in_word = word_index < len(word_events) and word_events[word_index]["start"] <= timestamp <= word_events[word_index]["end"]
        normalized_time = timestamp / duration_seconds if duration_seconds > 0 else 0.0
        pitch_scale = pitch_target(pitch_profile, normalized_time)

        nervous_opening_boost = 10.0 * max(0.0, 1.0 - normalized_time / 0.18) if normalized_time < 0.18 else 0.0
        contour = base_pitch_hz * pitch_scale + pitch_span_hz * math.sin(2.0 * math.pi * (normalized_time * 2.0 + 0.15))
        contour += 2.5 * math.sin(2.0 * math.pi * normalized_time * 9.0)
        contour += nervous_opening_boost
        contour += rng.uniform(-3.5, 3.5)

        if not in_word and rng.random() < 0.72:
            voiced = False
            pitch_hz = None
        else:
            voiced = True
            pitch_hz = round(max(70.0, min(360.0, contour)), 2)

        pitch_track.append(
            {
                "time": round(timestamp, 3),
                "pitch_hz": pitch_hz,
                "voiced": voiced,
            }
        )
        timestamp += pitch_step

    voiced_values = [entry["pitch_hz"] for entry in pitch_track if entry["pitch_hz"] is not None]
    pitch_summary = {
        "mean_hz": round(sum(voiced_values) / len(voiced_values), 2) if voiced_values else 0.0,
        "median_hz": round(sorted(voiced_values)[len(voiced_values) // 2], 2) if voiced_values else 0.0,
        "min_hz": round(min(voiced_values), 2) if voiced_values else 0.0,
        "max_hz": round(max(voiced_values), 2) if voiced_values else 0.0,
        "range_hz": round(max(voiced_values) - min(voiced_values), 2) if voiced_values else 0.0,
        "voiced_ratio": round(len(voiced_values) / len(pitch_track), 4) if pitch_track else 0.0,
    }

    return {
        "recording_id": f"recording_{recording_index:02d}",
        "speaker_id": speaker_id,
        "duration_seconds": round(duration_seconds, 2),
        "base_wpm_target": round(base_wpm, 2),
        "actual_wpm": round(wpm, 2),
        "pace_profile": pace_profile,
        "pitch_profile": pitch_profile,
        "transcript": transcript_text,
        "word_events": word_events,
        "filler_events": filler_events,
        "filler_counts": dict(sorted(filler_counts.items())),
        "filler_time_bins": filler_bin_counts,
        "filler_time_bins_cumulative": cumulative_series,
        "rolling_wpm_30s": rolling_wpm,
        "pitch_track": pitch_track,
        "pitch_summary": pitch_summary,
        "total_words": total_words,
        "total_filler_words": total_fillers,
        "filler_percentage": round((total_fillers / total_words * 100.0) if total_words else 0.0, 2),
    }


def build_word_cloud_summary(recordings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter()
    recording_hits = Counter()

    for recording in recordings:
        filler_counts = recording.get("filler_counts", {})
        for word, count in filler_counts.items():
            counts[word] += count
            recording_hits[word] += 1

    return [
        {
            "word": word,
            "count": counts[word],
            "recording_count": recording_hits[word],
        }
        for word in FILLER_WORDS
        if counts[word] > 0
    ]


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = make_rng(args.seed)
    recordings: list[dict[str, Any]] = []

    for recording_index in range(1, args.recordings + 1):
        duration_seconds = rng.uniform(args.min_duration, args.max_duration)
        recordings.append(
            generate_recording(
                rng=rng,
                recording_index=recording_index,
                duration_seconds=duration_seconds,
                pitch_step=args.pitch_step,
                bin_seconds=args.bin_seconds,
            )
        )

    total_words = sum(recording["total_words"] for recording in recordings)
    total_fillers = sum(recording["total_filler_words"] for recording in recordings)
    filler_summary = build_word_cloud_summary(recordings)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "recording_count": len(recordings),
        "total_words": total_words,
        "total_filler_words": total_fillers,
        "overall_filler_percentage": round((total_fillers / total_words * 100.0) if total_words else 0.0, 2),
        "filler_word_summary": filler_summary,
        "recordings": recordings,
    }


def main() -> None:
    args = parse_args()
    if args.min_duration <= 0 or args.max_duration <= 0:
        raise ValueError("Durations must be positive.")
    if args.min_duration > args.max_duration:
        raise ValueError("--min-duration must be less than or equal to --max-duration.")
    if args.recordings <= 0:
        raise ValueError("--recordings must be greater than 0.")
    if args.pitch_step <= 0:
        raise ValueError("--pitch-step must be greater than 0.")
    if args.bin_seconds <= 0:
        raise ValueError("--bin-seconds must be greater than 0.")

    dataset = build_dataset(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)

    print(f"Synthetic speech dataset written to {output_path}")
    print(f"Recordings: {dataset['recording_count']}")
    print(f"Total words: {dataset['total_words']}")
    print(f"Total filler words: {dataset['total_filler_words']}")


if __name__ == "__main__":
    main()