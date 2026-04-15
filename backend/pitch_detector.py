"""
Pitch Detection Module using Librosa
Detects fundamental frequency (pitch) in voice recordings
"""

from pathlib import Path
import argparse


PRAAT_SEMITONE_REFERENCE_HZ = 100.0


def _hz_to_praat_semitones(frequencies, reference_hz=PRAAT_SEMITONE_REFERENCE_HZ):
    """Convert Hertz values to Praat-style semitones relative to a reference frequency."""
    import numpy as np

    frequencies = np.asarray(frequencies, dtype=float)
    semitones = np.full_like(frequencies, np.nan, dtype=float)
    valid = np.isfinite(frequencies) & (frequencies > 0)
    semitones[valid] = 12.0 * np.log2(frequencies[valid] / float(reference_hz))
    return semitones


def _detect_pitch_librosa(
    audio_path,
    fmin=80,
    fmax=400,
    trim_silence=True,
    trim_top_db=30,
    trim_unvoiced_edges=True,
):
    """Pitch detection with librosa.pyin."""
    import numpy as np
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=trim_top_db)

    if len(y) == 0:
        raise ValueError("Audio is empty after silence trimming.")

    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )
    time = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

    if trim_unvoiced_edges:
        voiced_idx = np.flatnonzero(voiced_flag)
        if len(voiced_idx) > 0:
            start = int(voiced_idx[0])
            end = int(voiced_idx[-1]) + 1
            time = time[start:end]
            f0 = f0[start:end]
            voiced_flag = voiced_flag[start:end]

    return time, f0, voiced_flag


def detect_pitch(
    audio_path,
    fmin=80,
    fmax=400,
    backend="librosa",
    trim_silence=True,
    trim_top_db=30,
    trim_unvoiced_edges=True,
):
    """
    Detects pitch (fundamental frequency) from an audio file using Librosa's YIN algorithm.
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        fmin: Minimum frequency to search for (default: 80 Hz - typical for male voices)
        fmax: Maximum frequency to search for (default: 400 Hz - typical for speech)
        backend: Pitch backend ('librosa')
        trim_silence: Whether to trim leading/trailing silence before pitch extraction
        trim_top_db: Silence threshold in dB for trimming (lower = less aggressive)
        trim_unvoiced_edges: Whether to cut leading/trailing unvoiced frames after pitch detection
        
    Returns:
        Dictionary with:
            - time: time in seconds for each frame
            - frequency: detected frequency in Hz for each frame
                - pitch_semitones: detected pitch in Praat-style semitones relative to the file median
            - mean_pitch: mean pitch in Hz (legacy)
            - min_pitch: minimum pitch in Hz (legacy)
            - max_pitch: maximum pitch in Hz (legacy)
            - median_pitch: median pitch in Hz used as the semitone reference
                - mean_pitch_semitones: mean pitch in Praat-style semitones relative to the file median
                - min_pitch_semitones: minimum pitch in Praat-style semitones relative to the file median
                - max_pitch_semitones: maximum pitch in Praat-style semitones relative to the file median
            - median_pitch_semitones: median pitch in semitones, which should be near 0
                - pitch_reference_hz: median voiced frequency used as the semitone reference
            - voiced_ratio: proportion of voiced frames (0-1)
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Missing dependency numpy. Install in whisper_transcriber venv with: pip install numpy"
        ) from exc

    print(f"Loading audio and detecting pitch ({backend})...")

    if backend == "librosa":
        try:
            time, f0, voiced_flag = _detect_pitch_librosa(
                audio_path,
                fmin=fmin,
                fmax=fmax,
                trim_silence=trim_silence,
                trim_top_db=trim_top_db,
                trim_unvoiced_edges=trim_unvoiced_edges,
            )
        except ImportError as exc:
            raise ImportError(
                "Missing librosa backend dependencies. Install with: pip install librosa"
            ) from exc
    else:
        raise ValueError("Unsupported backend. Use 'librosa'.")
    
    # Filter out unvoiced frames (voiced_flag == False)
    voiced_frequencies = f0[voiced_flag]
    valid_frequencies = voiced_frequencies[np.isfinite(voiced_frequencies) & (voiced_frequencies > 0)]
    if len(valid_frequencies) > 0:
        pitch_reference_hz = float(np.median(valid_frequencies))
    else:
        pitch_reference_hz = PRAAT_SEMITONE_REFERENCE_HZ

    pitch_semitones = _hz_to_praat_semitones(f0, reference_hz=pitch_reference_hz)
    
    # Calculate statistics
    if len(valid_frequencies) > 0:
        median_pitch = float(np.median(valid_frequencies))
        mean_pitch = float(np.mean(valid_frequencies))
        min_pitch = float(np.min(valid_frequencies))
        max_pitch = float(np.max(valid_frequencies))
        valid_semitones = _hz_to_praat_semitones(valid_frequencies, reference_hz=pitch_reference_hz)
        median_pitch_semitones = float(np.median(valid_semitones))
        mean_pitch_semitones = float(np.mean(valid_semitones))
        min_pitch_semitones = float(np.min(valid_semitones))
        max_pitch_semitones = float(np.max(valid_semitones))
        voiced_ratio = float(np.sum(voiced_flag) / len(f0))
    else:
        median_pitch = 0.0
        mean_pitch = min_pitch = max_pitch = 0.0
        median_pitch_semitones = 0.0
        mean_pitch_semitones = min_pitch_semitones = max_pitch_semitones = 0.0
        voiced_ratio = 0.0
    
    result = {
        "time": time,
        "frequency": f0,
        "pitch_semitones": pitch_semitones,
        "voiced_flag": voiced_flag,
        "mean_pitch": mean_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "median_pitch": median_pitch,
        "mean_pitch_semitones": mean_pitch_semitones,
        "min_pitch_semitones": min_pitch_semitones,
        "max_pitch_semitones": max_pitch_semitones,
        "median_pitch_semitones": median_pitch_semitones,
        "pitch_reference_hz": pitch_reference_hz,
        "voiced_ratio": voiced_ratio,
    }
    
    return result


def format_pitch_stats(pitch_data):
    """
    Formats pitch statistics as human-readable text.
    
    Args:
        pitch_data: Dictionary returned from detect_pitch()
        
    Returns:
        Formatted string with pitch statistics
    """
    reference_hz = pitch_data.get("pitch_reference_hz", PRAAT_SEMITONE_REFERENCE_HZ)
    median_pitch = pitch_data.get("median_pitch")
    median_pitch_semitones = pitch_data.get("median_pitch_semitones")
    mean_pitch_semitones = pitch_data.get("mean_pitch_semitones")
    min_pitch_semitones = pitch_data.get("min_pitch_semitones")
    max_pitch_semitones = pitch_data.get("max_pitch_semitones")

    if mean_pitch_semitones is None and "mean_pitch" in pitch_data:
        mean_pitch_semitones = float(_hz_to_praat_semitones([pitch_data["mean_pitch"]])[0])
    if min_pitch_semitones is None and "min_pitch" in pitch_data:
        min_pitch_semitones = float(_hz_to_praat_semitones([pitch_data["min_pitch"]])[0])
    if max_pitch_semitones is None and "max_pitch" in pitch_data:
        max_pitch_semitones = float(_hz_to_praat_semitones([pitch_data["max_pitch"]])[0])

    output = "Pitch Analysis:\n"
    output += "-" * 50 + "\n"
    output += f"Reference Hz: {reference_hz:.2f} (median voiced pitch)\n"
    output += f"Median Pitch: {median_pitch:.2f} Hz / {median_pitch_semitones:.2f} st\n"
    output += f"Mean Pitch: {mean_pitch_semitones:.2f} st\n"
    output += f"Min Pitch:  {min_pitch_semitones:.2f} st\n"
    output += f"Max Pitch:  {max_pitch_semitones:.2f} st\n"
    output += f"Voiced Ratio: {pitch_data['voiced_ratio']*100:.1f}%\n"
    
    return output


def save_pitch_plot(
    pitch_data,
    output_path,
    title=None,
    smooth_seconds=0.2,
):
    """
    Saves a pitch-over-time plot as a PNG image.

    Args:
        pitch_data: Dictionary returned from detect_pitch()
        output_path: Path where the image should be saved
        title: Optional plot title
        smooth_seconds: Optional smoothing window in seconds (set 0 to disable)

    Returns:
        Path to saved plot image
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Missing plotting dependency. Install in whisper_transcriber venv with: pip install matplotlib"
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time = pitch_data["time"]
    semitones = pitch_data.get("pitch_semitones")
    if semitones is None:
        semitones = _hz_to_praat_semitones(pitch_data["frequency"])

    valid = np.isfinite(semitones)
    if np.sum(valid) == 0:
        raise ValueError("No voiced pitch values found to plot.")

    frame_idx = np.arange(len(semitones), dtype=float)
    voiced_idx = frame_idx[valid]
    voiced_semitones = semitones[valid]

    # Build a continuous contour by interpolating through unvoiced gaps.
    continuous = np.interp(frame_idx, voiced_idx, voiced_semitones)

    smoothed = continuous.copy()
    if smooth_seconds > 0 and len(time) > 1:
        frame_step = float(np.median(np.diff(time)))
        window = int(round(smooth_seconds / frame_step))
        if window > 1:
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window, dtype=float) / window
            pad = window // 2
            padded = np.pad(continuous, (pad, pad), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")

    plt.figure(figsize=(10, 4.2))
    plt.plot(
        time,
        smoothed,
        linewidth=2.2,
        label="Pitch (semitones re median voiced Hz, interpolated + smoothed)",
    )
    plt.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8, label="Median reference (0 st)")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (semitones re median voiced Hz)")
    plt.title(title or "Pitch Over Time (Praat-style semitones)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main():
    """Simple CLI for pitch detection on an existing audio file."""
    parser = argparse.ArgumentParser(description="Detect pitch from an audio file")
    parser.add_argument("audio_path", help="Path to audio file, e.g. recordings/recording_20260325_123456.wav")
    parser.add_argument("--fmin", type=float, default=80.0, help="Minimum frequency in Hz (default: 80)")
    parser.add_argument("--fmax", type=float, default=400.0, help="Maximum frequency in Hz (default: 400)")
    parser.add_argument("--backend", choices=["librosa"], default="librosa", help="Pitch backend to use")
    parser.add_argument("--no-trim-silence", action="store_true", help="Disable trimming of leading/trailing silence")
    parser.add_argument("--trim-top-db", type=float, default=30.0, help="Silence threshold in dB for trim (default: 30)")
    parser.add_argument("--no-trim-unvoiced-edges", action="store_true", help="Keep leading/trailing unvoiced frames")
    parser.add_argument("--plot-out", type=str, default=None, help="Optional output path for pitch plot PNG")
    parser.add_argument("--smooth-sec", type=float, default=0.2, help="Smoothing window in seconds (set 0 for raw only)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}")
        raise SystemExit(1)

    pitch_data = detect_pitch(
        audio_path,
        fmin=args.fmin,
        fmax=args.fmax,
        backend=args.backend,
        trim_silence=not args.no_trim_silence,
        trim_top_db=args.trim_top_db,
        trim_unvoiced_edges=not args.no_trim_unvoiced_edges,
    )
    print(format_pitch_stats(pitch_data))

    if args.plot_out:
        plot_path = save_pitch_plot(
            pitch_data,
            args.plot_out,
            title=f"Pitch: {audio_path.name}",
            smooth_seconds=args.smooth_sec,
        )
        print(f"Pitch plot saved to: {plot_path}")


if __name__ == "__main__":
    main()

# Run by doing this in terminal:
# python.exe pitch_detector.py recordings/recording_20260401_145518.wav --backend librosa