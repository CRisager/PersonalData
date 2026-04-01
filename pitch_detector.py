"""
Pitch Detection Module using Librosa
Detects fundamental frequency (pitch) in voice recordings
"""

from pathlib import Path
import argparse


def _detect_pitch_librosa(audio_path, fmin=80, fmax=400):
    """Pitch detection with librosa.pyin."""
    import numpy as np
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )
    time = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
    return time, f0, voiced_flag


def _detect_pitch_crepe(audio_path, fmin=80, fmax=400):
    """Pitch detection with torchcrepe (CREPE model)."""
    import numpy as np
    import librosa
    import torch
    import torchcrepe

    target_sr = 16000
    hop_length = 160  # 10 ms at 16 kHz

    y, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

    pitch = torchcrepe.predict(
        audio,
        target_sr,
        hop_length,
        fmin,
        fmax,
        model="full",
        batch_size=512,
        device="cpu",
    )
    f0 = pitch.squeeze(0).cpu().numpy()
    voiced_flag = np.isfinite(f0) & (f0 > 0)
    time = np.arange(len(f0), dtype=float) * (hop_length / target_sr)
    f0[~voiced_flag] = np.nan
    return time, f0, voiced_flag


def detect_pitch(audio_path, fmin=80, fmax=400, backend="crepe"):
    """
    Detects pitch (fundamental frequency) from an audio file using Librosa's YIN algorithm.
    
    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        fmin: Minimum frequency to search for (default: 80 Hz - typical for male voices)
        fmax: Maximum frequency to search for (default: 400 Hz - typical for speech)
        backend: Pitch backend ('crepe' or 'librosa')
        
    Returns:
        Dictionary with:
            - time: time in seconds for each frame
            - frequency: detected frequency in Hz for each frame
            - mean_pitch: mean pitch in Hz (excluding unvoiced frames)
            - min_pitch: minimum pitch in Hz
            - max_pitch: maximum pitch in Hz
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
            time, f0, voiced_flag = _detect_pitch_librosa(audio_path, fmin=fmin, fmax=fmax)
        except ImportError as exc:
            raise ImportError(
                "Missing librosa backend dependencies. Install with: pip install librosa"
            ) from exc
    elif backend == "crepe":
        try:
            time, f0, voiced_flag = _detect_pitch_crepe(audio_path, fmin=fmin, fmax=fmax)
        except ImportError as exc:
            raise ImportError(
                "Missing CREPE backend dependencies. Install with: pip install torchcrepe torchaudio"
            ) from exc
    else:
        raise ValueError("Unsupported backend. Use 'crepe' or 'librosa'.")
    
    # Filter out unvoiced frames (voiced_flag == False)
    voiced_frequencies = f0[voiced_flag]
    
    # Calculate statistics
    if len(voiced_frequencies) > 0:
        # Remove NaN values if any exist
        valid_frequencies = voiced_frequencies[~np.isnan(voiced_frequencies)]
        if len(valid_frequencies) > 0:
            mean_pitch = float(np.mean(valid_frequencies))
            min_pitch = float(np.min(valid_frequencies))
            max_pitch = float(np.max(valid_frequencies))
        else:
            mean_pitch = min_pitch = max_pitch = 0.0
        voiced_ratio = float(np.sum(voiced_flag) / len(f0))
    else:
        mean_pitch = min_pitch = max_pitch = 0.0
        voiced_ratio = 0.0
    
    result = {
        "time": time,
        "frequency": f0,
        "voiced_flag": voiced_flag,
        "mean_pitch": mean_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
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
    output = "Pitch Analysis:\n"
    output += "-" * 50 + "\n"
    output += f"Mean Pitch: {pitch_data['mean_pitch']:.2f} Hz\n"
    output += f"Min Pitch:  {pitch_data['min_pitch']:.2f} Hz\n"
    output += f"Max Pitch:  {pitch_data['max_pitch']:.2f} Hz\n"
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
    frequency = pitch_data["frequency"]

    valid = ~np.isnan(frequency)
    if np.sum(valid) == 0:
        raise ValueError("No voiced pitch values found to plot.")

    smoothed = frequency.copy()
    if smooth_seconds > 0 and len(time) > 1:
        frame_step = float(np.median(np.diff(time)))
        window = int(round(smooth_seconds / frame_step))
        if window > 1:
            kernel = np.ones(window, dtype=float)
            values = np.where(valid, frequency, 0.0)
            weights = valid.astype(float)
            smooth_values = np.convolve(values, kernel, mode="same")
            smooth_weights = np.convolve(weights, kernel, mode="same")
            smoothed = smooth_values / np.maximum(smooth_weights, 1e-12)
            smoothed[~valid] = np.nan

    plt.figure(figsize=(10, 4.2))
    plt.plot(
        time,
        frequency,
        linewidth=1.0,
        alpha=0.45,
        label="Raw F0",
    )
    plt.plot(
        time,
        smoothed,
        linewidth=2.0,
        label="Smoothed F0",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (Hz)")
    plt.title(title or "Pitch Over Time")
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
    parser.add_argument("--backend", choices=["crepe", "librosa"], default="crepe", help="Pitch backend to use")
    parser.add_argument("--plot-out", type=str, default=None, help="Optional output path for pitch plot PNG")
    parser.add_argument("--smooth-sec", type=float, default=0.2, help="Smoothing window in seconds (set 0 for raw only)")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}")
        raise SystemExit(1)

    pitch_data = detect_pitch(audio_path, fmin=args.fmin, fmax=args.fmax, backend=args.backend)
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
