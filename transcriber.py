"""
Simple Whisper Transcription App
Records audio from microphone and transcribes it to text
"""

import sounddevice as sd
import soundfile as sf
import whisper
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
SAMPLE_RATE = 16000  # Whisper works best at 16kHz
DURATION = 10  # Record for 10 seconds by default
OUTPUT_DIR = Path("recordings")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Records audio from the microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        numpy array with audio data
    """
    print(f"Recording for {duration} seconds... Press Ctrl+C to stop early")
    
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()  # Wait until recording is finished
        print("Recording complete!")
        return audio
    
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
        return None


def save_audio(audio, filename=None, sample_rate=SAMPLE_RATE):
    """
    Saves audio to a WAV file with timestamp.
    
    Args:
        audio: numpy array with audio data
        filename: output filename (if None, uses timestamp)
        sample_rate: sample rate in Hz
        
    Returns:
        Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
    
    filepath = OUTPUT_DIR / filename
    sf.write(filepath, audio, sample_rate)
    print(f"Audio saved to {filepath}")
    return filepath


def save_transcript(transcript, audio_path, timestamp):
    """
    Saves transcript text to a TXT file with a date/time headline.

    Args:
        transcript: Transcribed text
        audio_path: Path to related audio recording
        timestamp: datetime object used for naming and metadata

    Returns:
        Path to saved transcript file
    """
    timestamp_for_file = timestamp.strftime("%Y%m%d_%H%M%S")
    timestamp_for_header = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    transcript_path = OUTPUT_DIR / f"transcript_{timestamp_for_file}.txt"

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"Transcription - {timestamp_for_header}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Audio file: {audio_path.name}\n")
        f.write(f"Saved at: {timestamp_for_header}\n")
        f.write("\nTranscript:\n")
        f.write("-" * 50 + "\n")
        f.write(transcript.strip() if transcript.strip() else "[No text detected]")
        f.write("\n")

    print(f"Transcript saved to {transcript_path}")
    return transcript_path


def transcribe_audio(audio_path, model_size="base.en"):
    """
    Transcribes audio file using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                   'base' is a good balance between speed and accuracy
        
    Returns:
        Transcribed text
    """
    import ssl
    import urllib.request
    
    # Handle SSL certificate verification issues
    ssl._create_default_https_context = ssl._create_unverified_context
    
    print(f"\nLoading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    
    print("Transcribing audio...")
    result = model.transcribe(str(audio_path))
    
    return result["text"]


def main():
    """Main function to record and transcribe"""
    
    # Step 1: Record audio
    print("=" * 50)
    print("WHISPER TRANSCRIPTION APP")
    print("=" * 50)
    
    audio = record_audio(duration=DURATION)
    
    if audio is None:
        print("No audio recorded")
        return

    # Use one shared timestamp so audio and transcript filenames match.
    recording_time = datetime.now()
    
    # Step 2: Save audio with timestamp
    audio_filename = f"recording_{recording_time.strftime('%Y%m%d_%H%M%S')}.wav"
    audio_path = save_audio(audio, filename=audio_filename)
    
    # Step 3: Transcribe
    transcript = transcribe_audio(audio_path)
    
    # Step 4: Display results
    print("\n" + "=" * 50)
    print("TRANSCRIPTION RESULT:")
    print("=" * 50)
    print(transcript if transcript.strip() else "[No text detected - try speaking louder!]")
    print("=" * 50)
    
    # Save transcript with matching timestamp and a date/time headline.
    transcript_path = save_transcript(transcript, audio_path, recording_time)
    print(f"\n✓ Transcript saved to {transcript_path}")
    print(f"✓ Audio saved to {audio_path}")
    print(f"\n📁 All files stored in: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
