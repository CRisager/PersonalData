import os

# Fix for OpenMP library conflict on Windows with conda environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import whisper
from pathlib import Path


def transcribe_mp3(audio_file_path: str, model_name: str = "base") -> str:
    """
    Transcribe an mp3 audio file using OpenAI's Whisper model.
    
    Args:
        audio_file_path (str): Path to the mp3 file to transcribe
        model_name (str): Whisper model size - "tiny", "base", "small", "medium", "large"
                         Larger models are more accurate but slower
    
    Returns:
        str: The transcribed text
    """
    # Verify the audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    
    # Load the Whisper model
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    
    # Transcribe the audio file
    print(f"Transcribing audio file: {audio_file_path}...")
    result = model.transcribe(audio_file_path)
    
    return result["text"]


if __name__ == "__main__":
    # Example usage:
    # Replace "path/to/your/audio.mp3" with your actual mp3 file path
    audio_path = "path/to/your/audio.mp3"
    
    try:
        transcribed_text = transcribe_mp3(audio_path, model_name="base")
        print("\n--- Transcription Complete ---")
        print(transcribed_text)
        
        # Optionally save the transcription to a text file
        output_file = Path(audio_path).stem + "_transcription.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        print(f"\nTranscription saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")