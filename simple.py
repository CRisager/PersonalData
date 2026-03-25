import whisper
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = whisper.load_model("turbo")
result = model.transcribe("harvard.wav")
print(result["text"])