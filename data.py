import kagglehub

# Download latest version
path = kagglehub.dataset_download("pavanelisetty/sample-audio-files-for-speech-recognition")

print("Path to dataset files:", path)