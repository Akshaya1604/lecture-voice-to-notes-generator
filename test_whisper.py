import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Model loaded successfully!")

audio_file = "sample.ogg"  # change this to your audio file name

print("Transcribing audio...")

result = model.transcribe(audio_file)

print("\nTranscription Completed:\n")
print(result["text"])
