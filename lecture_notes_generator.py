import whisper
from transformers import pipeline

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading Summarization model...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def transcribe_audio(audio_path):
    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    print("Generating summary...")
    summary = summarizer(
        text,
        max_length=120,
        min_length=40,
        do_sample=False
    )
    return summary[0]["summary_text"]
def generate_quiz(summary_text):
    print("Generating quiz questions...")
    
    sentences = summary_text.split(".")
    questions = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:
            words = sentence.split()
            if len(words) > 4:
                question = "What is meant by " + " ".join(words[:4]) + "?"
                questions.append(question)
    
    return questions[:5]   # limit to 5 questi

if __name__ == "__main__":
    audio_file = "sample.ogg"   # change if needed

    transcript = transcribe_audio(audio_file)

    print("\nFull Transcript:\n")
    print(transcript)

    summary = summarize_text(transcript)

    print("\nSummary:\n")
    print(summary)

quiz_questions = generate_quiz(summary)

print("\nQuiz Questions:\n")
for i, q in enumerate(quiz_questions, 1):
    print(f"{i}. {q}")


    # Save output
with open("output_notes.txt", "w", encoding="utf-8") as f:
    f.write("TRANSCRIPT:\n")
    f.write(transcript)
    
    f.write("\n\nSUMMARY:\n")
    f.write(summary)
    
    f.write("\n\nQUIZ QUESTIONS:\n")
    
    for i, q in enumerate(quiz_questions, 1):
        f.write(f"{i}. {q}\n")

    print("\nNotes saved as output_notes.txt")


