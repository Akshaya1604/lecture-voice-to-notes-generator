import streamlit as st
import whisper
import os
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# -------------------- PAGE STYLE --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #e3f2fd, #ffffff);
}
h1 {
    color: #0d47a1;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI Lecture Voice-to-Notes Generator")

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return whisper_model, summarizer

whisper_model, summarizer = load_models()

# -------------------- FUNCTIONS --------------------
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]

# FAST QUIZ GENERATION (No heavy AI model)
def generate_quiz(summary):
    sentences = summary.split(".")
    questions = []
    for s in sentences:
        s = s.strip()
        if len(s) > 20:
            questions.append("Explain: " + s + "?")
        if len(questions) == 5:
            break
    return questions

def create_pdf(transcript, summary, quiz_questions):
    pdf_file = "lecture_notes.pdf"
    doc = SimpleDocTemplate(pdf_file)
    elements = []

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    elements.append(Paragraph("<b>TRANSCRIPT</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(transcript, normal_style))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("<b>SUMMARY</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(summary, normal_style))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph("<b>QUIZ QUESTIONS</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    quiz_list = [ListItem(Paragraph(q, normal_style)) for q in quiz_questions]
    elements.append(ListFlowable(quiz_list, bulletType="1"))

    doc.build(elements)
    return pdf_file

# -------------------- UI --------------------
uploaded_file = st.file_uploader("Upload Lecture Audio File", type=["mp3", "wav", "ogg"])

if uploaded_file is not None:
    with open("temp_audio.ogg", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing audio... Please wait"):
        transcript = transcribe_audio("temp_audio.ogg")
        summary = summarize_text(transcript)
        quiz_questions = generate_quiz(summary)

    st.subheader("📝 Transcript")
    st.write(transcript)

    st.subheader("📌 Summary")
    st.write(summary)

    st.subheader("❓ Quiz Questions")
    for i, q in enumerate(quiz_questions, 1):
        st.write(f"{i}. {q}")

    pdf_file = create_pdf(transcript, summary, quiz_questions)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download Notes as PDF",
            data=f,
            file_name="lecture_notes.pdf",
            mime="application/pdf"
        )