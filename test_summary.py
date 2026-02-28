from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
Artificial Intelligence is transforming the world. It is used in healthcare,
education, business, and many other fields. AI helps in automation,
decision making, and data analysis.
"""

summary = summarizer(text, max_length=50, min_length=20, do_sample=False)

print(summary[0]['summary_text'])
