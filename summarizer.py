from transformers import pipeline
import spacy

# Tải mô hình spaCy
nlp = spacy.load("en_core_web_sm")

# Tải mô hình tóm tắt của Hugging Face
summarizer = pipeline("summarization")

def preprocess(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def summarize(text):
    max_input_length = 1024
    sentences = preprocess(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_input_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    summaries = [summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return ' '.join(summaries)
