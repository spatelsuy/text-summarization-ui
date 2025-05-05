from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # Max file size: 10MB

# Load summarizer and tokenizer once
MODEL_NAME = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=MODEL_NAME, tokenizer=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def chunk_text(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_chunks(text):
    chunks = chunk_text(text, tokenizer)
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

# ✅ Route 1: POST /summarize - Accepts raw text
@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400
    try:
        summary = summarize_chunks(data["text"])
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Route 2: POST /upload - Accepts file upload
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        text = file.read().decode("utf-8")
        summary = summarize_chunks(text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🏁 Optional: Health check
@app.route("/")
def home():
    return "Summarization API is running."

if __name__ == "__main__":
    app.run(debug=True)
