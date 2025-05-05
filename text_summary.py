from transformers import pipeline, AutoTokenizer

# Load model and tokenizer once globally
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer)

def chunk_text(text, tokenizer, max_length=512, overlap=50):
    """Splits text into tokenized chunks of max_length tokens with an overlap."""
    tokens = tokenizer(text, truncation=False, padding=False)["input_ids"]
    chunks = []

    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)

    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_text_from_file(transcript_file):
    """Summarize text read from a file."""
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript = f.read()
    return summarize_chunks(transcript)

def summarize_text_from_string(text_string):
    """Summarize text passed directly as a string."""
    return summarize_chunks(text_string)

def summarize_chunks(text):
    """Shared logic to chunk and summarize text."""
    chunks = chunk_text(text, tokenizer, max_length=512, overlap=50)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    final_summary = " ".join(summaries)
    return final_summary

# Example: Summarize from file
file_summary = summarize_text_from_file("Second.txt")
print("**File-based Meeting Summary**\n", file_summary)

with open("Second_summary.txt", "w", encoding="utf-8") as f:
    f.write("**Meeting Summary**\n\n")
    f.write(file_summary)

# Example: Summarize from text string
input_text = """
Artificial intelligence has been rapidly evolving, enabling machines to perform tasks once thought only humans could do.
Natural language processing, computer vision, and decision-making capabilities are increasingly integrated into real-world applications.
"""
string_summary = summarize_text_from_string(input_text)
print("\n**String-based Summary**\n", string_summary)
