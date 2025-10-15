import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional
import threading

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../ml/models/finetuned-finbert-chinese-v1")

# Initialize model and tokenizer as None for lazy loading
tokenizer = None
model = None
model_lock = threading.Lock()

def _load_model():
    """Loads the model and tokenizer thread-safely."""
    global tokenizer, model
    with model_lock:
        if tokenizer is None or model is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def analyze_sentiment(title: str, content: Optional[str]) -> float:
    """
    Analyzes the sentiment of a given text using the speedxd/finetuned-finbert-chinese-v1 model.
    Returns a sentiment score between -1 (negative) and 1 (positive).
    """
    _load_model()
    # Combine title and content for a more comprehensive analysis
    full_text = f"{title}. {content}" if content else title

    if not full_text.strip():
        return 0.0

    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    # The model returns logits for [positive, neutral, negative].
    # We can convert this to a single score: (positive - negative).
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    positive_prob = probabilities[0].item()
    negative_prob = probabilities[2].item()
    # neutral_prob = probabilities[1].item() # This is available if needed
    
    score = positive_prob - negative_prob
    return score

def bulk_analyze_sentiment(articles: list[tuple[int, str, Optional[str]]]) -> list[tuple[float, int]]:
    """
    Performs sentiment analysis on a batch of articles (id, headline, content).
    """
    if not articles:
        return []

    results = []
    for article_id, headline, content in articles:
        score = analyze_sentiment(headline, content)
        results.append((score, article_id))
    return results