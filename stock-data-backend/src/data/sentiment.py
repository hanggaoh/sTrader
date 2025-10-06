import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text: str) -> float:
    """
    Analyzes the sentiment of a given text using FinBERT.
    Returns a sentiment score between -1 (negative) and 1 (positive).
    """
    if not text:
        return 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    # The model returns logits for positive, negative, neutral.
    # We can convert this to a single score: (positive - negative).
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    # According to the model card, the labels are: 0: positive, 1: negative, 2: neutral
    positive_prob = probabilities[0].item()
    negative_prob = probabilities[1].item()
    
    score = positive_prob - negative_prob
    return score

def bulk_analyze_sentiment(headlines: list[tuple[int, str]]) -> list[tuple[float, int]]:
    """
    Performs sentiment analysis on a batch of headlines.
    """
    if not headlines:
        return []

    results = []
    for id, headline_text in headlines:
        score = analyze_sentiment(headline_text)
        results.append((score, id))
    return results
