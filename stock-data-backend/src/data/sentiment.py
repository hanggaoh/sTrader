import os
import logging
from typing import List, Dict, Optional

import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

log = logging.getLogger(__name__)

# --- Configuration ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Hugging Face Model Loading ---
# We load the model and tokenizer only once when the module is first imported.
# This is a time-consuming operation, so we want to avoid doing it repeatedly.
SENTIMENT_MODEL_NAME = "yiyang-zhang/finbert-pretrain-chinese"
try:
    log.info(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    log.info("Sentiment model loaded successfully.")
except Exception as e:
    log.error(f"Failed to load sentiment model. Sentiment analysis will be disabled. Error: {e}")
    sentiment_pipeline = None

# --- News Fetching ---
def fetch_news_for_symbol(symbol: str, stock_name: str) -> Optional[List[Dict]]:
    """
    Fetches news articles for a given stock symbol by searching for its name.

    Args:
        symbol: The stock symbol (e.g., '000001.SZ').
        stock_name: The name of the stock to search for (e.g., '平安银行').

    Returns:
        A list of article dictionaries, or None if the API key is missing or an error occurs.
    """
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY is not set. Cannot fetch news.")
        return None

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=\"{stock_name}\"&"
        f"language=zh&"
        f"sortBy=publishedAt&"
        f"apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        
        if data.get("status") == "ok":
            log.info(f"Successfully fetched {len(data.get('articles', []))} articles for {stock_name}.")
            return data.get('articles', [])
        else:
            log.error(f"NewsAPI returned an error: {data.get('message')}")
            return None

    except requests.exceptions.RequestException as e:
        log.error(f"Failed to fetch news for {stock_name}. Error: {e}")
        return None

# --- Sentiment Analysis ---
def analyze_sentiment(headline: str) -> Optional[float]:
    """
    Analyzes the sentiment of a single news headline.

    Args:
        headline: The news headline string.

    Returns:
        A numerical sentiment score, or None if the model is not available.
        - Positive -> 1.0
        - Negative -> -1.0
        - Neutral  -> 0.0
    """
    if not sentiment_pipeline:
        log.warning("Sentiment pipeline is not available. Cannot analyze sentiment.")
        return None
    
    if not headline:
        return 0.0

    try:
        result = sentiment_pipeline(headline)[0]
        label = result['label']
        
        if label == 'positive':
            return 1.0
        elif label == 'negative':
            return -1.0
        else: # neutral
            return 0.0
            
    except Exception as e:
        log.error(f"Error during sentiment analysis for headline '{headline}'. Error: {e}")
        return None
