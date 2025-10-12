#!/usr/bin/env python
# This script downloads the specified Hugging Face model and tokenizer to a local directory.

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_NAME = "speedxd/finetuned-finbert-chinese-v1"
# Define the local directory where the model will be saved
SAVE_DIRECTORY = os.path.join(os.path.dirname(__file__), "../ml/models/finetuned-finbert-chinese-v1")


def main():
    """Downloads and saves the model and tokenizer."""
    print(f"--- Starting download of model: {MODEL_NAME} ---")

    # 1. Create the save directory if it doesn't exist
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
        print(f"Created directory: {SAVE_DIRECTORY}")

    # 2. Download and save the tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIRECTORY)
    print("Tokenizer saved successfully.")

    # 3. Download and save the model
    print("Downloading model... (This may take a few minutes)")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(SAVE_DIRECTORY)
    print("Model saved successfully.")

    print(f"--- Model and tokenizer saved to: {SAVE_DIRECTORY} ---")


if __name__ == "__main__":
    main()
