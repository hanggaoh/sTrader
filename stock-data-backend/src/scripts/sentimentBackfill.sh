#!/bin/bash

# This script repeatedly calls the sentiment analysis backfill endpoint
# in batches until all pending articles have been processed.

# --- Configuration ---
HOST="http://localhost:8000"
ENDPOINT="/sentiment/backfill"
BATCH_SIZE=5000 # Number of articles to process per API call

echo "--- Starting Sentiment Analysis Batch Backfill ---"
echo "Processing articles in batches of $BATCH_SIZE..."

while true; do
    echo "-----------------------------------------------------"
    echo "Requesting a new batch of $BATCH_SIZE articles for analysis..."

    # Make the API call and capture the response
    response=$(curl -s -X POST \
      -H "Content-Type: application/json" \
      -d "{\"limit\": $BATCH_SIZE}" \
      "$HOST$ENDPOINT")

    # Check if the response indicates that no more articles are pending
    if echo "$response" | grep -q "No pending articles to analyze"; then
        echo "✅ All pending articles have been processed."
        break
    fi

    # Log the response from the server
    echo "Server response: $response"
    echo "Batch processed. Waiting 10 seconds before the next batch..."
    sleep 10
done

echo "--- Sentiment Analysis Batch Backfill Complete ---"
