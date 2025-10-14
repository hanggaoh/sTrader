#!/bin/bash

# This script triggers a targeted news backfill for a single stock symbol
# for a specified date range.

# --- Configuration ---
HOST="http://localhost:8000"
ENDPOINT="/backfill/news"
SYMBOL="002594.SZ"

# Calculate the start date (6 months ago) and end date (today)
START_DATE=$(date -v-6m +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)

echo "--- Starting Single-Symbol News Backfill ---"
echo "Requesting backfill for symbol: $SYMBOL"
echo "Date range: $START_DATE to $END_DATE"

# --- Main Logic ---
curl -X POST \
  -H "Content-Type: application/json" \
  -d "{
        \"symbol\": \"$SYMBOL\",
        \"start_date\": \"$START_DATE\",
        \"end_date\": \"$END_DATE\"
      }" \
  "$HOST$ENDPOINT"

echo "\n--- Backfill Request Sent ---"
