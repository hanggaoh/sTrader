curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "skip_existing": false
  }' \
  http://localhost:8000/backfill/news/full