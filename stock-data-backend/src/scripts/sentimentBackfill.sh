#!/bin/bash

# This script triggers the sentiment analysis backfill endpoint.

# The correct URL is now /sentiment/backfill.

# To run a full backfill for all pending articles:
curl -X POST -H "Content-Type: application/json" http://localhost:8000/sentiment/backfill

# To run a limited backfill (e.g., for 100 articles), uncomment the following line:
# curl -X POST -H "Content-Type: application/json" -d '{"limit": 100}' http://localhost:5000/sentiment/backfill
