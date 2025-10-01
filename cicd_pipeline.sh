#!/bin/bash
# A simple CI/CD pipeline script to test, build, and deploy the backend.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Run Test Suite ---
echo "--- Running Test Suite ---"

# Build the test image. Docker will use the cache for layers that haven't changed.
docker compose build backend-test

# Run the backend-test service. The --rm flag removes the container after the run.
# If pytest fails, this command will exit with a non-zero status, and the script will stop here.
docker compose run --rm backend-test


# --- 2. Rebuild and Deploy ---
echo "--- Tests Passed! Rebuilding and Deploying Backend Service ---"

# If tests passed, build the new production image and then recreate the backend service.
# Caching will be used, so only changed layers (like new source code) are rebuilt.
docker compose build backend
docker compose up -d backend

echo "--- Deployment Successful! ---"
