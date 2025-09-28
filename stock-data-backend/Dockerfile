# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the initial working directory
WORKDIR /app

# Install system dependencies, including curl for the health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the final working directory to the src folder
WORKDIR /app/src

# Command to run the application with Gunicorn using a threaded worker
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--log-level", "debug", "--workers", "1", "--threads", "10", "app:app"]
