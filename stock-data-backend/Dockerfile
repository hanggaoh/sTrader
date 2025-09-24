# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the initial working directory
WORKDIR /app

# Install system dependencies needed for binary package builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the final working directory to the src folder
# This makes all intra-application imports work correctly
WORKDIR /app/src

# Command to run the application from within the src directory
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
