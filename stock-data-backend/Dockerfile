# ===== Base Stage =====
# This stage installs system dependencies and base Python packages.
FROM python:3.10-slim AS base

WORKDIR /app

# Install system dependencies required for both prod and test
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies
# Note: This assumes requirements.txt is at the root of the build context.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ===== Test Stage =====
# This stage runs the tests. It inherits from base and adds test-specific dependencies.
FROM base AS test

# Install development/test dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Set the Python path to include both the project root and the src directory
ENV PYTHONPATH=/app:/app/src

# Copy the application code
COPY . .

# Default command for this stage
CMD ["python", "-m", "pytest"]


# ===== Final/Production Stage =====
# This is the final, lean image that will be used in production.
FROM base AS final

# Install postgresql-client to get access to pg_dump for backups
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client && rm -rf /var/lib/apt/lists/*

# Copy the application code from the test stage to ensure we deploy what was tested.
COPY --from=test /app /app

# Set the final working directory to the nested src folder
WORKDIR /app/src

# Command to run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--log-level", "debug", "--workers", "1", "--threads", "10", "app:app"]
