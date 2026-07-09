# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Create a non-privileged user to run the app
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Programmatically filter out dev dependencies from requirements.txt to keep image slim
RUN grep -vE "pytest|black|flake8|isort|ipykernel" requirements.txt > prod-requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r prod-requirements.txt

# Copy source code, model, server, and client directories
COPY src/ /app/src/
COPY models/ /app/models/
COPY server/ /app/server/
COPY client/ /app/client/

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-privileged user
USER appuser

# Expose port
EXPOSE 8000

# Define self-contained healthcheck using Python's standard library to avoid installing curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the application using Uvicorn
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
