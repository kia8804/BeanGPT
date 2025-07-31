# Use Python 3.11 slim image to ensure compatibility
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will set this dynamically)
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "echo 'Starting BeanGPT Backend...' && echo 'PORT is:' $PORT && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]