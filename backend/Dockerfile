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

# Copy start script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# Expose port (Railway will set this dynamically)
EXPOSE 8000

# Command to run the application
CMD ["./start.sh"]