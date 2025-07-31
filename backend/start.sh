#!/bin/bash

# Set default port if not provided
if [ -z "$PORT" ]; then
    export PORT=8000
fi

echo "Starting uvicorn on port $PORT"

# Start the application
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1