#!/bin/bash
# Docker Quick Start Script
# Sets up and runs the SingN'Seek application with Docker
# Source code is located in src/ directory

set -e

echo "========================================="
echo "SingN'Seek Docker Setup & Run"
echo "========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo ""
    echo "ERROR: .env file not found!"
    echo ""
    echo "Please create .env file from template:"
    echo "  cp .env.example .env"
    echo ""
    echo "Then edit .env with your credentials:"
    echo "  - ELASTICSEARCH_URL"
    echo "  - ELASTICSEARCH_API_KEY"
    echo "  - GOOGLE_CLOUD_PROJECT"
    echo "  - GOOGLE_CLOUD_REGION"
    echo ""
    exit 1
fi

echo ""
echo "✓ .env file found"
echo ""

# Check if src directory exists
if [ ! -d src ]; then
    echo "ERROR: src/ directory not found!"
    echo "Source code should be in src/ directory"
    exit 1
fi

echo "✓ Source code directory (src/) found"
echo ""

# Validate required environment variables
required_vars=("ELASTICSEARCH_URL" "ELASTICSEARCH_API_KEY" "GOOGLE_CLOUD_PROJECT")

missing_vars=()
for var in "${required_vars[@]}"; do
    value=$(grep "^${var}=" .env | cut -d= -f2-)
    if [ -z "$value" ] || [ "$value" == "your-"* ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "ERROR: Missing or invalid required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please update .env with actual values"
    exit 1
fi

echo "✓ Required environment variables configured"
echo ""

# Build the Docker image
echo "Building Docker image..."
docker-compose build

echo ""
echo "✓ Docker image built successfully"
echo ""

# Start the containers
echo "Starting application..."
docker-compose up -d

echo ""
echo "✓ Application started!"
echo ""

# Wait for application to be ready
echo "Waiting for application to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if docker exec singnseek-app curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo ""
        echo "========================================="
        echo "SUCCESS! Application is ready"
        echo "========================================="
        echo ""
        echo "Access the application at:"
        echo "  http://localhost:8501"
        echo ""
        echo "Source code is in the src/ directory:"
        echo "  src/main.py (entry point)"
        echo "  src/utils/ (utilities, logging)"
        echo "  src/config/ (configuration files)"
        echo "  src/converters/ (audio conversion tools)"
        echo ""
        echo "View logs with:"
        echo "  docker-compose logs -f singnseek"
        echo ""
        echo "Stop the application with:"
        echo "  docker-compose down"
        echo ""
        exit 0
    fi
    
    attempt=$((attempt + 1))
    sleep 2
done

echo ""
echo "WARNING: Application startup check timed out"
echo "Check logs with: docker-compose logs singnseek"
echo ""
