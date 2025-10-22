#!/bin/bash

# Setup script for SingN'Seek
# This script helps you create your .env file

set -e

echo "=============================================="
echo "  SingN'Seek Configuration Setup"
echo "=============================================="
echo ""

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Copy .env.example to .env
echo "📝 Creating .env file from template..."
cp .env.example .env
echo "✅ .env file created!"
echo ""

echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "1. Edit the .env file with your settings:"
echo "   nano .env"
echo ""
echo "2. Update these required values:"
echo "   - ELASTICSEARCH_PASSWORD"
echo ""
echo "3. Optional: Configure Vertex AI for semantic search"
echo "   - GOOGLE_CLOUD_PROJECT"
echo "   - GOOGLE_CLOUD_REGION"
echo "   - GOOGLE_APPLICATION_CREDENTIALS"
echo ""
echo "4. Start the application:"
echo "   streamlit run main.py"
echo ""
echo "For detailed configuration help, see:"
echo "   📖 CONFIGURATION.md"
echo "   🚀 QUICKSTART.md"
echo ""
