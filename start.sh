#!/bin/bash

# Frequency Hopping Decoder - Simple Start Script
echo "🚀 Starting Frequency Hopping Decoder..."

# Check if MongoDB is running
if ! pgrep -x "mongod" > /dev/null; then
    echo "⚠️  Starting MongoDB..."
    sudo systemctl start mongod || sudo service mongodb start || echo "❌ MongoDB failed to start - continuing anyway"
fi

# Install dependencies if needed
echo "📦 Installing Python dependencies..."
pip3 install --break-system-packages -r requirements.txt

# Create necessary directories
mkdir -p uploads static/outputs static/plots

# Start the application
echo "🌐 Starting Flask application..."
echo "🔗 Access at: http://localhost:5000"
python3 app.py