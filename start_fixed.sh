#!/bin/bash
# Fixed Frequency Hopping Decoder Startup Script

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Frequency Hopping Decoder...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip and install/update dependencies
echo -e "${GREEN}Installing/updating dependencies...${NC}"
pip install --upgrade pip

# Install dependencies with more specific versions that work well together
pip install Flask==2.3.3
pip install "Flask-SocketIO>=5.3.0,<6.0.0"
pip install "python-socketio>=5.8.0,<6.0.0"
pip install pymongo==4.5.0
pip install numpy==1.24.3
pip install scipy==1.11.3
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install matplotlib==3.7.2
pip install "Werkzeug>=2.3.0,<3.0.0"
pip install python-dotenv==1.0.0
pip install requests==2.31.0

# Check if MongoDB is running
echo -e "${GREEN}Checking MongoDB status...${NC}"
if ! pgrep -x "mongod" > /dev/null; then
    echo -e "${YELLOW}Starting MongoDB...${NC}"
    sudo systemctl start mongod
    sleep 2
fi

if ! sudo systemctl is-active --quiet mongod; then
    echo -e "${RED}Failed to start MongoDB. Please check the service.${NC}"
    exit 1
fi

echo -e "${GREEN}MongoDB is running${NC}"

# Create necessary directories
mkdir -p uploads static/outputs static/plots logs

# Set proper permissions
chmod 755 uploads static/outputs static/plots logs

# Start the application
echo -e "${GREEN}Starting Frequency Hopping Decoder application...${NC}"
echo "Access the application at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

export FLASK_ENV=development
export PYTHONPATH=$PWD:$PYTHONPATH

python app.py