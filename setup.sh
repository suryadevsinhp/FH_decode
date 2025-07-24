#!/bin/bash

# Frequency Hopping Decoder Setup Script for Ubuntu 22.04
# This script installs all required dependencies and sets up the application

set -e  # Exit on any error

echo "🔧 Setting up Frequency Hopping Decoder on Ubuntu 22.04..."
echo "============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons."
   exit 1
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    curl \
    wget \
    git \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install audio and signal processing libraries
print_status "Installing audio and signal processing libraries..."
sudo apt install -y \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libfftw3-dev \
    liblapack-dev \
    libblas-dev \
    gfortran

# Install MongoDB
print_status "Installing MongoDB..."
if ! command -v mongod &> /dev/null; then
    # Import MongoDB GPG key
    curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
    
    # Add MongoDB repository
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
    
    # Update and install MongoDB
    sudo apt update
    sudo apt install -y mongodb-org
    
    # Start and enable MongoDB
    sudo systemctl start mongod
    sudo systemctl enable mongod
    
    print_success "MongoDB installed and started"
else
    print_success "MongoDB already installed"
fi

# Create Python virtual environment
print_status "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment and install Python dependencies
print_status "Installing Python dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

print_success "Python dependencies installed"

# Create necessary directories
print_status "Creating application directories..."
mkdir -p uploads static/outputs static/plots logs

# Set proper permissions
chmod 755 uploads static/outputs static/plots logs

# Create systemd service file
print_status "Creating systemd service file..."
sudo tee /etc/systemd/system/frequency-hopping-decoder.service > /dev/null <<EOF
[Unit]
Description=Frequency Hopping Decoder Web Application
After=network.target mongodb.service
Requires=mongodb.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python app.py
Restart=always
RestartSec=10

# Output to syslog
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=frequency-hopping-decoder

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

print_success "Systemd service created"

# Create environment file
print_status "Creating environment configuration..."
cat > .env <<EOF
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=frequency_hopping_db

# File Upload Configuration
MAX_CONTENT_LENGTH=524288000  # 500MB in bytes
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=static/outputs

# Application Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=False
EOF

print_success "Environment configuration created"

# Create startup script
print_status "Creating startup script..."
cat > start.sh <<EOF
#!/bin/bash
# Frequency Hopping Decoder Startup Script

# Activate virtual environment
source venv/bin/activate

# Check if MongoDB is running
if ! pgrep -x "mongod" > /dev/null; then
    echo "Starting MongoDB..."
    sudo systemctl start mongod
fi

# Start the application
echo "Starting Frequency Hopping Decoder..."
python app.py
EOF

chmod +x start.sh

# Create production startup script
cat > start_production.sh <<EOF
#!/bin/bash
# Production startup script using systemd

echo "Starting Frequency Hopping Decoder in production mode..."

# Start MongoDB if not running
sudo systemctl start mongod

# Start the application service
sudo systemctl start frequency-hopping-decoder

# Enable auto-start on boot
sudo systemctl enable frequency-hopping-decoder

echo "Service started. Check status with: sudo systemctl status frequency-hopping-decoder"
echo "View logs with: sudo journalctl -u frequency-hopping-decoder -f"
EOF

chmod +x start_production.sh

# Create log rotation configuration
print_status "Setting up log rotation..."
sudo tee /etc/logrotate.d/frequency-hopping-decoder > /dev/null <<EOF
/var/log/frequency-hopping-decoder/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload frequency-hopping-decoder
    endscript
}
EOF

# Create log directory
sudo mkdir -p /var/log/frequency-hopping-decoder
sudo chown $USER:$USER /var/log/frequency-hopping-decoder

# Firewall configuration
print_status "Configuring firewall..."
if command -v ufw &> /dev/null; then
    sudo ufw allow 5000/tcp
    print_success "Firewall configured to allow port 5000"
else
    print_warning "UFW not found. Please manually configure firewall to allow port 5000"
fi

# Create test data generator script
print_status "Creating test data generator..."
cat > generate_test_data.py <<EOF
#!/usr/bin/env python3
"""
Generate test IQ data for frequency hopping decoder testing
"""

import numpy as np
import struct
import argparse

def generate_frequency_hopping_signal(duration=10, sample_rate=2e6, 
                                    center_freq=433e6, hop_rate=100, 
                                    frequencies=None):
    """Generate a synthetic frequency hopping signal"""
    
    if frequencies is None:
        # Default frequency hops around center frequency
        frequencies = [
            center_freq - 200e3,
            center_freq - 100e3,
            center_freq,
            center_freq + 100e3,
            center_freq + 200e3
        ]
    
    num_samples = int(duration * sample_rate)
    hop_duration = 1.0 / hop_rate  # seconds per hop
    samples_per_hop = int(hop_duration * sample_rate)
    
    t = np.arange(num_samples) / sample_rate
    signal = np.zeros(num_samples, dtype=complex)
    
    for i in range(num_samples):
        hop_index = int(i / samples_per_hop) % len(frequencies)
        freq = frequencies[hop_index]
        
        # Generate complex sinusoid at hop frequency
        phase = 2 * np.pi * (freq - center_freq) * t[i]
        signal[i] = np.exp(1j * phase)
    
    # Add some noise
    noise_power = 0.1
    noise = noise_power * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    signal += noise
    
    return signal

def save_iq_data(signal, filename):
    """Save IQ data to binary file"""
    # Interleave I and Q components
    iq_data = np.zeros(len(signal) * 2, dtype=np.float32)
    iq_data[0::2] = np.real(signal).astype(np.float32)
    iq_data[1::2] = np.imag(signal).astype(np.float32)
    
    with open(filename, 'wb') as f:
        f.write(iq_data.tobytes())
    
    print(f"Generated {len(signal)} IQ samples saved to {filename}")
    print(f"File size: {len(iq_data) * 4} bytes ({len(iq_data) * 4 / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test IQ data for frequency hopping decoder")
    parser.add_argument("--duration", type=float, default=10, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=float, default=2e6, help="Sample rate in Hz")
    parser.add_argument("--center-freq", type=float, default=433e6, help="Center frequency in Hz")
    parser.add_argument("--hop-rate", type=float, default=100, help="Hop rate in Hz")
    parser.add_argument("--output", type=str, default="test_signal.bin", help="Output filename")
    
    args = parser.parse_args()
    
    print(f"Generating test signal...")
    print(f"Duration: {args.duration}s")
    print(f"Sample rate: {args.sample_rate/1e6:.1f} MHz")
    print(f"Center frequency: {args.center_freq/1e6:.1f} MHz")
    print(f"Hop rate: {args.hop_rate} Hz")
    
    signal = generate_frequency_hopping_signal(
        duration=args.duration,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        hop_rate=args.hop_rate
    )
    
    save_iq_data(signal, args.output)
EOF

chmod +x generate_test_data.py

# Final setup steps
print_status "Performing final setup steps..."

# Test MongoDB connection
if sudo systemctl is-active --quiet mongod; then
    print_success "MongoDB is running"
else
    print_warning "MongoDB is not running. Starting it now..."
    sudo systemctl start mongod
fi

# Test Python environment
source venv/bin/activate
python3 -c "import flask, pymongo, numpy, scipy, librosa; print('All Python dependencies are working')"

print_success "Setup completed successfully!"
echo ""
echo "============================================================="
echo -e "${GREEN}🎉 Frequency Hopping Decoder Setup Complete!${NC}"
echo "============================================================="
echo ""
echo "Next steps:"
echo "1. Generate test data: ./generate_test_data.py --duration 5 --output test.bin"
echo "2. Start the application:"
echo "   Development: ./start.sh"
echo "   Production: ./start_production.sh"
echo ""
echo "3. Open your browser to: http://localhost:5000"
echo ""
echo "Useful commands:"
echo "- Check service status: sudo systemctl status frequency-hopping-decoder"
echo "- View logs: sudo journalctl -u frequency-hopping-decoder -f"
echo "- Stop service: sudo systemctl stop frequency-hopping-decoder"
echo ""
echo "The application supports .bin files with interleaved I/Q float32 data."
echo "Default parameters: Center Freq=433MHz, Bandwidth=1MHz, Sample Rate=2MHz"
echo ""
print_success "Happy frequency hopping analysis! 📡"