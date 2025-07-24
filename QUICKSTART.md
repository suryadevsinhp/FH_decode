# 🚀 Quick Start Guide

Get your Frequency Hopping Decoder running in minutes!

## Prerequisites

- Ubuntu 22.04 LTS
- Internet connection for package downloads
- Sudo privileges

## Installation (5 minutes)

```bash
# 1. Run the automated setup
./setup.sh

# 2. Generate test data
./generate_test_data.py --duration 5 --output test.bin

# 3. Start the application
./start.sh
```

## Access the Application

Open your browser and go to: **http://localhost:5000**

## Test the System

1. **Upload File**: Click "Choose File" and select `test.bin`
2. **Set Parameters**:
   - Center Frequency: `433.0` MHz
   - Bandwidth: `1.0` MHz  
   - Sample Rate: `2.0` MHz
3. **Process**: Click "Process File"
4. **Watch Progress**: Real-time updates will show processing status
5. **Results**: View spectrogram and play decoded audio

## What You'll See

✅ **Upload Progress**: File upload with validation  
✅ **Processing Bar**: Real-time progress from 0-100%  
✅ **Frequency Plot**: Interactive spectrogram visualization  
✅ **Audio Player**: Decoded audio with download option  
✅ **Statistics**: Hop count and processing details  

## Quick Commands

```bash
# Check if everything is running
sudo systemctl status mongod
sudo systemctl status frequency-hopping-decoder

# View logs
sudo journalctl -u frequency-hopping-decoder -f

# Generate larger test file
./generate_test_data.py --duration 30 --hop-rate 200 --output large_test.bin

# Stop services
sudo systemctl stop frequency-hopping-decoder
sudo systemctl stop mongod
```

## Troubleshooting

**Port 5000 already in use?**
```bash
sudo lsof -i :5000
# Kill the process using the port, then restart
```

**MongoDB not starting?**
```bash
sudo systemctl start mongod
sudo systemctl enable mongod
```

**Permission errors?**
```bash
chmod 755 uploads static/outputs static/plots
```

## File Format for Your IQ Data

- **Format**: Binary (.bin)
- **Data**: Interleaved I/Q float32
- **Layout**: [I₁, Q₁, I₂, Q₂, ...]
- **Max Size**: 500MB

## Production Deployment

For production use:
```bash
./start_production.sh
```

This will:
- Run as a systemd service
- Auto-start on boot
- Enable log rotation
- Use production settings

---

🎉 **You're ready to analyze frequency hopping signals!** 📡

For detailed documentation, see [README.md](README.md)