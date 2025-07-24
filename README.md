# Frequency Hopping Decoder

A complete full-stack web application for analyzing and decoding frequency hopping signals from IQ recordings. Built with Flask backend, MongoDB database, WebSocket real-time communication, and a modern responsive web interface.

![Frequency Hopping Decoder](https://img.shields.io/badge/Platform-Ubuntu%2022.04-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green)
![MongoDB](https://img.shields.io/badge/MongoDB-6.0-green)

## 🚀 Features

- **File Upload**: Support for .bin IQ recording files up to 500MB
- **Real-time Processing**: Live progress tracking via WebSockets
- **Frequency Analysis**: Advanced signal processing using STFT and spectral analysis
- **Audio Decoding**: Extract and decode audio from frequency hopping patterns
- **Visualization**: Interactive spectrograms and frequency analysis plots
- **Audio Playback**: Built-in audio player with download capability
- **Job Management**: Track processing history with MongoDB
- **Responsive UI**: Modern Bootstrap-based interface
- **Production Ready**: Systemd service integration and log management

## 📋 Requirements

- Ubuntu 22.04 LTS (tested and optimized)
- Python 3.8 or higher
- MongoDB 6.0
- Minimum 4GB RAM (8GB recommended for large files)
- Audio libraries and signal processing dependencies

## 🛠️ Quick Installation

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd frequency-hopping-decoder

# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

The automated setup script will:
- Install all system dependencies
- Set up MongoDB
- Create Python virtual environment
- Install Python packages
- Configure systemd service
- Set up firewall rules
- Create necessary directories

### Manual Installation

<details>
<summary>Click to expand manual installation steps</summary>

1. **Update system packages:**
```bash
sudo apt update && sudo apt upgrade -y
```

2. **Install system dependencies:**
```bash
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential \
    pkg-config libffi-dev libssl-dev curl wget git \
    libasound2-dev libportaudio2 portaudio19-dev libsndfile1-dev \
    libfftw3-dev liblapack-dev libblas-dev gfortran
```

3. **Install MongoDB:**
```bash
# Import MongoDB GPG key
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Install MongoDB
sudo apt update
sudo apt install -y mongodb-org
sudo systemctl start mongod
sudo systemctl enable mongod
```

4. **Set up Python environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

5. **Create directories:**
```bash
mkdir -p uploads static/outputs static/plots logs
```

</details>

## 🚀 Usage

### Starting the Application

#### Development Mode
```bash
./start.sh
```

#### Production Mode
```bash
./start_production.sh
```

### Accessing the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

### Using the Application

1. **Upload IQ File**: Select a .bin file containing IQ data
2. **Configure Parameters**:
   - Center Frequency (MHz): The center frequency of your recording
   - Bandwidth (MHz): The bandwidth of the signal
   - Sample Rate (MHz): The sample rate of your IQ recording
3. **Process**: Click "Process File" to start analysis
4. **Monitor Progress**: Watch real-time progress updates
5. **View Results**: 
   - Listen to decoded audio
   - View frequency spectrogram
   - Download processed files

### Generating Test Data

Create synthetic frequency hopping signals for testing:

```bash
# Generate 5-second test signal
./generate_test_data.py --duration 5 --output test.bin

# Custom parameters
./generate_test_data.py --duration 10 --sample-rate 2000000 \
    --center-freq 433000000 --hop-rate 50 --output custom_test.bin
```

## 📁 File Format

The application expects IQ data in binary format:
- **Format**: Interleaved I/Q samples
- **Data Type**: 32-bit float (float32)
- **Layout**: [I₁, Q₁, I₂, Q₂, I₃, Q₃, ...]
- **Endianness**: Little-endian (standard)

### Example IQ File Structure
```
Offset  | Data
--------|--------
0x0000  | I₁ (float32)
0x0004  | Q₁ (float32)  
0x0008  | I₂ (float32)
0x000C  | Q₂ (float32)
...     | ...
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file to customize configuration:

```bash
# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_ENV=production

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
DATABASE_NAME=frequency_hopping_db

# File Upload Configuration
MAX_CONTENT_LENGTH=524288000  # 500MB
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=static/outputs

# Application Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=False
```

### Advanced Signal Processing Parameters

Modify these parameters in `app.py` for different signal types:

```python
# STFT Parameters
nperseg = 1024          # FFT size
noverlap = nperseg // 4 # Overlap between segments

# Audio Processing
target_sample_rate = 44100  # Output audio sample rate
```

## 🎛️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/upload` | POST | Upload and process IQ file |
| `/download/<filename>` | GET | Download processed audio |
| `/plot/<filename>` | GET | View spectrogram images |
| `/api/jobs` | GET | Get recent processing jobs |

### WebSocket Events

| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client → Server | Client connection established |
| `progress_update` | Server → Client | Processing progress updates |
| `processing_complete` | Server → Client | Processing finished |

## 🔍 Technical Details

### Signal Processing Pipeline

1. **File Reading**: Load IQ data from binary file
2. **STFT Analysis**: Compute Short-Time Fourier Transform
3. **Peak Detection**: Identify frequency peaks in each time window
4. **Hop Detection**: Track frequency changes over time
5. **Audio Extraction**: Demodulate and extract audio from each hop
6. **Visualization**: Generate spectrogram plots
7. **Output**: Save processed audio and metadata

### Frequency Hop Detection Algorithm

```python
# Simplified algorithm
for each_time_window in stft_result:
    peak_frequency = find_peak(power_spectrum[time_window])
    hop_frequencies.append(peak_frequency)
    
# Extract audio segments based on detected hops
for each_hop in hop_frequencies:
    audio_segment = demodulate(iq_data[hop_time_range])
    audio_segments.append(audio_segment)
```

### Performance Considerations

- **Memory Usage**: ~8x file size during processing
- **Processing Time**: ~1-2x file duration for analysis
- **Recommended Specs**: 
  - 8GB RAM for files up to 100MB
  - SSD storage for better I/O performance
  - Multi-core CPU beneficial for FFT operations

## 🔧 System Administration

### Service Management

```bash
# Check service status
sudo systemctl status frequency-hopping-decoder

# View real-time logs
sudo journalctl -u frequency-hopping-decoder -f

# Restart service
sudo systemctl restart frequency-hopping-decoder

# Stop service
sudo systemctl stop frequency-hopping-decoder
```

### Database Management

```bash
# Connect to MongoDB
mongosh

# List databases
show dbs

# Use application database
use frequency_hopping_db

# View processing jobs
db.processing_jobs.find().pretty()

# Clear old jobs
db.processing_jobs.deleteMany({"timestamp": {"$lt": new Date(Date.now() - 30*24*60*60*1000)}})
```

### Log Files

- **Application Logs**: `/var/log/frequency-hopping-decoder/`
- **MongoDB Logs**: `/var/log/mongodb/mongod.log`
- **System Logs**: `journalctl -u frequency-hopping-decoder`

## 🚨 Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   ```bash
   sudo systemctl start mongod
   sudo systemctl enable mongod
   ```

2. **Permission Denied on Upload**
   ```bash
   chmod 755 uploads static/outputs static/plots
   ```

3. **Audio Libraries Missing**
   ```bash
   sudo apt install -y libsndfile1-dev portaudio19-dev
   pip install soundfile librosa
   ```

4. **Large File Processing Fails**
   - Check available RAM and disk space
   - Reduce file size or increase system resources
   - Monitor with `htop` during processing

### Performance Tuning

1. **Increase MongoDB memory**:
   ```javascript
   // In MongoDB shell
   db.adminCommand({setParameter: 1, wiredTigerCacheSizeGB: 2})
   ```

2. **Optimize Python processing**:
   ```python
   # Reduce STFT size for faster processing
   nperseg = 512  # Instead of 1024
   ```

3. **Enable compression**:
   ```bash
   # Add to nginx config if using reverse proxy
   gzip_types text/css application/javascript application/json;
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SciPy**: Signal processing algorithms
- **NumPy**: Numerical computing
- **Flask-SocketIO**: Real-time communication
- **Librosa**: Audio processing
- **Bootstrap**: UI framework

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review system logs
3. Create an issue with:
   - System information
   - Error messages
   - Steps to reproduce

---

**Note**: This application is designed for educational and research purposes. Ensure you have proper authorization before analyzing radio signals in your jurisdiction.