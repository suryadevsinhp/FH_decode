# GNU Radio Frequency Hopping Decoder - Quick Start

## Simple 3-Step Process

### Step 1: Edit the Script
Open `gnuradio_decoder.py` and change these settings at the top:

```python
# Your GNU Radio .bin file name (change this to your file)
INPUT_FILE = "my_recording.bin"

# Sample rate used in GNU Radio (Hz)
SAMPLE_RATE = 2000000  # Change to your sample rate

# Center frequency used in GNU Radio (Hz)  
CENTER_FREQ = 433000000  # Change to your center frequency

# Audio demodulation type
MODULATION = "fm"  # Usually "fm" works best

# Detection sensitivity (lower = more sensitive)
THRESHOLD = 2.0  # Try 1.5 for weak signals
```

### Step 2: Run the Script
```bash
python3 gnuradio_decoder.py
```

### Step 3: Get Your Audio
The script automatically creates: `my_recording_decoded_audio.wav`

## Common Settings for GNU Radio

### Typical GNU Radio Settings:
- **File Format**: Complex64 (automatic)
- **Sample Rates**: 1 MSPS, 2 MSPS, 5 MSPS, 10 MSPS
- **Common Frequencies**: 433 MHz, 868 MHz, 915 MHz, 2.4 GHz

### Example Configurations:

**433 MHz ISM Band:**
```python
INPUT_FILE = "433mhz_recording.bin"
SAMPLE_RATE = 2000000
CENTER_FREQ = 433000000
MODULATION = "fm"
```

**915 MHz ISM Band:**
```python
INPUT_FILE = "915mhz_recording.bin" 
SAMPLE_RATE = 5000000
CENTER_FREQ = 915000000
MODULATION = "fm"
```

**2.4 GHz (WiFi/Bluetooth):**
```python
INPUT_FILE = "bluetooth_recording.bin"
SAMPLE_RATE = 10000000
CENTER_FREQ = 2440000000
MODULATION = "fm"
```

## What the Script Does Automatically

1. **Loads** your GNU Radio .bin file (complex64 format)
2. **Analyzes** the signal with advanced algorithms
3. **Detects** frequency hopping patterns automatically
4. **Tracks** frequency changes over time
5. **Demodulates** audio from each frequency hop
6. **Combines** all audio segments into one file
7. **Filters** and optimizes audio for listening
8. **Saves** as standard 44.1 kHz WAV file

## Output Information

The script shows you:
- File size and duration
- Number of frequency hops detected
- Frequency range covered
- Hop rate estimation
- Audio extraction success rate
- Final audio file details

## Troubleshooting

### No hops detected?
```python
THRESHOLD = 1.5  # Make more sensitive
```

### Audio sounds wrong?
```python
MODULATION = "am"  # Try different demodulation
```

### Wrong frequency range?
```python
CENTER_FREQ = 915000000  # Check your center frequency
```

## That's It!
Just change the filename and settings, run the script, and get your audio file automatically!