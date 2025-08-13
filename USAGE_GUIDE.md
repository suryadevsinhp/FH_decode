# How to Convert .bin IQ Files to Audio

This guide shows you how to use your existing .bin IQ recording file to extract audio from frequency hopping signals.

## Quick Start

### Basic Usage
```bash
python3 decode_bin_to_audio.py your_recording.bin
```

This will:
- Load your .bin file as interleaved float32 format (most common)
- Assume 2 MSPS sample rate and 433 MHz center frequency
- Detect frequency hops automatically
- Extract audio using FM demodulation
- Save as `your_recording_decoded_audio.wav`

## Common Usage Examples

### 1. RTL-SDR Recording (most common)
```bash
python3 decode_bin_to_audio.py recording.bin \
    --sample-rate 2400000 \
    --center-freq 433000000 \
    --format interleaved_float32 \
    --modulation fm
```

### 2. HackRF Recording
```bash
python3 decode_bin_to_audio.py hackrf_signal.bin \
    --sample-rate 10000000 \
    --center-freq 915000000 \
    --format interleaved_float32 \
    --modulation fm
```

### 3. SDRplay or Similar (complex64 format)
```bash
python3 decode_bin_to_audio.py sdrplay.bin \
    --sample-rate 5000000 \
    --center-freq 400000000 \
    --format complex64 \
    --modulation am
```

### 4. GNU Radio Recording (complex64)
```bash
python3 decode_bin_to_audio.py gnuradio.bin \
    --format complex64 \
    --sample-rate 1000000 \
    --center-freq 868000000
```

### 5. RTL-SDR Raw Format (8-bit unsigned)
```bash
python3 decode_bin_to_audio.py rtlsdr_raw.bin \
    --format uint8_interleaved \
    --sample-rate 2048000 \
    --center-freq 434000000
```

## Parameters Explained

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `input_file` | Your .bin IQ file | Required | `recording.bin` |
| `--sample-rate` | IQ sample rate in Hz | 2000000 | `2400000` |
| `--center-freq` | Center frequency in Hz | 433000000 | `915000000` |
| `--format` | File format | `interleaved_float32` | `complex64` |
| `--modulation` | Audio demodulation | `fm` | `am`, `usb` |
| `--output` | Output audio file | Auto-generated | `audio.wav` |
| `--threshold-factor` | Detection sensitivity | 2.5 | `1.8` (more sensitive) |

## File Formats Supported

- **`interleaved_float32`** - I,Q,I,Q... as 32-bit floats (most common)
- **`complex64`** - Native complex64 format (GNU Radio, SDR++)
- **`complex128`** - Double precision complex
- **`int16_interleaved`** - 16-bit signed integers (some SDR software)
- **`uint8_interleaved`** - 8-bit unsigned (RTL-SDR raw)

## Modulation Types

- **`fm`** - Frequency Modulation (good for voice, most common)
- **`am`** - Amplitude Modulation 
- **`usb`** - Upper Sideband (SSB voice)
- **`lsb`** - Lower Sideband (SSB voice)
- **`fsk`** - Frequency Shift Keying (digital)
- **`ask`** - Amplitude Shift Keying (digital)

## Troubleshooting

### No hops detected?
Try lowering the threshold:
```bash
python3 decode_bin_to_audio.py recording.bin --threshold-factor 1.5
```

### Wrong frequency range?
Check your center frequency:
```bash
python3 decode_bin_to_audio.py recording.bin --center-freq 915000000
```

### Audio sounds distorted?
Try different modulation:
```bash
python3 decode_bin_to_audio.py recording.bin --modulation am
```

### File format error?
Most common formats:
- GNU Radio: `--format complex64`
- SDR#: `--format interleaved_float32`
- RTL-SDR: `--format uint8_interleaved`

## Complete Example

```bash
# Analyze a 2.4 GHz ISM band recording from RTL-SDR
python3 decode_bin_to_audio.py bluetooth_capture.bin \
    --sample-rate 2400000 \
    --center-freq 2440000000 \
    --format interleaved_float32 \
    --modulation fm \
    --threshold-factor 2.0 \
    --output bluetooth_audio.wav \
    --verbose
```

This will:
1. Load `bluetooth_capture.bin` as interleaved float32
2. Analyze 2.4 MSPS IQ data centered at 2.44 GHz
3. Detect frequency hops with threshold factor 2.0
4. Demodulate using FM
5. Save audio as `bluetooth_audio.wav`
6. Show detailed progress information

## Tips

1. **Start with default settings** and adjust if needed
2. **Lower threshold-factor** (1.5-2.0) for weak signals
3. **Higher threshold-factor** (3.0-4.0) for noisy environments
4. **Try different modulation types** if audio quality is poor
5. **Check center frequency** matches your recording frequency
6. **Verify sample rate** matches your SDR settings

The output will be a standard WAV file that you can play with any audio player!