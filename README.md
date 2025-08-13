# Frequency Hopping Signal Decoder

A comprehensive Python3 library for detecting, synchronizing, and decoding frequency hopping spread spectrum (FHSS) signals from IQ data. This library provides advanced signal processing capabilities for analyzing signals from Software Defined Radio (SDR) devices or recorded IQ files.

## Features

- **Multi-format IQ Data Support**: Load data from various formats (complex64, complex128, interleaved float32)
- **Advanced Hop Detection**: Multiple algorithms including threshold-based and clustering methods
- **Modulation Support**: FSK, PSK, and ASK demodulation capabilities
- **Pattern Synchronization**: Auto-detect repeating patterns or sync with known sequences
- **Real-time Processing**: Optimized algorithms with numba acceleration
- **Comprehensive Visualization**: Spectrograms, hop sequences, and advanced 3D plots
- **Data Export**: Save analysis results in HDF5 format

## Installation

### Prerequisites

```bash
# Install Python 3.8 or higher
python3 --version

# Install required packages
pip install -r requirements.txt
```

### Dependencies

The following packages are required:

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pyFFTW>=0.13.0
scikit-learn>=1.0.0
numba>=0.56.0
h5py>=3.6.0
tqdm>=4.62.0
```

## Quick Start

### Basic Usage

```python
from frequency_hopping_decoder import FrequencyHoppingDecoder

# Create decoder instance
decoder = FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)

# Load IQ data from file
decoder.load_iq_file("signal.iq", file_format="complex64")

# Or load from numpy array
# decoder.load_iq_data(iq_samples)

# Detect frequency hops
hops = decoder.detect_frequency_hops(method='threshold', threshold_factor=2.5)

# Decode the hops
decoded_data = decoder.decode_all_hops(modulation='fsk')

# Visualize results
decoder.plot_spectrogram()
```

### Generate Test Signals

```python
from frequency_hopping_decoder import generate_test_fhss_signal

# Generate a test FHSS signal
test_signal, hop_frequencies = generate_test_fhss_signal(
    sample_rate=2e6,
    duration=0.1,        # 100ms
    hop_rate=1000,       # 1kHz hop rate
    num_frequencies=10,  # 10 different frequencies
    modulation='fsk'     # FSK modulation
)
```

## Advanced Usage

### Pattern Synchronization

```python
# Auto-detect repeating patterns
sync_info = decoder.synchronize_hops()
print(f"Pattern length: {sync_info['pattern_length']}")
print(f"Confidence: {sync_info['confidence']:.2f}")

# Synchronize with known pattern
known_frequencies = [915.1e6, 915.2e6, 915.3e6, 915.4e6]  # MHz
sync_info = decoder.synchronize_hops(reference_pattern=known_frequencies)
```

### Multiple Detection Methods

```python
# Threshold-based detection (fast, good for strong signals)
hops_threshold = decoder.detect_frequency_hops(
    method='threshold',
    threshold_factor=3.0,
    min_duration=1e-3  # 1ms minimum hop duration
)

# Clustering-based detection (better for weak signals)
hops_clustering = decoder.detect_frequency_hops(
    method='clustering',
    n_hops=50
)
```

### Different Modulation Schemes

```python
# FSK (Frequency Shift Keying)
fsk_data = decoder.decode_all_hops(modulation='fsk')

# PSK (Phase Shift Keying)
psk_data = decoder.decode_all_hops(modulation='psk')

# ASK (Amplitude Shift Keying)
ask_data = decoder.decode_all_hops(modulation='ask')
```

## Command Line Interface

Use the demonstration script for quick analysis:

```bash
# Analyze an IQ file
python demo_fhss_decoder.py --file signal.iq --format complex64

# Generate and analyze test signal
python demo_fhss_decoder.py --test --modulation fsk --duration 0.1

# Run all demonstrations with visualizations
python demo_fhss_decoder.py --all-demos --visualize

# Process file with known hop pattern
python demo_fhss_decoder.py --file signal.iq --sync-pattern "915.1e6,915.2e6,915.3e6"
```

### CLI Options

- `--file`: Path to IQ data file
- `--format`: File format (complex64, complex128, interleaved_float32)
- `--test`: Generate and analyze test signal
- `--modulation`: Modulation scheme (fsk, psk, ask)
- `--duration`: Test signal duration in seconds
- `--sample-rate`: Sample rate in Hz
- `--center-freq`: Center frequency in Hz
- `--visualize`: Show advanced visualizations
- `--all-demos`: Run all demonstration scenarios

## API Reference

### FrequencyHoppingDecoder Class

#### Constructor
```python
FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)
```
- `sample_rate`: Sample rate of IQ data in Hz
- `center_freq`: Center frequency in Hz

#### Methods

##### Data Loading
```python
load_iq_data(data)  # Load from numpy array
load_iq_file(filename, file_format='complex64')  # Load from file
```

##### Signal Analysis
```python
compute_spectrogram(nperseg=None)  # Compute STFT spectrogram
detect_frequency_hops(method='threshold', **kwargs)  # Detect hops
synchronize_hops(reference_pattern=None)  # Synchronize timing
```

##### Demodulation
```python
demodulate_hop(frequency, start_time, duration, modulation='fsk')  # Single hop
decode_all_hops(modulation='fsk')  # All detected hops
```

##### Visualization
```python
plot_spectrogram(figsize=(12, 8))  # Plot spectrogram with hops
```

##### Data Export
```python
save_results(filename)  # Save to HDF5 format
```

### Detection Methods

#### Threshold-based Detection
Best for: Strong signals, fast processing
```python
hops = decoder.detect_frequency_hops(
    method='threshold',
    threshold_factor=3.0,  # Signal strength threshold
    min_duration=1e-3      # Minimum hop duration
)
```

#### Clustering-based Detection
Best for: Weak signals, complex patterns
```python
hops = decoder.detect_frequency_hops(
    method='clustering',
    n_hops=50  # Expected number of hops
)
```

### Modulation Schemes

#### FSK (Frequency Shift Keying)
- Uses instantaneous frequency discrimination
- Good for frequency-stable signals
- Robust to phase noise

#### PSK (Phase Shift Keying)
- Uses phase difference detection
- Suitable for BPSK and DPSK signals
- Sensitive to phase coherence

#### ASK (Amplitude Shift Keying)
- Uses amplitude detection with smoothing
- Simple but sensitive to amplitude variations
- Good for OOK (On-Off Keying) signals

## File Formats

### Supported IQ Formats

1. **complex64**: Native numpy complex64 format
   ```python
   decoder.load_iq_file("signal.iq", file_format="complex64")
   ```

2. **complex128**: Double precision complex format
   ```python
   decoder.load_iq_file("signal.iq", file_format="complex128")
   ```

3. **interleaved_float32**: Interleaved I/Q float32 format
   ```python
   decoder.load_iq_file("signal.iq", file_format="interleaved_float32")
   ```

### Output Formats

Results can be saved in HDF5 format containing:
- Original IQ data
- Spectrogram data
- Detected hop frequencies and timing
- Analysis parameters

## Performance Optimization

### Memory Efficiency
- Use `complex64` instead of `complex128` for large files
- Process data in chunks for very large datasets
- Adjust FFT size based on available memory

### Processing Speed
- Numba JIT compilation accelerates critical functions
- Use threshold-based detection for fastest processing
- Reduce spectrogram resolution for faster analysis

### Accuracy Tuning
- Adjust `threshold_factor` based on signal strength
- Use clustering method for difficult signals
- Increase FFT size for better frequency resolution

## Examples

### Example 1: Basic SDR Analysis
```python
import numpy as np
from frequency_hopping_decoder import FrequencyHoppingDecoder

# Load IQ data from SDR recording
decoder = FrequencyHoppingDecoder(sample_rate=2.4e6, center_freq=433e6)
decoder.load_iq_file("sdr_recording.iq")

# Analyze signal
hops = decoder.detect_frequency_hops(threshold_factor=2.0)
print(f"Detected {len(hops)} frequency hops")

# Decode FSK data
decoded = decoder.decode_all_hops(modulation='fsk')
```

### Example 2: Pattern Analysis
```python
# Detect repeating hop patterns
sync_info = decoder.synchronize_hops()
if sync_info['synchronized']:
    pattern_len = sync_info['pattern_length']
    print(f"Found repeating pattern of length {pattern_len}")
    
    # Extract one complete pattern
    pattern_freqs = decoder.hop_frequencies[:pattern_len]
    print(f"Pattern: {[f/1e6 for f in pattern_freqs]} MHz")
```

### Example 3: Multi-format Processing
```python
# Process different file formats
formats = ['complex64', 'complex128', 'interleaved_float32']
for fmt in formats:
    try:
        decoder = FrequencyHoppingDecoder()
        decoder.load_iq_file(f"signal.{fmt}", file_format=fmt)
        hops = decoder.detect_frequency_hops()
        print(f"{fmt}: {len(hops)} hops detected")
    except Exception as e:
        print(f"{fmt}: Error - {e}")
```

## Troubleshooting

### Common Issues

1. **No hops detected**
   - Lower the `threshold_factor`
   - Check signal strength and center frequency
   - Verify file format and sample rate

2. **Poor synchronization**
   - Increase signal duration for pattern detection
   - Verify hop timing consistency
   - Check for frequency drift

3. **Low decode success rate**
   - Adjust demodulation bandwidth
   - Check modulation type
   - Verify signal quality (SNR)

### Performance Issues

1. **Slow processing**
   - Reduce FFT size or signal duration
   - Use threshold method instead of clustering
   - Enable numba JIT compilation

2. **High memory usage**
   - Process data in smaller chunks
   - Use lower precision (complex64 vs complex128)
   - Reduce spectrogram resolution

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Built with NumPy, SciPy, and scikit-learn
- Optimized with Numba for performance
- Visualization with Matplotlib
- SDR community for testing and feedback

## References

- [Frequency Hopping Spread Spectrum](https://en.wikipedia.org/wiki/Frequency-hopping_spread_spectrum)
- [PySDR - A Guide to SDR and DSP using Python](https://pysdr.org/)
- [GNU Radio Documentation](https://www.gnuradio.org/doc/)
- [Software Defined Radio with HackRF](https://greatscottgadgets.com/hackrf/)