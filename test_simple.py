#!/usr/bin/env python3
"""
Simple test script to verify frequency hopping decoder functionality
without web interface dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
import os

def generate_test_signal(duration=5, sample_rate=2e6, center_freq=433e6, hop_rate=100):
    """Generate a simple frequency hopping test signal"""
    
    # Frequency hops around center frequency
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
    signal_data = np.zeros(num_samples, dtype=complex)
    
    print(f"Generating {duration}s test signal...")
    print(f"Sample rate: {sample_rate/1e6:.1f} MHz")
    print(f"Center frequency: {center_freq/1e6:.1f} MHz")
    print(f"Hop rate: {hop_rate} Hz")
    print(f"Samples per hop: {samples_per_hop}")
    
    for i in range(num_samples):
        hop_index = int(i / samples_per_hop) % len(frequencies)
        freq = frequencies[hop_index]
        
        # Generate complex sinusoid at hop frequency
        phase = 2 * np.pi * (freq - center_freq) * t[i]
        signal_data[i] = np.exp(1j * phase)
    
    # Add some noise
    noise_power = 0.1
    noise = noise_power * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
    signal_data += noise
    
    return signal_data, frequencies

def save_iq_data(signal_data, filename):
    """Save IQ data to binary file"""
    # Interleave I and Q components
    iq_data = np.zeros(len(signal_data) * 2, dtype=np.float32)
    iq_data[0::2] = np.real(signal_data).astype(np.float32)
    iq_data[1::2] = np.imag(signal_data).astype(np.float32)
    
    with open(filename, 'wb') as f:
        f.write(iq_data.tobytes())
    
    print(f"Saved {len(signal_data)} IQ samples to {filename}")
    return filename

def test_frequency_analysis(signal_data, sample_rate, center_freq):
    """Test frequency analysis functionality"""
    print("\nTesting frequency analysis...")
    
    # Parameters for STFT
    nperseg = 1024
    noverlap = nperseg // 4
    
    # Compute Short-Time Fourier Transform
    frequencies, times, Zxx = signal.stft(
        signal_data, 
        fs=sample_rate, 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    
    # Convert to power spectrum
    power_spectrum = np.abs(Zxx) ** 2
    
    print(f"STFT computed: {len(frequencies)} frequency bins, {len(times)} time windows")
    
    # Find peak frequencies in each time window
    hop_frequencies = []
    hop_times = []
    
    for i, time_slice in enumerate(power_spectrum.T):
        # Find the peak frequency in this time slice
        peak_idx = np.argmax(time_slice)
        peak_freq = frequencies[peak_idx] + center_freq
        
        hop_frequencies.append(peak_freq)
        hop_times.append(times[i])
    
    print(f"Detected {len(hop_frequencies)} frequency hops")
    
    # Create spectrogram plot
    plt.figure(figsize=(12, 8))
    
    # Convert to dB scale
    power_db = 10 * np.log10(power_spectrum + 1e-10)
    
    plt.pcolormesh(times, frequencies + center_freq, power_db, shading='gouraud')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequency Hopping Spectrogram')
    
    plot_filename = 'test_spectrogram.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Spectrogram saved to {plot_filename}")
    
    return hop_frequencies, hop_times

def test_audio_extraction(signal_data, hop_frequencies, hop_times, sample_rate):
    """Test audio extraction functionality"""
    print("\nTesting audio extraction...")
    
    audio_segments = []
    
    for i, (freq, start_time) in enumerate(zip(hop_frequencies, hop_times)):
        if i >= len(hop_times) - 1:
            break
            
        # Calculate time window
        end_time = hop_times[i + 1] if i + 1 < len(hop_times) else hop_times[i] + 0.1
        
        # Convert time to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        if end_sample > len(signal_data):
            end_sample = len(signal_data)
        
        # Extract segment
        segment = signal_data[start_sample:end_sample]
        
        if len(segment) > 0:
            # Simple AM demodulation (magnitude)
            audio_segment = np.abs(segment)
            
            # Normalize
            if np.max(audio_segment) > 0:
                audio_segment = audio_segment / np.max(audio_segment)
            
            audio_segments.append(audio_segment)
    
    # Concatenate all segments
    if audio_segments:
        full_audio = np.concatenate(audio_segments)
        
        # Resample to standard audio sample rate
        target_sample_rate = 44100
        if sample_rate != target_sample_rate:
            import librosa
            full_audio = librosa.resample(full_audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        
        # Ensure audio is in valid range
        full_audio = np.clip(full_audio, -1.0, 1.0)
        
        # Save as WAV file
        output_filename = 'test_decoded_audio.wav'
        sf.write(output_filename, full_audio, target_sample_rate)
        
        print(f"Decoded audio saved to {output_filename}")
        print(f"Audio duration: {len(full_audio)/target_sample_rate:.2f} seconds")
        
        return output_filename
    else:
        print("No audio segments extracted")
        return None

def main():
    print("🧪 Frequency Hopping Decoder - Simple Test")
    print("=" * 50)
    
    # Test parameters
    duration = 5  # seconds
    sample_rate = 2e6  # 2 MHz
    center_freq = 433e6  # 433 MHz
    hop_rate = 100  # 100 hops per second
    
    try:
        # Generate test signal
        signal_data, expected_freqs = generate_test_signal(duration, sample_rate, center_freq, hop_rate)
        
        # Save to file
        bin_filename = save_iq_data(signal_data, 'test_signal.bin')
        
        # Test frequency analysis
        hop_frequencies, hop_times = test_frequency_analysis(signal_data, sample_rate, center_freq)
        
        # Test audio extraction
        audio_filename = test_audio_extraction(signal_data, hop_frequencies, hop_times, sample_rate)
        
        print(f"\n✅ Test completed successfully!")
        print(f"Generated files:")
        print(f"  - {bin_filename} (IQ data)")
        print(f"  - test_spectrogram.png (frequency plot)")
        if audio_filename:
            print(f"  - {audio_filename} (decoded audio)")
        
        print(f"\nDetected {len(hop_frequencies)} frequency hops")
        print(f"Expected frequencies: {[f/1e6 for f in expected_freqs]} MHz")
        
        # Show some detected frequencies
        unique_freqs = sorted(list(set([round(f/1e6, 1) for f in hop_frequencies])))
        print(f"Detected frequencies: {unique_freqs} MHz")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()