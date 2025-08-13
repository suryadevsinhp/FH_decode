#!/usr/bin/env python3
"""
GNU Radio Frequency Hopping Decoder - Simple Version
===================================================

This script automatically processes GNU Radio .bin files and extracts audio.
Just change the filename below and run the script.

Usage:
    python3 gnuradio_decoder.py
"""

import numpy as np
import scipy.signal as signal
import soundfile as sf
import os
from frequency_hopping_decoder import FrequencyHoppingDecoder

# =============================================================================
# CONFIGURATION - CHANGE THESE VALUES FOR YOUR FILE
# =============================================================================

# Your GNU Radio .bin file name (change this to your file)
INPUT_FILE = "your_gnuradio_recording.bin"

# Sample rate used in GNU Radio (Hz)
SAMPLE_RATE = 2000000  # 2 MSPS - change if different

# Center frequency used in GNU Radio (Hz)  
CENTER_FREQ = 433000000  # 433 MHz - change to your frequency

# Audio demodulation type
MODULATION = "fm"  # Options: "fm", "am", "usb", "lsb"

# Detection sensitivity (lower = more sensitive)
THRESHOLD = 2.0  # Try 1.5 for weak signals, 3.0 for strong signals

# =============================================================================
# AUTOMATIC PROCESSING CODE - NO NEED TO CHANGE BELOW
# =============================================================================

def load_gnuradio_file(filename):
    """Load GNU Radio complex64 .bin file."""
    print(f"Loading GNU Radio file: {filename}")
    
    if not os.path.exists(filename):
        print(f"ERROR: File not found: {filename}")
        print("Please check the filename in the script.")
        return None
    
    # GNU Radio typically saves as complex64 format
    iq_data = np.fromfile(filename, dtype=np.complex64)
    
    file_size = os.path.getsize(filename)
    duration = len(iq_data) / SAMPLE_RATE
    
    print(f"File size: {file_size / (1024*1024):.1f} MB")
    print(f"Samples: {len(iq_data):,}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {SAMPLE_RATE/1e6:.1f} MSPS")
    print(f"Center frequency: {CENTER_FREQ/1e6:.1f} MHz")
    
    return iq_data

def extract_audio_automatically(decoder, hops):
    """Automatically extract audio from detected hops."""
    print(f"\nExtracting audio from {len(hops)} hops using {MODULATION.upper()} demodulation...")
    
    all_audio = []
    success_count = 0
    
    for i, (freq, start_time, duration) in enumerate(hops):
        try:
            # Get the hop data
            start_sample = int(start_time * decoder.sample_rate)
            duration_samples = int(duration * decoder.sample_rate)
            end_sample = min(start_sample + duration_samples, len(decoder.iq_data))
            
            if end_sample <= start_sample:
                continue
                
            hop_data = decoder.iq_data[start_sample:end_sample]
            
            # Shift to baseband
            freq_offset = freq - decoder.center_freq
            t = np.arange(len(hop_data)) / decoder.sample_rate
            lo_signal = np.exp(-2j * np.pi * freq_offset * t)
            baseband_data = hop_data * lo_signal
            
            # Low-pass filter
            cutoff = 20e3  # 20 kHz audio bandwidth
            nyquist = decoder.sample_rate / 2
            b, a = signal.butter(6, cutoff / nyquist, btype='low')
            filtered_data = signal.filtfilt(b, a, baseband_data)
            
            # Demodulate based on type
            if MODULATION.lower() == 'fm':
                # FM demodulation
                instantaneous_phase = np.unwrap(np.angle(filtered_data))
                instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi) * decoder.sample_rate
                audio_data = instantaneous_freq - np.mean(instantaneous_freq)
                
            elif MODULATION.lower() == 'am':
                # AM demodulation
                audio_data = np.abs(filtered_data)
                audio_data = audio_data - np.mean(audio_data)
                
            elif MODULATION.lower() in ['usb', 'lsb']:
                # SSB demodulation
                audio_data = np.real(filtered_data)
                # High-pass filter for voice
                b, a = signal.butter(2, 300 / (decoder.sample_rate / 2), btype='high')
                audio_data = signal.filtfilt(b, a, audio_data)
                
            else:
                # Default to envelope detection
                audio_data = np.abs(filtered_data)
                audio_data = audio_data - np.mean(audio_data)
            
            # Normalize
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample to 44.1 kHz for audio
            if len(audio_data) > 0:
                audio_sample_rate = 44100
                resample_ratio = audio_sample_rate / decoder.sample_rate
                new_length = int(len(audio_data) * resample_ratio)
                
                if new_length > 0:
                    audio_data = signal.resample(audio_data, new_length)
                    all_audio.append(audio_data)
                    success_count += 1
                    
        except Exception as e:
            print(f"Warning: Failed to process hop {i+1}: {e}")
            continue
    
    if not all_audio:
        print("ERROR: No audio could be extracted from any hops!")
        return None
    
    # Combine all audio
    combined_audio = np.concatenate(all_audio)
    
    # Apply audio bandpass filter (300 Hz - 3.4 kHz for voice)
    audio_sample_rate = 44100
    low_cutoff = 300 / (audio_sample_rate / 2)
    high_cutoff = 3400 / (audio_sample_rate / 2)
    
    try:
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        combined_audio = signal.filtfilt(b, a, combined_audio)
    except:
        pass  # Skip filtering if it fails
    
    # Final normalization
    if np.max(np.abs(combined_audio)) > 0:
        combined_audio = combined_audio / np.max(np.abs(combined_audio)) * 0.8
    
    print(f"Successfully extracted audio from {success_count}/{len(hops)} hops")
    print(f"Audio duration: {len(combined_audio)/44100:.2f} seconds")
    
    return combined_audio

def main():
    """Main processing function."""
    print("="*60)
    print("GNU Radio Frequency Hopping Decoder")
    print("="*60)
    
    # Load the file
    iq_data = load_gnuradio_file(INPUT_FILE)
    if iq_data is None:
        return
    
    # Create decoder
    print(f"\nInitializing decoder...")
    decoder = FrequencyHoppingDecoder(sample_rate=SAMPLE_RATE, center_freq=CENTER_FREQ)
    decoder.load_iq_data(iq_data)
    
    # Analyze signal
    print("Computing spectrogram...")
    decoder.compute_spectrogram()
    
    # Detect frequency hops
    print("Detecting frequency hops...")
    hops = decoder.detect_frequency_hops(
        method='threshold',
        threshold_factor=THRESHOLD,
        min_duration=1e-3  # 1ms minimum
    )
    
    if not hops:
        print("\nERROR: No frequency hops detected!")
        print("\nTroubleshooting:")
        print(f"1. Try lowering THRESHOLD to 1.5 (currently {THRESHOLD})")
        print(f"2. Check CENTER_FREQ is correct (currently {CENTER_FREQ/1e6:.1f} MHz)")
        print(f"3. Check SAMPLE_RATE is correct (currently {SAMPLE_RATE/1e6:.1f} MSPS)")
        return
    
    # Show detection results
    print(f"\n✓ Detected {len(hops)} frequency hops!")
    
    hop_freqs = [h[0] for h in hops]
    hop_times = [h[1] for h in hops]
    hop_durations = [h[2] for h in hops]
    
    print(f"Frequency range: {min(hop_freqs)/1e6:.3f} - {max(hop_freqs)/1e6:.3f} MHz")
    print(f"Time span: {min(hop_times)*1000:.1f} - {max(hop_times)*1000:.1f} ms")
    print(f"Average hop duration: {np.mean(hop_durations)*1000:.2f} ms")
    
    # Calculate hop rate
    if len(hops) > 1:
        time_span = max(hop_times) - min(hop_times)
        hop_rate = (len(hops) - 1) / time_span if time_span > 0 else 0
        print(f"Estimated hop rate: {hop_rate:.1f} Hz")
    
    # Extract audio
    audio_data = extract_audio_automatically(decoder, hops)
    if audio_data is None:
        return
    
    # Save audio file
    output_filename = INPUT_FILE.replace('.bin', '_decoded_audio.wav')
    print(f"\nSaving audio to: {output_filename}")
    
    sf.write(output_filename, audio_data, 44100)
    
    # Show final results
    file_size = os.path.getsize(output_filename)
    print(f"\n✓ SUCCESS! Audio file saved!")
    print(f"Output file: {output_filename}")
    print(f"File size: {file_size / 1024:.1f} KB")
    print(f"Audio duration: {len(audio_data)/44100:.2f} seconds")
    print(f"Sample rate: 44100 Hz")
    
    print(f"\n" + "="*60)
    print("Processing completed! You can now play the audio file.")
    print("="*60)

if __name__ == "__main__":
    main()