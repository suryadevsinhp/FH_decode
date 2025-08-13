#!/usr/bin/env python3
"""
Frequency Hopping Decoder: .bin IQ File to Audio
================================================

This script processes .bin IQ recording files and extracts audio from
frequency hopping signals, saving the output as WAV files.

Usage:
    python decode_bin_to_audio.py input.bin [options]

Examples:
    python decode_bin_to_audio.py recording.bin --sample-rate 2e6 --center-freq 433e6
    python decode_bin_to_audio.py signal.bin --format interleaved_float32 --modulation psk
"""

import argparse
import numpy as np
import scipy.signal as signal
import soundfile as sf
import os
from pathlib import Path
import logging

from frequency_hopping_decoder import FrequencyHoppingDecoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_bin_file(filename: str, file_format: str = 'interleaved_float32') -> np.ndarray:
    """
    Load IQ data from .bin file.
    
    Args:
        filename: Path to .bin file
        file_format: Format of the data ('interleaved_float32', 'complex64', 'complex128')
    
    Returns:
        Complex IQ data array
    """
    logger.info(f"Loading {filename} as {file_format}")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    file_size = os.path.getsize(filename)
    logger.info(f"File size: {file_size / (1024*1024):.1f} MB")
    
    if file_format == 'interleaved_float32':
        # Most common format: I,Q,I,Q... as float32
        raw_data = np.fromfile(filename, dtype=np.float32)
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]  # Remove last sample if odd
        iq_data = raw_data[::2] + 1j * raw_data[1::2]
        
    elif file_format == 'complex64':
        iq_data = np.fromfile(filename, dtype=np.complex64)
        
    elif file_format == 'complex128':
        iq_data = np.fromfile(filename, dtype=np.complex128)
        
    elif file_format == 'int16_interleaved':
        # Signed 16-bit integers (RTLSDR format)
        raw_data = np.fromfile(filename, dtype=np.int16)
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]
        # Normalize to [-1, 1] range
        raw_data = raw_data.astype(np.float32) / 32768.0
        iq_data = raw_data[::2] + 1j * raw_data[1::2]
        
    elif file_format == 'uint8_interleaved':
        # Unsigned 8-bit integers (RTL-SDR raw format)
        raw_data = np.fromfile(filename, dtype=np.uint8)
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]
        # Convert to signed and normalize
        raw_data = (raw_data.astype(np.float32) - 127.5) / 127.5
        iq_data = raw_data[::2] + 1j * raw_data[1::2]
        
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    logger.info(f"Loaded {len(iq_data):,} IQ samples")
    return iq_data


def extract_audio_from_hops(decoder, hops, modulation='fsk', audio_sample_rate=44100):
    """
    Extract audio data from detected frequency hops.
    
    Args:
        decoder: FrequencyHoppingDecoder instance
        hops: List of detected hops
        modulation: Modulation type
        audio_sample_rate: Target audio sample rate
    
    Returns:
        Audio data as numpy array
    """
    logger.info(f"Extracting audio from {len(hops)} hops using {modulation.upper()}")
    
    all_audio = []
    successful_extractions = 0
    
    for i, (freq, start_time, duration) in enumerate(hops):
        try:
            # Extract and demodulate the hop
            start_sample = int(start_time * decoder.sample_rate)
            duration_samples = int(duration * decoder.sample_rate)
            end_sample = min(start_sample + duration_samples, len(decoder.iq_data))
            
            if end_sample <= start_sample:
                continue
                
            hop_data = decoder.iq_data[start_sample:end_sample]
            
            # Frequency shift to baseband
            freq_offset = freq - decoder.center_freq
            t = np.arange(len(hop_data)) / decoder.sample_rate
            lo_signal = np.exp(-2j * np.pi * freq_offset * t)
            baseband_data = hop_data * lo_signal
            
            # Low-pass filter to remove aliases
            cutoff = min(20e3, decoder.sample_rate / 10)  # 20 kHz or 1/10 sample rate
            nyquist = decoder.sample_rate / 2
            
            if cutoff < nyquist:
                b, a = signal.butter(6, cutoff / nyquist, btype='low')
                filtered_data = signal.filtfilt(b, a, baseband_data)
            else:
                filtered_data = baseband_data
            
            # Demodulate to get audio
            if modulation.lower() == 'fm' or modulation.lower() == 'fsk':
                # FM demodulation using instantaneous frequency
                if np.iscomplexobj(filtered_data):
                    instantaneous_phase = np.unwrap(np.angle(filtered_data))
                else:
                    analytic_signal = signal.hilbert(filtered_data)
                    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                
                instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi) * decoder.sample_rate
                
                # Remove DC component and normalize
                audio_data = instantaneous_freq - np.mean(instantaneous_freq)
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
                
            elif modulation.lower() == 'am' or modulation.lower() == 'ask':
                # AM demodulation using envelope detection
                audio_data = np.abs(filtered_data)
                # Remove DC and normalize
                audio_data = audio_data - np.mean(audio_data)
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
                
            elif modulation.lower() == 'usb' or modulation.lower() == 'lsb':
                # SSB demodulation
                audio_data = np.real(filtered_data)
                # High-pass filter to remove DC
                if len(audio_data) > 100:
                    b, a = signal.butter(2, 300 / (decoder.sample_rate / 2), btype='high')
                    audio_data = signal.filtfilt(b, a, audio_data)
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
                
            else:
                # Default: envelope detection
                audio_data = np.abs(filtered_data)
                audio_data = audio_data - np.mean(audio_data)
                audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
            
            # Resample to audio sample rate if needed
            if len(audio_data) > 0:
                if decoder.sample_rate != audio_sample_rate:
                    # Calculate resampling ratio
                    resample_ratio = audio_sample_rate / decoder.sample_rate
                    new_length = int(len(audio_data) * resample_ratio)
                    
                    if new_length > 0:
                        audio_data = signal.resample(audio_data, new_length)
                
                all_audio.append(audio_data)
                successful_extractions += 1
                
        except Exception as e:
            logger.warning(f"Failed to extract audio from hop {i}: {e}")
            continue
    
    if not all_audio:
        logger.error("No audio could be extracted from any hops")
        return np.array([])
    
    # Concatenate all audio segments
    combined_audio = np.concatenate(all_audio)
    
    # Apply gentle bandpass filter for audio (300 Hz - 3.4 kHz for voice)
    if audio_sample_rate > 7000:  # Only if sample rate is high enough
        try:
            low_cutoff = 300 / (audio_sample_rate / 2)
            high_cutoff = min(3400, audio_sample_rate / 2 - 100) / (audio_sample_rate / 2)
            
            if low_cutoff < high_cutoff and low_cutoff > 0 and high_cutoff < 1:
                b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
                combined_audio = signal.filtfilt(b, a, combined_audio)
        except Exception as e:
            logger.warning(f"Audio filtering failed: {e}")
    
    # Final normalization
    if np.max(np.abs(combined_audio)) > 0:
        combined_audio = combined_audio / np.max(np.abs(combined_audio)) * 0.8
    
    logger.info(f"Successfully extracted audio from {successful_extractions}/{len(hops)} hops")
    logger.info(f"Audio duration: {len(combined_audio)/audio_sample_rate:.2f} seconds")
    
    return combined_audio


def process_bin_file(input_file, output_file=None, sample_rate=2e6, center_freq=433e6, 
                    file_format='interleaved_float32', modulation='fm', 
                    audio_sample_rate=44100, threshold_factor=2.5):
    """
    Process a .bin IQ file and extract audio.
    
    Args:
        input_file: Path to input .bin file
        output_file: Path to output audio file (auto-generated if None)
        sample_rate: IQ sample rate in Hz
        center_freq: Center frequency in Hz
        file_format: Format of the .bin file
        modulation: Modulation type for audio extraction
        audio_sample_rate: Output audio sample rate
        threshold_factor: Detection threshold factor
    """
    logger.info("="*60)
    logger.info("Frequency Hopping Decoder: .bin to Audio")
    logger.info("="*60)
    
    # Generate output filename if not specified
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.stem + "_decoded_audio.wav"
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Sample rate: {sample_rate/1e6:.1f} MSPS")
    logger.info(f"Center frequency: {center_freq/1e6:.1f} MHz")
    logger.info(f"File format: {file_format}")
    logger.info(f"Modulation: {modulation.upper()}")
    
    try:
        # Load IQ data
        iq_data = load_bin_file(input_file, file_format)
        
        # Create decoder
        decoder = FrequencyHoppingDecoder(sample_rate=sample_rate, center_freq=center_freq)
        decoder.load_iq_data(iq_data)
        
        # Analyze signal
        logger.info("Computing spectrogram...")
        decoder.compute_spectrogram()
        
        # Detect hops
        logger.info("Detecting frequency hops...")
        hops = decoder.detect_frequency_hops(
            method='threshold',
            threshold_factor=threshold_factor,
            min_duration=1e-3  # 1ms minimum hop duration
        )
        
        if not hops:
            logger.error("No frequency hops detected! Try adjusting --threshold-factor")
            logger.info("Suggestions:")
            logger.info("  - Lower threshold: --threshold-factor 1.5")
            logger.info("  - Check center frequency and sample rate")
            logger.info("  - Verify file format")
            return False
        
        logger.info(f"Detected {len(hops)} frequency hops")
        
        # Show hop statistics
        hop_freqs = [h[0] for h in hops]
        hop_times = [h[1] for h in hops]
        hop_durations = [h[2] for h in hops]
        
        logger.info(f"Frequency range: {min(hop_freqs)/1e6:.3f} - {max(hop_freqs)/1e6:.3f} MHz")
        logger.info(f"Time span: {min(hop_times)*1000:.1f} - {max(hop_times)*1000:.1f} ms")
        logger.info(f"Average hop duration: {np.mean(hop_durations)*1000:.2f} ms")
        
        # Extract audio
        audio_data = extract_audio_from_hops(
            decoder, hops, modulation, audio_sample_rate
        )
        
        if len(audio_data) == 0:
            logger.error("No audio could be extracted")
            return False
        
        # Save audio file
        logger.info(f"Saving audio to {output_file}")
        sf.write(output_file, audio_data, audio_sample_rate)
        
        # Show final statistics
        file_size = os.path.getsize(output_file)
        logger.info(f"Audio file saved successfully!")
        logger.info(f"Output file size: {file_size / 1024:.1f} KB")
        logger.info(f"Audio duration: {len(audio_data)/audio_sample_rate:.2f} seconds")
        logger.info(f"Audio sample rate: {audio_sample_rate} Hz")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Decode frequency hopping signals from .bin IQ files to audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.bin
  %(prog)s signal.bin --sample-rate 2000000 --center-freq 433000000
  %(prog)s data.bin --format complex64 --modulation fm
  %(prog)s rtlsdr.bin --format uint8_interleaved --threshold-factor 1.8
        """
    )
    
    parser.add_argument('input_file', help='Input .bin IQ file')
    parser.add_argument('-o', '--output', help='Output audio file (.wav)')
    parser.add_argument('-s', '--sample-rate', type=float, default=2e6,
                       help='IQ sample rate in Hz (default: 2000000)')
    parser.add_argument('-f', '--center-freq', type=float, default=433e6,
                       help='Center frequency in Hz (default: 433000000)')
    parser.add_argument('--format', default='interleaved_float32',
                       choices=['interleaved_float32', 'complex64', 'complex128', 
                               'int16_interleaved', 'uint8_interleaved'],
                       help='Input file format (default: interleaved_float32)')
    parser.add_argument('-m', '--modulation', default='fm',
                       choices=['fm', 'fsk', 'am', 'ask', 'usb', 'lsb'],
                       help='Modulation type for audio extraction (default: fm)')
    parser.add_argument('--audio-rate', type=int, default=44100,
                       help='Output audio sample rate (default: 44100)')
    parser.add_argument('--threshold-factor', type=float, default=2.5,
                       help='Detection threshold factor (default: 2.5, lower=more sensitive)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    # Process the file
    success = process_bin_file(
        input_file=args.input_file,
        output_file=args.output,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        file_format=args.format,
        modulation=args.modulation,
        audio_sample_rate=args.audio_rate,
        threshold_factor=args.threshold_factor
    )
    
    if success:
        logger.info("Processing completed successfully!")
        return 0
    else:
        logger.error("Processing failed!")
        return 1


if __name__ == "__main__":
    exit(main())