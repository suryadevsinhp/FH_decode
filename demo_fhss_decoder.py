#!/usr/bin/env python3
"""
Frequency Hopping Decoder - Demonstration Script
===============================================

This script demonstrates various capabilities of the frequency hopping decoder:
1. Processing IQ files from SDR
2. Real-time analysis with different modulation schemes
3. Pattern synchronization and tracking
4. Advanced signal analysis and visualization

Usage:
    python demo_fhss_decoder.py [options]

Examples:
    # Analyze a file
    python demo_fhss_decoder.py --file signal.iq --format complex64
    
    # Generate and analyze test signal
    python demo_fhss_decoder.py --test --modulation fsk --duration 0.1
    
    # Advanced analysis with known pattern
    python demo_fhss_decoder.py --file signal.iq --sync-pattern "915.1e6,915.2e6,915.3e6"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from frequency_hopping_decoder import FrequencyHoppingDecoder, generate_test_fhss_signal


def demo_basic_analysis():
    """Demonstrate basic frequency hopping analysis."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Frequency Hopping Analysis")
    print("="*60)
    
    # Create decoder
    decoder = FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)
    
    # Generate test signal
    print("Generating test FHSS signal (FSK modulation)...")
    test_signal, expected_freqs = generate_test_fhss_signal(
        sample_rate=2e6,
        duration=0.1,  # 100ms
        hop_rate=500,  # 500 Hz hop rate
        num_frequencies=8,
        modulation='fsk'
    )
    
    print(f"Generated {len(test_signal)} samples ({len(test_signal)/2e6*1000:.1f} ms)")
    print(f"Expected hop frequencies: {len(set(expected_freqs))} unique frequencies")
    
    # Load and analyze
    decoder.load_iq_data(test_signal)
    
    # Compute spectrogram
    print("\nComputing spectrogram...")
    freqs, times, spec = decoder.compute_spectrogram()
    
    # Detect hops
    print("Detecting frequency hops...")
    hops = decoder.detect_frequency_hops(method='threshold', threshold_factor=2.5)
    
    print(f"\nResults:")
    print(f"- Detected {len(hops)} hops")
    print(f"- Frequency range: {min(h[0] for h in hops)/1e3:.1f} to {max(h[0] for h in hops)/1e3:.1f} kHz")
    print(f"- Time span: {min(h[1] for h in hops)*1000:.1f} to {max(h[1] for h in hops)*1000:.1f} ms")
    
    # Show first few hops
    print(f"\nFirst 5 detected hops:")
    for i, (freq, start_time, duration) in enumerate(hops[:5]):
        print(f"  {i+1}: {freq/1e3:+7.1f} kHz @ {start_time*1000:6.1f} ms (Δt={duration*1000:4.1f} ms)")
    
    # Synchronization
    print("\nSynchronizing hop pattern...")
    sync_info = decoder.synchronize_hops()
    print(f"Synchronization confidence: {sync_info['confidence']:.2f}")
    if sync_info['pattern_length'] > 0:
        print(f"Detected pattern length: {sync_info['pattern_length']} hops")
    
    # Decode signals
    print("\nDecoding frequency hops...")
    decoded_data = decoder.decode_all_hops(modulation='fsk')
    successful_decodes = len([d for d in decoded_data if len(d) > 0])
    print(f"Successfully decoded {successful_decodes}/{len(decoded_data)} hops")
    
    return decoder


def demo_modulation_comparison():
    """Demonstrate analysis of different modulation schemes."""
    print("\n" + "="*60)
    print("DEMO 2: Modulation Scheme Comparison")
    print("="*60)
    
    modulations = ['fsk', 'psk', 'ask']
    results = {}
    
    for mod in modulations:
        print(f"\nAnalyzing {mod.upper()} modulation...")
        
        # Generate signal
        test_signal, _ = generate_test_fhss_signal(
            sample_rate=2e6,
            duration=0.05,  # 50ms
            hop_rate=1000,  # 1 kHz
            num_frequencies=5,
            modulation=mod
        )
        
        # Analyze
        decoder = FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)
        decoder.load_iq_data(test_signal)
        
        start_time = time.time()
        hops = decoder.detect_frequency_hops(method='threshold', threshold_factor=2.0)
        decode_time = time.time() - start_time
        
        decoded_data = decoder.decode_all_hops(modulation=mod)
        successful_decodes = len([d for d in decoded_data if len(d) > 0])
        
        results[mod] = {
            'hops_detected': len(hops),
            'decode_success_rate': successful_decodes / len(hops) if hops else 0,
            'processing_time': decode_time,
            'avg_hop_duration': np.mean([h[2] for h in hops]) * 1000 if hops else 0
        }
        
        print(f"  - Detected {len(hops)} hops")
        print(f"  - Decode success rate: {results[mod]['decode_success_rate']:.1%}")
        print(f"  - Processing time: {decode_time:.3f} seconds")
    
    # Summary comparison
    print(f"\n{'Modulation':<12} {'Hops':<6} {'Success':<8} {'Time':<8} {'Avg Duration'}")
    print("-" * 50)
    for mod, res in results.items():
        print(f"{mod.upper():<12} {res['hops_detected']:<6} {res['decode_success_rate']:.1%:<8} "
              f"{res['processing_time']:.3f}s {res['avg_hop_duration']:.1f}ms")


def demo_pattern_synchronization():
    """Demonstrate synchronization with known hop patterns."""
    print("\n" + "="*60)
    print("DEMO 3: Pattern Synchronization")
    print("="*60)
    
    # Define a known hopping pattern (frequencies in Hz relative to baseband)
    known_pattern = [-100e3, -50e3, 0, 50e3, 100e3, 150e3, -150e3, -75e3]
    print(f"Known pattern: {[f/1e3 for f in known_pattern]} kHz")
    
    # Generate signal with this pattern repeated
    decoder = FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)
    
    # Create a signal that repeats the known pattern
    duration = 0.08  # 80ms
    hop_rate = len(known_pattern) * 10  # Complete pattern 10 times
    
    num_samples = int(duration * decoder.sample_rate)
    samples_per_hop = num_samples // (len(known_pattern) * 10)
    
    signal_data = np.zeros(num_samples, dtype=np.complex64)
    current_sample = 0
    
    print(f"Generating signal with repeating pattern...")
    for repeat in range(10):  # Repeat pattern 10 times
        for freq in known_pattern:
            if current_sample >= num_samples:
                break
            
            hop_samples = min(samples_per_hop, num_samples - current_sample)
            t = np.arange(hop_samples) / decoder.sample_rate
            
            # Generate FSK signal
            deviation = 2e3  # 2 kHz deviation
            data_bit = np.random.randint(0, 2)
            freq_offset = deviation if data_bit else -deviation
            
            hop_signal = np.exp(2j * np.pi * (freq + freq_offset) * t)
            signal_data[current_sample:current_sample + hop_samples] = hop_signal
            current_sample += hop_samples
    
    # Add noise
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * 0.1
    signal_data += noise
    
    # Load and analyze
    decoder.load_iq_data(signal_data)
    print("Detecting hops...")
    hops = decoder.detect_frequency_hops(method='threshold', threshold_factor=2.0)
    
    print(f"Detected {len(hops)} hops")
    
    # Try synchronization without known pattern
    print("\nAuto-detecting pattern...")
    sync_auto = decoder.synchronize_hops()
    print(f"Auto-detected pattern length: {sync_auto['pattern_length']}")
    print(f"Auto-sync confidence: {sync_auto['confidence']:.2f}")
    
    # Try synchronization with known pattern
    print("\nSynchronizing with known pattern...")
    # Convert known pattern to absolute frequencies
    known_pattern_abs = [decoder.center_freq + f for f in known_pattern]
    sync_known = decoder.synchronize_hops(reference_pattern=known_pattern_abs)
    
    print(f"Known pattern sync confidence: {sync_known['confidence']:.2f}")
    print(f"Pattern offset: {sync_known['offset']} hops")
    
    # Compare detected vs expected pattern
    if len(hops) >= len(known_pattern):
        detected_freqs = [h[0] for h in hops[:len(known_pattern)]]
        print(f"\nPattern comparison (first {len(known_pattern)} hops):")
        print(f"{'Expected':<10} {'Detected':<10} {'Error'}")
        print("-" * 35)
        for exp, det in zip(known_pattern_abs, detected_freqs):
            error = abs(exp - det)
            print(f"{exp/1e6:.3f} MHz   {det/1e6:.3f} MHz   {error/1e3:.1f} kHz")


def demo_file_processing(filename: str, file_format: str = 'complex64'):
    """Demonstrate processing of IQ data from file."""
    print("\n" + "="*60)
    print(f"DEMO 4: File Processing - {filename}")
    print("="*60)
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        print("Creating a sample file for demonstration...")
        
        # Create a sample file
        sample_signal, _ = generate_test_fhss_signal(
            sample_rate=2e6,
            duration=0.2,  # 200ms
            hop_rate=800,
            num_frequencies=12,
            modulation='fsk'
        )
        
        sample_signal.tofile(filename)
        print(f"Created sample file: {filename}")
    
    # Load and process the file
    decoder = FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)
    
    print(f"Loading IQ file: {filename}")
    print(f"Format: {file_format}")
    
    try:
        decoder.load_iq_file(filename, file_format=file_format)
        
        # File statistics
        signal_length = len(decoder.iq_data)
        duration_ms = signal_length / decoder.sample_rate * 1000
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        
        print(f"File statistics:")
        print(f"  - File size: {file_size_mb:.2f} MB")
        print(f"  - Samples: {signal_length:,}")
        print(f"  - Duration: {duration_ms:.1f} ms")
        print(f"  - Sample rate: {decoder.sample_rate/1e6:.1f} MSPS")
        
        # Signal quality analysis
        signal_power = np.mean(np.abs(decoder.iq_data)**2)
        max_amplitude = np.max(np.abs(decoder.iq_data))
        
        print(f"  - Average power: {10*np.log10(signal_power):.1f} dB")
        print(f"  - Peak amplitude: {max_amplitude:.3f}")
        print(f"  - Dynamic range: {20*np.log10(max_amplitude/np.sqrt(signal_power)):.1f} dB")
        
        # Process signal
        print("\nProcessing signal...")
        start_time = time.time()
        
        # Compute spectrogram with appropriate resolution
        fft_size = min(1024, signal_length // 10)
        decoder.fft_size = fft_size
        decoder.compute_spectrogram()
        
        # Detect hops
        hops = decoder.detect_frequency_hops(method='threshold', threshold_factor=2.5)
        
        processing_time = time.time() - start_time
        
        print(f"Processing results:")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Detected {len(hops)} frequency hops")
        
        if hops:
            # Hop statistics
            hop_freqs = [h[0] for h in hops]
            hop_times = [h[1] for h in hops]
            hop_durations = [h[2] for h in hops]
            
            print(f"  - Frequency range: {min(hop_freqs)/1e6:.3f} - {max(hop_freqs)/1e6:.3f} MHz")
            print(f"  - Time span: {min(hop_times)*1000:.1f} - {max(hop_times)*1000:.1f} ms")
            print(f"  - Hop duration: {np.mean(hop_durations)*1000:.1f} ± {np.std(hop_durations)*1000:.1f} ms")
            
            # Decode hops
            print("\nDecoding hops...")
            decoded_data = decoder.decode_all_hops(modulation='fsk')
            success_rate = len([d for d in decoded_data if len(d) > 0]) / len(decoded_data)
            print(f"  - Decode success rate: {success_rate:.1%}")
            
            # Save results
            results_file = f"{Path(filename).stem}_analysis.h5"
            decoder.save_results(results_file)
            print(f"  - Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error processing file: {e}")


def demo_advanced_visualization(decoder):
    """Create advanced visualizations of the frequency hopping analysis."""
    print("\n" + "="*60)
    print("DEMO 5: Advanced Visualization")
    print("="*60)
    
    if decoder.spectrogram is None:
        print("Computing spectrogram for visualization...")
        decoder.compute_spectrogram()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Frequency Hopping Signal Analysis', fontsize=16)
    
    # 1. Spectrogram
    ax1 = axes[0, 0]
    spec_db = 20 * np.log10(decoder.spectrogram + 1e-10)
    im1 = ax1.pcolormesh(
        decoder.time_bins * 1000,
        decoder.frequency_bins / 1e6,
        spec_db,
        shading='gouraud',
        cmap='viridis'
    )
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (MHz)')
    ax1.set_title('Spectrogram')
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    # Overlay detected hops
    if decoder.hop_frequencies and decoder.hop_times:
        ax1.scatter(
            [t * 1000 for t in decoder.hop_times],
            [f / 1e6 for f in decoder.hop_frequencies],
            c='red', s=50, alpha=0.7, label='Detected Hops'
        )
        ax1.legend()
    
    # 2. Hop frequency vs time
    ax2 = axes[0, 1]
    if decoder.hop_frequencies and decoder.hop_times:
        ax2.plot(
            [t * 1000 for t in decoder.hop_times],
            [f / 1e6 for f in decoder.hop_frequencies],
            'bo-', markersize=6, linewidth=2
        )
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Hop Frequency (MHz)')
        ax2.set_title('Hop Sequence')
        ax2.grid(True, alpha=0.3)
    
    # 3. Frequency histogram
    ax3 = axes[1, 0]
    if decoder.hop_frequencies:
        freq_mhz = [f / 1e6 for f in decoder.hop_frequencies]
        ax3.hist(freq_mhz, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Frequency (MHz)')
        ax3.set_ylabel('Hop Count')
        ax3.set_title('Frequency Usage Distribution')
        ax3.grid(True, alpha=0.3)
    
    # 4. Signal power over time
    ax4 = axes[1, 1]
    if decoder.iq_data is not None:
        # Calculate signal power in time segments
        segment_length = len(decoder.iq_data) // 100
        power_segments = []
        time_segments = []
        
        for i in range(0, len(decoder.iq_data) - segment_length, segment_length):
            segment = decoder.iq_data[i:i + segment_length]
            power = np.mean(np.abs(segment)**2)
            power_segments.append(10 * np.log10(power + 1e-10))
            time_segments.append(i / decoder.sample_rate * 1000)
        
        ax4.plot(time_segments, power_segments, 'g-', linewidth=2)
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('Power (dB)')
        ax4.set_title('Signal Power vs Time')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: 3D waterfall
    if decoder.spectrogram is not None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Subsample for better performance
        step = max(1, decoder.spectrogram.shape[1] // 100)
        T, F = np.meshgrid(
            decoder.time_bins[::step] * 1000,
            decoder.frequency_bins / 1e6
        )
        Z = decoder.spectrogram[:, ::step]
        
        surf = ax.plot_surface(T, F, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_zlabel('Magnitude')
        ax.set_title('3D Spectrogram Waterfall')
        
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description='Frequency Hopping Decoder Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                           # Run basic test with generated signal
  %(prog)s --test --modulation psk          # Test with PSK modulation
  %(prog)s --file signal.iq                 # Analyze IQ file
  %(prog)s --file signal.iq --format interleaved_float32
  %(prog)s --test --visualize               # Include advanced visualizations
  %(prog)s --all-demos                      # Run all demonstrations
        """
    )
    
    parser.add_argument('--file', type=str, help='IQ data file to process')
    parser.add_argument('--format', type=str, default='complex64',
                       choices=['complex64', 'complex128', 'interleaved_float32'],
                       help='IQ file format')
    parser.add_argument('--test', action='store_true', help='Run with generated test signal')
    parser.add_argument('--modulation', type=str, default='fsk',
                       choices=['fsk', 'psk', 'ask'],
                       help='Modulation scheme for test signal')
    parser.add_argument('--duration', type=float, default=0.1,
                       help='Duration of test signal in seconds')
    parser.add_argument('--sample-rate', type=float, default=2e6,
                       help='Sample rate in Hz')
    parser.add_argument('--center-freq', type=float, default=915e6,
                       help='Center frequency in Hz')
    parser.add_argument('--sync-pattern', type=str,
                       help='Known hop pattern for synchronization (comma-separated frequencies)')
    parser.add_argument('--visualize', action='store_true',
                       help='Show advanced visualizations')
    parser.add_argument('--all-demos', action='store_true',
                       help='Run all demonstration scenarios')
    
    args = parser.parse_args()
    
    print("Frequency Hopping Decoder - Comprehensive Demonstration")
    print("=" * 60)
    
    decoder = None
    
    try:
        if args.all_demos:
            # Run all demonstrations
            decoder = demo_basic_analysis()
            demo_modulation_comparison()
            demo_pattern_synchronization()
            
            # Create a sample file for file demo
            sample_file = "sample_fhss.iq"
            if not os.path.exists(sample_file):
                sample_signal, _ = generate_test_fhss_signal(duration=0.1)
                sample_signal.tofile(sample_file)
            demo_file_processing(sample_file)
            
            if args.visualize:
                demo_advanced_visualization(decoder)
        
        elif args.file:
            # Process specified file
            demo_file_processing(args.file, args.format)
        
        elif args.test:
            # Run basic test
            decoder = demo_basic_analysis()
            
            if args.visualize:
                demo_advanced_visualization(decoder)
        
        else:
            # Default: run basic analysis
            decoder = demo_basic_analysis()
        
        print("\nDemonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())