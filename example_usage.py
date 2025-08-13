#!/usr/bin/env python3
"""
Practical Example: Frequency Hopping Signal Analysis
===================================================

This example demonstrates practical usage of the frequency hopping decoder
for analyzing real-world signals from SDR recordings or test data.
"""

import numpy as np
import matplotlib.pyplot as plt
from frequency_hopping_decoder import FrequencyHoppingDecoder, generate_test_fhss_signal

def analyze_fhss_signal(iq_data, sample_rate, center_freq, modulation='fsk'):
    """
    Analyze a frequency hopping spread spectrum signal.
    
    Args:
        iq_data: Complex IQ samples
        sample_rate: Sample rate in Hz
        center_freq: Center frequency in Hz
        modulation: Modulation type ('fsk', 'psk', 'ask')
    
    Returns:
        Dictionary with analysis results
    """
    print(f"\nAnalyzing FHSS signal:")
    print(f"- Samples: {len(iq_data):,}")
    print(f"- Duration: {len(iq_data)/sample_rate*1000:.1f} ms")
    print(f"- Sample rate: {sample_rate/1e6:.1f} MSPS")
    print(f"- Center frequency: {center_freq/1e6:.1f} MHz")
    
    # Create decoder
    decoder = FrequencyHoppingDecoder(sample_rate=sample_rate, center_freq=center_freq)
    
    # Load data
    decoder.load_iq_data(iq_data)
    
    # Analyze signal
    print("\n1. Computing spectrogram...")
    decoder.compute_spectrogram()
    
    # Detect hops with different methods
    print("2. Detecting frequency hops...")
    
    # Method 1: Threshold-based (fast)
    hops_threshold = decoder.detect_frequency_hops(
        method='threshold', 
        threshold_factor=2.5,
        min_duration=0.5e-3  # 0.5ms minimum
    )
    
    # Method 2: Clustering-based (more robust)
    hops_clustering = decoder.detect_frequency_hops(
        method='clustering',
        n_hops=min(50, len(hops_threshold))
    )
    
    print(f"   Threshold method: {len(hops_threshold)} hops")
    print(f"   Clustering method: {len(hops_clustering)} hops")
    
    # Use the method with more reasonable results
    hops = hops_threshold if len(hops_threshold) < len(hops_clustering) * 2 else hops_clustering
    
    if not hops:
        print("No hops detected!")
        return None
    
    # Analyze hop characteristics
    hop_freqs = [h[0] for h in hops]
    hop_times = [h[1] for h in hops]
    hop_durations = [h[2] for h in hops]
    
    print(f"\n3. Hop Analysis:")
    print(f"   Total hops: {len(hops)}")
    print(f"   Frequency range: {min(hop_freqs)/1e6:.3f} - {max(hop_freqs)/1e6:.3f} MHz")
    print(f"   Time span: {min(hop_times)*1000:.1f} - {max(hop_times)*1000:.1f} ms")
    print(f"   Avg hop duration: {np.mean(hop_durations)*1000:.2f} ± {np.std(hop_durations)*1000:.2f} ms")
    
    # Calculate hop rate
    if len(hops) > 1:
        time_span = max(hop_times) - min(hop_times)
        hop_rate = (len(hops) - 1) / time_span if time_span > 0 else 0
        print(f"   Estimated hop rate: {hop_rate:.1f} Hz")
    
    # Unique frequencies
    unique_freqs = set([round(f, -3) for f in hop_freqs])  # Round to nearest kHz
    print(f"   Unique frequencies: {len(unique_freqs)}")
    
    # Pattern synchronization
    print("\n4. Pattern Analysis:")
    sync_info = decoder.synchronize_hops()
    if sync_info['synchronized']:
        print(f"   Pattern detected: Length {sync_info['pattern_length']}")
        print(f"   Confidence: {sync_info['confidence']:.2f}")
    else:
        print("   No repeating pattern detected")
    
    # Decode signals
    print(f"\n5. Signal Decoding ({modulation.upper()}):")
    decoded_data = decoder.decode_all_hops(modulation=modulation)
    successful_decodes = len([d for d in decoded_data if len(d) > 0])
    success_rate = successful_decodes / len(decoded_data) if decoded_data else 0
    
    print(f"   Success rate: {success_rate:.1%} ({successful_decodes}/{len(decoded_data)})")
    
    # Extract data bits from successful decodes
    total_bits = sum(len(d) for d in decoded_data if len(d) > 0)
    print(f"   Total decoded bits: {total_bits}")
    
    # Return results
    return {
        'decoder': decoder,
        'hops': hops,
        'hop_rate': hop_rate if len(hops) > 1 else 0,
        'unique_frequencies': len(unique_freqs),
        'decode_success_rate': success_rate,
        'total_bits': total_bits,
        'sync_info': sync_info
    }

def example_1_bluetooth_like():
    """Example 1: Bluetooth-like frequency hopping analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Bluetooth-like Frequency Hopping")
    print("="*60)
    
    # Simulate Bluetooth-like hopping (79 channels, 1600 hops/sec)
    # Simplified for demonstration
    test_signal, _ = generate_test_fhss_signal(
        sample_rate=2e6,
        duration=0.01,  # 10ms
        hop_rate=1600,  # 1600 Hz like Bluetooth
        num_frequencies=20,  # Simplified from 79
        modulation='fsk'
    )
    
    results = analyze_fhss_signal(
        iq_data=test_signal,
        sample_rate=2e6,
        center_freq=2.44e9,  # 2.44 GHz ISM band
        modulation='fsk'
    )
    
    if results:
        print(f"\nBluetooth Analysis Results:")
        print(f"- Estimated hop rate: {results['hop_rate']:.0f} Hz")
        print(f"- Frequency diversity: {results['unique_frequencies']} channels")
        print(f"- Data recovery: {results['decode_success_rate']:.1%}")

def example_2_military_style():
    """Example 2: Military-style frequency hopping analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Military-style Frequency Hopping")
    print("="*60)
    
    # Simulate military-style hopping (slower, more frequencies)
    test_signal, _ = generate_test_fhss_signal(
        sample_rate=5e6,
        duration=0.05,  # 50ms
        hop_rate=200,   # 200 Hz
        num_frequencies=50,
        modulation='psk'  # Phase shift keying
    )
    
    results = analyze_fhss_signal(
        iq_data=test_signal,
        sample_rate=5e6,
        center_freq=400e6,  # 400 MHz
        modulation='psk'
    )
    
    if results:
        print(f"\nMilitary Analysis Results:")
        print(f"- Estimated hop rate: {results['hop_rate']:.0f} Hz")
        print(f"- Frequency diversity: {results['unique_frequencies']} channels")
        print(f"- Data recovery: {results['decode_success_rate']:.1%}")

def example_3_custom_pattern():
    """Example 3: Custom hopping pattern analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Pattern Analysis")
    print("="*60)
    
    # Create a signal with a known repeating pattern
    decoder = FrequencyHoppingDecoder(sample_rate=1e6, center_freq=433e6)
    
    # Define custom pattern (frequencies relative to center)
    custom_pattern = [-50e3, -25e3, 0, 25e3, 50e3, 75e3]  # 6-frequency pattern
    
    # Generate signal manually with this pattern
    duration = 0.03  # 30ms
    samples_per_hop = int(0.005 * 1e6)  # 5ms per hop
    
    signal_data = np.zeros(int(duration * 1e6), dtype=np.complex64)
    current_sample = 0
    
    print(f"Generating signal with custom pattern: {[f/1e3 for f in custom_pattern]} kHz")
    
    # Repeat pattern 5 times
    for repeat in range(5):
        for freq in custom_pattern:
            if current_sample >= len(signal_data):
                break
            
            hop_samples = min(samples_per_hop, len(signal_data) - current_sample)
            t = np.arange(hop_samples) / 1e6
            
            # Generate BPSK signal
            data_bit = np.random.randint(0, 2)
            phase = np.pi if data_bit else 0
            
            hop_signal = np.exp(2j * np.pi * freq * t + 1j * phase)
            signal_data[current_sample:current_sample + hop_samples] = hop_signal
            current_sample += hop_samples
    
    # Add some noise
    noise = (np.random.randn(len(signal_data)) + 1j * np.random.randn(len(signal_data))) * 0.05
    signal_data += noise
    
    results = analyze_fhss_signal(
        iq_data=signal_data,
        sample_rate=1e6,
        center_freq=433e6,
        modulation='psk'
    )
    
    if results:
        print(f"\nCustom Pattern Results:")
        print(f"- Expected pattern length: {len(custom_pattern)}")
        print(f"- Detected pattern length: {results['sync_info']['pattern_length']}")
        print(f"- Pattern confidence: {results['sync_info']['confidence']:.2f}")

def visualize_results(decoder):
    """Create visualization of analysis results."""
    if decoder and decoder.spectrogram is not None:
        plt.figure(figsize=(12, 6))
        
        # Plot spectrogram
        spec_db = 20 * np.log10(decoder.spectrogram + 1e-10)
        
        plt.pcolormesh(
            decoder.time_bins * 1000,  # ms
            decoder.frequency_bins / 1e6,  # MHz
            spec_db,
            shading='gouraud',
            cmap='viridis'
        )
        
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.title('Frequency Hopping Signal Analysis')
        
        # Overlay detected hops
        if decoder.hop_frequencies and decoder.hop_times:
            plt.scatter(
                [t * 1000 for t in decoder.hop_times],
                [f / 1e6 for f in decoder.hop_frequencies],
                c='red', s=20, alpha=0.7, label='Detected Hops'
            )
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('/workspace/fhss_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: fhss_analysis.png")

def main():
    """Run all examples."""
    print("Frequency Hopping Signal Analysis Examples")
    print("=" * 60)
    
    # Run examples
    try:
        example_1_bluetooth_like()
        example_2_military_style()
        example_3_custom_pattern()
        
        print(f"\n" + "="*60)
        print("All examples completed successfully!")
        print("\nTo analyze your own IQ files:")
        print("1. Load your IQ data: decoder.load_iq_file('your_file.iq')")
        print("2. Run analysis: analyze_fhss_signal(iq_data, sample_rate, center_freq)")
        print("3. Adjust threshold_factor if needed for better detection")
        
    except Exception as e:
        print(f"Error running examples: {e}")

if __name__ == "__main__":
    main()