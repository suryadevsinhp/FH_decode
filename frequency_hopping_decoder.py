#!/usr/bin/env python3
"""
Frequency Hopping Signal Decoder for IQ Data
===========================================

This module provides a comprehensive frequency hopping decoder that can:
- Process IQ data from SDR devices or files
- Detect and synchronize with frequency hopping patterns
- Demodulate various modulation schemes (FSK, PSK, etc.)
- Extract data from frequency hopping signals

Author: AI Assistant
License: MIT
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy.signal import butter, filtfilt, hilbert
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numba import jit
import h5py
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrequencyHoppingDecoder:
    """
    A comprehensive frequency hopping signal decoder for IQ data.
    
    This class provides methods to:
    - Load and preprocess IQ data
    - Detect frequency hopping patterns
    - Synchronize with hop timing
    - Demodulate signals at each frequency
    - Extract transmitted data
    """
    
    def __init__(self, sample_rate: float = 2e6, center_freq: float = 915e6):
        """
        Initialize the frequency hopping decoder.
        
        Args:
            sample_rate: Sample rate of the IQ data in Hz
            center_freq: Center frequency of the received signal in Hz
        """
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.iq_data = None
        self.hop_frequencies = []
        self.hop_times = []
        self.hop_duration = None
        self.frequency_bins = None
        self.time_bins = None
        self.spectrogram = None
        
        # Signal processing parameters
        self.fft_size = 1024
        self.overlap = 0.5
        self.window = 'hann'
        
        # Demodulation parameters
        self.demod_bandwidth = 50e3  # 50 kHz default bandwidth per hop
        self.filter_order = 6
        
    def load_iq_data(self, data: np.ndarray) -> None:
        """
        Load IQ data for processing.
        
        Args:
            data: Complex IQ data as numpy array
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if data.dtype != np.complex64 and data.dtype != np.complex128:
            if data.shape[-1] == 2:  # I/Q as separate channels
                data = data[..., 0] + 1j * data[..., 1]
            else:
                raise ValueError("Data must be complex or have I/Q as last dimension")
        
        self.iq_data = data.astype(np.complex64)
        logger.info(f"Loaded {len(self.iq_data)} IQ samples at {self.sample_rate/1e6:.1f} MSPS")
    
    def load_iq_file(self, filename: str, file_format: str = 'complex64') -> None:
        """
        Load IQ data from file.
        
        Args:
            filename: Path to the IQ data file
            file_format: Format of the file ('complex64', 'complex128', 'interleaved_float32')
        """
        try:
            if file_format == 'complex64':
                data = np.fromfile(filename, dtype=np.complex64)
            elif file_format == 'complex128':
                data = np.fromfile(filename, dtype=np.complex128)
            elif file_format == 'interleaved_float32':
                raw_data = np.fromfile(filename, dtype=np.float32)
                data = raw_data[::2] + 1j * raw_data[1::2]
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            self.load_iq_data(data)
            
        except Exception as e:
            logger.error(f"Failed to load IQ file {filename}: {e}")
            raise
    
    def compute_spectrogram(self, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram of the IQ data for frequency hopping analysis.
        
        Args:
            nperseg: Length of each segment for STFT
            
        Returns:
            Tuple of (frequencies, time, spectrogram_magnitude)
        """
        if self.iq_data is None:
            raise ValueError("No IQ data loaded")
        
        if nperseg is None:
            nperseg = self.fft_size
        
        # Compute Short-Time Fourier Transform
        f, t, Zxx = signal.stft(
            self.iq_data,
            fs=self.sample_rate,
            window=self.window,
            nperseg=nperseg,
            noverlap=int(nperseg * self.overlap),
            return_onesided=False,
            boundary=None
        )
        
        # Convert to magnitude and shift zero frequency to center
        self.spectrogram = np.fft.fftshift(np.abs(Zxx), axes=0)
        self.frequency_bins = np.fft.fftshift(f) + self.center_freq
        self.time_bins = t
        
        logger.info(f"Computed spectrogram: {self.spectrogram.shape[0]} freq bins x {self.spectrogram.shape[1]} time bins")
        
        return self.frequency_bins, self.time_bins, self.spectrogram
    
    @staticmethod
    @jit(nopython=True)
    def _detect_hops_threshold(spectrogram: np.ndarray, threshold_factor: float = 3.0) -> np.ndarray:
        """
        Detect frequency hops using threshold-based method (numba-optimized).
        
        Args:
            spectrogram: 2D spectrogram magnitude array
            threshold_factor: Factor above noise floor for detection
            
        Returns:
            Binary array indicating hop presence
        """
        # Estimate noise floor
        noise_floor = np.median(spectrogram)
        threshold = noise_floor * threshold_factor
        
        # Detect peaks above threshold
        hop_mask = spectrogram > threshold
        
        return hop_mask
    
    def detect_frequency_hops(self, method: str = 'threshold', **kwargs) -> List[Tuple[float, float, float]]:
        """
        Detect frequency hopping pattern in the spectrogram.
        
        Args:
            method: Detection method ('threshold', 'clustering', 'template')
            **kwargs: Additional parameters for the detection method
            
        Returns:
            List of (frequency, start_time, duration) tuples for each detected hop
        """
        if self.spectrogram is None:
            self.compute_spectrogram()
        
        hops = []
        
        if method == 'threshold':
            threshold_factor = kwargs.get('threshold_factor', 3.0)
            min_duration = kwargs.get('min_duration', 1e-3)  # 1ms minimum
            
            # Use the JIT-compiled function
            hop_mask = self._detect_hops_threshold(self.spectrogram, threshold_factor)
            
            # Find connected components (hops)
            for t_idx in range(hop_mask.shape[1]):
                active_freqs = np.where(hop_mask[:, t_idx])[0]
                
                if len(active_freqs) > 0:
                    # Group nearby frequencies
                    freq_groups = self._group_frequencies(active_freqs)
                    
                    for group in freq_groups:
                        freq_idx = int(np.mean(group))
                        frequency = self.frequency_bins[freq_idx]
                        start_time = self.time_bins[t_idx]
                        
                        # Find duration of this hop
                        duration = self._find_hop_duration(hop_mask, freq_idx, t_idx)
                        
                        if duration >= min_duration:
                            hops.append((frequency, start_time, duration))
        
        elif method == 'clustering':
            # Use K-means clustering to identify hop frequencies
            n_hops = kwargs.get('n_hops', 50)
            
            # Find peak locations
            peak_indices = self._find_spectral_peaks()
            
            if len(peak_indices) > n_hops:
                # Cluster the peaks
                kmeans = KMeans(n_clusters=n_hops, random_state=42)
                peak_freqs = self.frequency_bins[peak_indices[:, 0]]
                peak_times = self.time_bins[peak_indices[:, 1]]
                
                features = np.column_stack([peak_freqs, peak_times])
                clusters = kmeans.fit_predict(features)
                
                # Convert clusters to hops
                for cluster_id in range(n_hops):
                    cluster_mask = clusters == cluster_id
                    if np.any(cluster_mask):
                        cluster_freqs = peak_freqs[cluster_mask]
                        cluster_times = peak_times[cluster_mask]
                        
                        frequency = np.mean(cluster_freqs)
                        start_time = np.min(cluster_times)
                        duration = np.max(cluster_times) - start_time
                        
                        hops.append((frequency, start_time, duration))
        
        # Sort hops by start time
        hops = sorted(hops, key=lambda x: x[1])
        
        # Store hop information
        self.hop_frequencies = [h[0] for h in hops]
        self.hop_times = [h[1] for h in hops]
        
        if len(hops) > 1:
            hop_intervals = np.diff([h[1] for h in hops])
            self.hop_duration = np.median(hop_intervals)
        
        logger.info(f"Detected {len(hops)} frequency hops")
        if self.hop_duration:
            logger.info(f"Estimated hop duration: {self.hop_duration*1000:.2f} ms")
        
        return hops
    
    def _group_frequencies(self, freq_indices: np.ndarray, max_gap: int = 3) -> List[List[int]]:
        """Group nearby frequency indices into separate hops."""
        if len(freq_indices) == 0:
            return []
        
        groups = []
        current_group = [freq_indices[0]]
        
        for i in range(1, len(freq_indices)):
            if freq_indices[i] - freq_indices[i-1] <= max_gap:
                current_group.append(freq_indices[i])
            else:
                groups.append(current_group)
                current_group = [freq_indices[i]]
        
        groups.append(current_group)
        return groups
    
    def _find_hop_duration(self, hop_mask: np.ndarray, freq_idx: int, start_t_idx: int) -> float:
        """Find the duration of a hop starting at given frequency and time indices."""
        duration_samples = 1
        
        for t_idx in range(start_t_idx + 1, hop_mask.shape[1]):
            if hop_mask[freq_idx, t_idx]:
                duration_samples += 1
            else:
                break
        
        duration = duration_samples * (self.time_bins[1] - self.time_bins[0])
        return duration
    
    def _find_spectral_peaks(self, prominence: float = 0.1) -> np.ndarray:
        """Find spectral peaks in the spectrogram."""
        peaks = []
        
        for t_idx in range(self.spectrogram.shape[1]):
            spectrum = self.spectrogram[:, t_idx]
            peak_indices, _ = signal.find_peaks(spectrum, prominence=prominence * np.max(spectrum))
            
            for peak_idx in peak_indices:
                peaks.append([peak_idx, t_idx])
        
        return np.array(peaks)
    
    def synchronize_hops(self, reference_pattern: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Synchronize with the frequency hopping pattern.
        
        Args:
            reference_pattern: Known hopping pattern frequencies for synchronization
            
        Returns:
            Dictionary containing synchronization results
        """
        if not self.hop_frequencies:
            raise ValueError("No frequency hops detected. Run detect_frequency_hops() first.")
        
        sync_info = {
            'synchronized': False,
            'offset': 0,
            'confidence': 0.0,
            'pattern_length': 0
        }
        
        if reference_pattern is not None:
            # Cross-correlate detected pattern with reference
            detected_pattern = np.array(self.hop_frequencies)
            reference = np.array(reference_pattern)
            
            # Normalize patterns
            detected_norm = (detected_pattern - np.mean(detected_pattern)) / np.std(detected_pattern)
            reference_norm = (reference - np.mean(reference)) / np.std(reference)
            
            # Find best alignment
            correlation = np.correlate(detected_norm, reference_norm, mode='full')
            best_offset = np.argmax(np.abs(correlation)) - len(reference) + 1
            
            sync_info['synchronized'] = True
            sync_info['offset'] = best_offset
            sync_info['confidence'] = np.max(np.abs(correlation)) / len(reference)
            sync_info['pattern_length'] = len(reference)
            
        else:
            # Auto-detect pattern repetition
            pattern_length = self._detect_pattern_repetition()
            if pattern_length > 0:
                sync_info['synchronized'] = True
                sync_info['pattern_length'] = pattern_length
                sync_info['confidence'] = 0.8  # Estimated confidence
        
        logger.info(f"Synchronization: {sync_info}")
        return sync_info
    
    def _detect_pattern_repetition(self) -> int:
        """Detect if the frequency hopping pattern repeats."""
        if len(self.hop_frequencies) < 4:
            return 0
        
        # Try different pattern lengths
        for pattern_len in range(2, len(self.hop_frequencies) // 2):
            # Check if pattern repeats
            pattern = self.hop_frequencies[:pattern_len]
            matches = 0
            
            for start_idx in range(pattern_len, len(self.hop_frequencies) - pattern_len, pattern_len):
                segment = self.hop_frequencies[start_idx:start_idx + pattern_len]
                
                # Compare with tolerance
                if self._patterns_match(pattern, segment, tolerance=1e3):  # 1 kHz tolerance
                    matches += 1
            
            # If we found enough matches, this is likely the pattern length
            if matches >= 2:
                return pattern_len
        
        return 0
    
    def _patterns_match(self, pattern1: List[float], pattern2: List[float], tolerance: float) -> bool:
        """Check if two frequency patterns match within tolerance."""
        if len(pattern1) != len(pattern2):
            return False
        
        for f1, f2 in zip(pattern1, pattern2):
            if abs(f1 - f2) > tolerance:
                return False
        
        return True
    
    def demodulate_hop(self, frequency: float, start_time: float, duration: float, 
                      modulation: str = 'fsk') -> np.ndarray:
        """
        Demodulate a single frequency hop.
        
        Args:
            frequency: Center frequency of the hop
            start_time: Start time of the hop
            duration: Duration of the hop
            modulation: Modulation type ('fsk', 'psk', 'ask')
            
        Returns:
            Demodulated data samples
        """
        # Extract time segment
        start_sample = int(start_time * self.sample_rate)
        duration_samples = int(duration * self.sample_rate)
        end_sample = start_sample + duration_samples
        
        if end_sample > len(self.iq_data):
            end_sample = len(self.iq_data)
            duration_samples = end_sample - start_sample
        
        hop_data = self.iq_data[start_sample:end_sample]
        
        # Frequency shift to baseband
        freq_offset = frequency - self.center_freq
        t = np.arange(len(hop_data)) / self.sample_rate
        lo_signal = np.exp(-2j * np.pi * freq_offset * t)
        baseband_data = hop_data * lo_signal
        
        # Low-pass filter
        nyquist = self.sample_rate / 2
        cutoff = self.demod_bandwidth / 2
        b, a = butter(self.filter_order, cutoff / nyquist, btype='low')
        filtered_data = filtfilt(b, a, baseband_data)
        
        # Demodulate based on modulation type
        if modulation.lower() == 'fsk':
            demod_data = self._demodulate_fsk(filtered_data)
        elif modulation.lower() == 'psk':
            demod_data = self._demodulate_psk(filtered_data)
        elif modulation.lower() == 'ask':
            demod_data = self._demodulate_ask(filtered_data)
        else:
            raise ValueError(f"Unsupported modulation type: {modulation}")
        
        return demod_data
    
    def _demodulate_fsk(self, data: np.ndarray) -> np.ndarray:
        """Demodulate FSK signal using frequency discrimination."""
        # Calculate instantaneous frequency using complex data directly
        if np.iscomplexobj(data):
            # For complex data, use the phase directly
            instantaneous_phase = np.unwrap(np.angle(data))
        else:
            # For real data, use Hilbert transform
            analytic_signal = hilbert(data)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Calculate instantaneous frequency
        instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi) * self.sample_rate
        
        # Convert frequency deviation to binary data
        freq_threshold = np.median(instantaneous_freq)
        binary_data = (instantaneous_freq > freq_threshold).astype(int)
        
        return binary_data
    
    def _demodulate_psk(self, data: np.ndarray) -> np.ndarray:
        """Demodulate PSK signal using phase detection."""
        # Calculate instantaneous phase
        instantaneous_phase = np.angle(data)
        
        # Differential decoding for DPSK
        phase_diff = np.diff(np.unwrap(instantaneous_phase))
        
        # Convert phase difference to binary data
        phase_threshold = 0  # For BPSK
        binary_data = (phase_diff > phase_threshold).astype(int)
        
        return binary_data
    
    def _demodulate_ask(self, data: np.ndarray) -> np.ndarray:
        """Demodulate ASK signal using amplitude detection."""
        # Calculate instantaneous amplitude
        amplitude = np.abs(data)
        
        # Smooth the amplitude
        window_size = max(3, min(len(amplitude) // 10, 51))  # Ensure odd number >= 3
        if len(amplitude) > window_size:
            # Use simple moving average if savgol_filter fails
            try:
                smoothed_amplitude = signal.savgol_filter(amplitude, window_size | 1, 3)  # Ensure odd
            except ValueError:
                # Fallback to moving average
                kernel = np.ones(window_size) / window_size
                smoothed_amplitude = np.convolve(amplitude, kernel, mode='same')
        else:
            smoothed_amplitude = amplitude
        
        # Convert amplitude to binary data
        amp_threshold = np.median(smoothed_amplitude)
        binary_data = (smoothed_amplitude > amp_threshold).astype(int)
        
        return binary_data
    
    def decode_all_hops(self, modulation: str = 'fsk') -> List[np.ndarray]:
        """
        Decode all detected frequency hops.
        
        Args:
            modulation: Modulation type for all hops
            
        Returns:
            List of demodulated data arrays for each hop
        """
        if not self.hop_frequencies:
            raise ValueError("No frequency hops detected. Run detect_frequency_hops() first.")
        
        decoded_data = []
        
        for i, (freq, start_time) in enumerate(zip(self.hop_frequencies, self.hop_times)):
            # Estimate duration
            if i < len(self.hop_times) - 1:
                duration = self.hop_times[i + 1] - start_time
            else:
                duration = self.hop_duration if self.hop_duration else 10e-3  # 10ms default
            
            try:
                hop_data = self.demodulate_hop(freq, start_time, duration, modulation)
                decoded_data.append(hop_data)
                
            except Exception as e:
                logger.warning(f"Failed to decode hop {i} at {freq/1e6:.3f} MHz: {e}")
                decoded_data.append(np.array([]))
        
        logger.info(f"Successfully decoded {len([d for d in decoded_data if len(d) > 0])} out of {len(decoded_data)} hops")
        
        return decoded_data
    
    def plot_spectrogram(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot the spectrogram with detected hops highlighted."""
        if self.spectrogram is None:
            self.compute_spectrogram()
        
        plt.figure(figsize=figsize)
        
        # Convert to dB scale for plotting
        spec_db = 20 * np.log10(self.spectrogram + 1e-10)
        
        plt.pcolormesh(
            self.time_bins * 1000,  # Convert to ms
            self.frequency_bins / 1e6,  # Convert to MHz
            spec_db,
            shading='gouraud',
            cmap='viridis'
        )
        
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.title('Frequency Hopping Signal Spectrogram')
        
        # Highlight detected hops
        if self.hop_frequencies and self.hop_times:
            for freq, time in zip(self.hop_frequencies, self.hop_times):
                plt.plot(time * 1000, freq / 1e6, 'ro', markersize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename: str) -> None:
        """Save analysis results to HDF5 file."""
        with h5py.File(filename, 'w') as f:
            # Save parameters
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['center_freq'] = self.center_freq
            
            # Save raw data
            if self.iq_data is not None:
                f.create_dataset('iq_data', data=self.iq_data)
            
            # Save analysis results
            if self.spectrogram is not None:
                f.create_dataset('spectrogram', data=self.spectrogram)
                f.create_dataset('frequency_bins', data=self.frequency_bins)
                f.create_dataset('time_bins', data=self.time_bins)
            
            if self.hop_frequencies:
                f.create_dataset('hop_frequencies', data=self.hop_frequencies)
                f.create_dataset('hop_times', data=self.hop_times)
                
                if self.hop_duration:
                    f.attrs['hop_duration'] = self.hop_duration
        
        logger.info(f"Results saved to {filename}")


# Utility functions
def generate_test_fhss_signal(sample_rate: float = 2e6, duration: float = 0.1, 
                             hop_rate: float = 1000, num_frequencies: int = 10,
                             modulation: str = 'fsk') -> Tuple[np.ndarray, List[float]]:
    """
    Generate a test frequency hopping spread spectrum signal.
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Signal duration in seconds
        hop_rate: Hop rate in Hz
        num_frequencies: Number of different hop frequencies
        modulation: Modulation type ('fsk', 'psk', 'ask')
        
    Returns:
        Tuple of (IQ signal, hop frequencies used)
    """
    num_samples = int(duration * sample_rate)
    hop_duration = 1.0 / hop_rate
    samples_per_hop = int(hop_duration * sample_rate)
    
    # Generate hop frequencies
    center_freq = 0  # Baseband
    freq_spacing = 50e3  # 50 kHz spacing
    hop_frequencies = [(i - num_frequencies // 2) * freq_spacing for i in range(num_frequencies)]
    
    signal_data = np.zeros(num_samples, dtype=np.complex64)
    t = np.arange(num_samples) / sample_rate
    
    # Generate data bits
    data_rate = 1000  # 1000 bps
    samples_per_bit = int(sample_rate / data_rate)
    data_bits = np.random.randint(0, 2, size=int(duration * data_rate))
    
    current_sample = 0
    used_frequencies = []
    
    while current_sample < num_samples:
        # Choose random frequency
        hop_freq = np.random.choice(hop_frequencies)
        used_frequencies.append(hop_freq)
        
        # Generate signal for this hop
        hop_samples = min(samples_per_hop, num_samples - current_sample)
        t_hop = t[current_sample:current_sample + hop_samples]
        
        if modulation == 'fsk':
            # FSK with ±5kHz deviation
            deviation = 5e3
            bit_idx = current_sample // samples_per_bit
            if bit_idx < len(data_bits):
                freq_offset = deviation if data_bits[bit_idx] else -deviation
            else:
                freq_offset = 0
            
            hop_signal = np.exp(2j * np.pi * (hop_freq + freq_offset) * t_hop)
            
        elif modulation == 'psk':
            # BPSK
            bit_idx = current_sample // samples_per_bit
            if bit_idx < len(data_bits):
                phase = np.pi if data_bits[bit_idx] else 0
            else:
                phase = 0
            
            hop_signal = np.exp(2j * np.pi * hop_freq * t_hop + 1j * phase)
            
        elif modulation == 'ask':
            # ASK
            bit_idx = current_sample // samples_per_bit
            if bit_idx < len(data_bits):
                amplitude = 1.0 if data_bits[bit_idx] else 0.3
            else:
                amplitude = 1.0
            
            hop_signal = amplitude * np.exp(2j * np.pi * hop_freq * t_hop)
        
        signal_data[current_sample:current_sample + hop_samples] = hop_signal
        current_sample += hop_samples
    
    # Add noise
    noise_power = 0.01
    noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * np.sqrt(noise_power)
    signal_data += noise
    
    return signal_data, used_frequencies


if __name__ == "__main__":
    # Example usage
    print("Frequency Hopping Decoder - Example Usage")
    print("=" * 50)
    
    # Create decoder instance
    decoder = FrequencyHoppingDecoder(sample_rate=2e6, center_freq=915e6)
    
    # Generate test signal
    print("Generating test FHSS signal...")
    test_signal, test_freqs = generate_test_fhss_signal(
        sample_rate=2e6,
        duration=0.05,  # 50ms
        hop_rate=1000,  # 1kHz hop rate
        num_frequencies=10,
        modulation='fsk'
    )
    
    # Load the test data
    decoder.load_iq_data(test_signal)
    
    # Analyze the signal
    print("Computing spectrogram...")
    decoder.compute_spectrogram()
    
    print("Detecting frequency hops...")
    hops = decoder.detect_frequency_hops(method='threshold', threshold_factor=2.0)
    
    print(f"Detected {len(hops)} hops:")
    for i, (freq, time, duration) in enumerate(hops[:10]):  # Show first 10
        print(f"  Hop {i+1}: {freq/1e3:.1f} kHz at {time*1000:.1f} ms (duration: {duration*1000:.1f} ms)")
    
    # Synchronize
    print("Synchronizing with hop pattern...")
    sync_info = decoder.synchronize_hops()
    
    # Decode hops
    print("Decoding frequency hops...")
    decoded_data = decoder.decode_all_hops(modulation='fsk')
    
    print(f"Successfully decoded {len([d for d in decoded_data if len(d) > 0])} hops")
    
    # Plot results
    print("Plotting spectrogram...")
    decoder.plot_spectrogram()
    
    print("Analysis complete!")