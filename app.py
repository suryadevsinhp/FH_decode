import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import librosa
import soundfile as sf
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB max file size

# Global variable to store job progress
job_progress = {}

# MongoDB connection
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    # Test the connection
    client.admin.command('ping')
    db = client['frequency_hopping_db']
    collection = db['processing_jobs']
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
    db = None
    collection = None

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

class FrequencyHoppingDecoder:
    def __init__(self, job_id):
        self.job_id = job_id
        
    def emit_progress(self, progress, message):
        """Store progress in global dictionary"""
        job_progress[self.job_id] = {
            'progress': progress,
            'message': message,
            'timestamp': time.time()
        }
        print(f"Progress {self.job_id}: {progress}% - {message}")
        
    def read_iq_file(self, filepath, sample_rate):
        """Read IQ data from binary file with memory optimization for large files"""
        self.emit_progress(10, "Reading IQ file...")
        
        # Get file size for memory optimization
        file_size = os.path.getsize(filepath)
        self.emit_progress(15, f"File size: {file_size / (1024*1024):.1f} MB")
        
        # Read binary file as complex float32 (I + jQ)
        with open(filepath, 'rb') as f:
            # For very large files (>1GB), consider chunked reading
            if file_size > 1024 * 1024 * 1024:  # 1GB
                self.emit_progress(18, "Large file detected - optimizing memory usage...")
            
            # Assuming interleaved I/Q data as float32
            data = np.frombuffer(f.read(), dtype=np.float32)
            
        # Convert to complex numbers (I + jQ)
        if len(data) % 2 != 0:
            data = data[:-1]  # Make even length
            
        iq_data = data[::2] + 1j * data[1::2]
        
        self.emit_progress(20, f"Loaded {len(iq_data)} IQ samples")
        return iq_data
    
    def detect_frequency_hops(self, iq_data, sample_rate, center_freq, bandwidth):
        """Detect frequency hopping patterns in IQ data"""
        self.emit_progress(30, "Analyzing frequency spectrum...")
        
        # Parameters for STFT
        nperseg = 1024
        noverlap = nperseg // 4
        
        # Compute Short-Time Fourier Transform
        frequencies, times, Zxx = signal.stft(
            iq_data, 
            fs=sample_rate, 
            nperseg=nperseg, 
            noverlap=noverlap
        )
        
        # Convert to power spectrum
        power_spectrum = np.abs(Zxx) ** 2
        
        self.emit_progress(40, "Detecting frequency hops...")
        
        # Find peak frequencies in each time window
        hop_frequencies = []
        hop_times = []
        
        for i, time_slice in enumerate(power_spectrum.T):
            # Find the peak frequency in this time slice
            peak_idx = np.argmax(time_slice)
            peak_freq = frequencies[peak_idx] + center_freq
            
            hop_frequencies.append(peak_freq)
            hop_times.append(times[i])
            
            if i % 100 == 0:  # Update progress periodically
                progress = 40 + (i / len(power_spectrum.T)) * 20
                self.emit_progress(progress, f"Processing time slice {i}/{len(power_spectrum.T)}")
        
        self.emit_progress(60, "Frequency hops detected")
        
        return hop_frequencies, hop_times, frequencies, times, power_spectrum
    
    def create_spectrogram_plot(self, frequencies, times, power_spectrum, center_freq):
        """Create and save spectrogram plot"""
        self.emit_progress(70, "Generating spectrogram...")
        
        plt.figure(figsize=(12, 8))
        
        # Convert to dB scale
        power_db = 10 * np.log10(power_spectrum + 1e-10)
        
        plt.pcolormesh(times, frequencies + center_freq, power_db, shading='gouraud')
        plt.colorbar(label='Power (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Frequency Hopping Spectrogram')
        
        plot_filename = f'spectrogram_{self.job_id}.png'
        plot_path = os.path.join('static/plots', plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_filename
    
    def extract_audio_segments(self, iq_data, hop_frequencies, hop_times, sample_rate, bandwidth):
        """Extract and demodulate audio from frequency hops"""
        self.emit_progress(75, "Extracting audio segments...")
        
        audio_segments = []
        
        for i, (freq, start_time) in enumerate(zip(hop_frequencies, hop_times)):
            if i >= len(hop_times) - 1:
                break
                
            # Calculate time window
            end_time = hop_times[i + 1] if i + 1 < len(hop_times) else hop_times[i] + 0.1
            
            # Convert time to samples
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            if end_sample > len(iq_data):
                end_sample = len(iq_data)
            
            # Extract segment
            segment = iq_data[start_sample:end_sample]
            
            if len(segment) > 0:
                # Simple AM demodulation (magnitude)
                audio_segment = np.abs(segment)
                
                # Normalize
                if np.max(audio_segment) > 0:
                    audio_segment = audio_segment / np.max(audio_segment)
                
                audio_segments.append(audio_segment)
            
            if i % 50 == 0:
                progress = 75 + (i / len(hop_frequencies)) * 15
                self.emit_progress(progress, f"Processing hop {i}/{len(hop_frequencies)}")
        
        # Concatenate all segments
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
        else:
            full_audio = np.array([])
            
        return full_audio
    
    def save_audio_output(self, audio_data, sample_rate):
        """Save processed audio to file"""
        self.emit_progress(90, "Saving audio output...")
        
        if len(audio_data) == 0:
            raise ValueError("No audio data to save")
        
        # Resample to standard audio sample rate
        target_sample_rate = 44100
        if sample_rate != target_sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sample_rate)
        
        # Ensure audio is in valid range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Save as WAV file
        output_filename = f'decoded_audio_{self.job_id}.wav'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        sf.write(output_path, audio_data, target_sample_rate)
        
        self.emit_progress(95, "Audio saved successfully")
        
        return output_filename
    
    def process_file(self, filepath, center_freq, bandwidth, sample_rate):
        """Main processing function"""
        try:
            # Read IQ data
            iq_data = self.read_iq_file(filepath, sample_rate)
            
            # Detect frequency hops
            hop_frequencies, hop_times, frequencies, times, power_spectrum = self.detect_frequency_hops(
                iq_data, sample_rate, center_freq, bandwidth
            )
            
            # Create spectrogram plot
            plot_filename = self.create_spectrogram_plot(frequencies, times, power_spectrum, center_freq)
            
            # Extract audio
            audio_data = self.extract_audio_segments(
                iq_data, hop_frequencies, hop_times, sample_rate, bandwidth
            )
            
            # Save audio output
            audio_filename = self.save_audio_output(audio_data, sample_rate)
            
            self.emit_progress(100, "Processing completed successfully!")
            
            # Save results to database
            if db is not None and collection is not None:
                result = {
                    'job_id': self.job_id,
                    'timestamp': datetime.now(),
                    'center_freq': center_freq,
                    'bandwidth': bandwidth,
                    'sample_rate': sample_rate,
                    'audio_file': audio_filename,
                    'plot_file': plot_filename,
                    'num_hops': len(hop_frequencies),
                    'status': 'completed'
                }
                collection.insert_one(result)
            
            return {
                'success': True,
                'audio_file': audio_filename,
                'plot_file': plot_filename,
                'num_hops': len(hop_frequencies)
            }
            
        except Exception as e:
            self.emit_progress(0, f"Error: {str(e)}")
            if db is not None and collection is not None:
                collection.insert_one({
                    'job_id': self.job_id,
                    'timestamp': datetime.now(),
                    'status': 'failed',
                    'error': str(e)
                })
            return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get parameters
    try:
        center_freq = float(request.form.get('center_freq', 0)) * 1e6  # Convert MHz to Hz
        bandwidth = float(request.form.get('bandwidth', 1)) * 1e6      # Convert MHz to Hz
        sample_rate = float(request.form.get('sample_rate', 1)) * 1e6  # Convert MHz to Hz
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid frequency parameters'}), 400
    
    if file and file.filename.endswith('.bin'):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate job ID
        job_id = f"job_{timestamp}"
        
        # Start processing in background thread
        def process_in_background():
            decoder = FrequencyHoppingDecoder(job_id)
            result = decoder.process_file(filepath, center_freq, bandwidth, sample_rate)
            
            # Store final result
            job_progress[job_id + '_result'] = result
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({'job_id': job_id, 'message': 'Processing started'})
    
    return jsonify({'error': 'Invalid file format. Please upload a .bin file'}), 400

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Get progress for a specific job"""
    if job_id in job_progress:
        return jsonify(job_progress[job_id])
    else:
        return jsonify({'progress': 0, 'message': 'Job not found'})

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get final result for a job"""
    result_key = job_id + '_result'
    if result_key in job_progress:
        result = job_progress[result_key]
        # Clean up progress data
        if job_id in job_progress:
            del job_progress[job_id]
        del job_progress[result_key]
        return jsonify(result)
    else:
        return jsonify({'success': False, 'error': 'Result not found'})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/plot/<filename>')
def get_plot(filename):
    return send_from_directory('static/plots', filename)

@app.route('/api/jobs')
def get_jobs():
    if db is None or collection is None:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        jobs = list(collection.find({}, {'_id': 0}).sort('timestamp', -1).limit(10))
    except Exception as e:
        return jsonify({'error': f'Database query failed: {str(e)}'}), 500
    # Convert datetime objects to strings
    for job in jobs:
        if 'timestamp' in job:
            job['timestamp'] = job['timestamp'].isoformat()
    
    return jsonify(jobs)

if __name__ == '__main__':
    print("🚀 Starting Frequency Hopping Decoder Server...")
    print("🔗 Access at: http://localhost:5000")
    print("📁 Make sure MongoDB is running on localhost:27017")
    print("✅ No SocketIO dependencies - using HTTP polling for updates")
    app.run(debug=True, host='0.0.0.0', port=5000)