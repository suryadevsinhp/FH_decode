// Frequency Hopping Decoder - Frontend JavaScript

class FrequencyHoppingApp {
    constructor() {
        this.socket = null;
        this.currentJobId = null;
        this.isProcessing = false;
        
        this.initializeSocket();
        this.bindEvents();
        this.loadRecentJobs();
    }
    
    initializeSocket() {
        // Initialize Socket.IO connection
        this.socket = io();
        
        // Connection events
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        // Processing events
        this.socket.on('progress_update', (data) => {
            if (data.job_id === this.currentJobId) {
                this.updateProgress(data.progress, data.message);
            }
        });
        
        this.socket.on('processing_complete', (data) => {
            if (data.job_id === this.currentJobId) {
                this.handleProcessingComplete(data.result);
            }
        });
        
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.showError('Connection error: ' + error);
        });
    }
    
    bindEvents() {
        // Form submission
        const uploadForm = document.getElementById('uploadForm');
        uploadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFileUpload();
        });
        
        // File input change
        const fileInput = document.getElementById('fileInput');
        fileInput.addEventListener('change', (e) => {
            this.validateFileInput(e.target);
        });
        
        // Window events
        window.addEventListener('beforeunload', (e) => {
            if (this.isProcessing) {
                e.preventDefault();
                e.returnValue = 'Processing is in progress. Are you sure you want to leave?';
            }
        });
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            statusElement.className = 'badge bg-success connected';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            statusElement.className = 'badge bg-danger disconnected';
        }
    }
    
    validateFileInput(input) {
        const file = input.files[0];
        if (!file) return;
        
        // Check file extension
        if (!file.name.toLowerCase().endsWith('.bin')) {
            this.showError('Please select a .bin file');
            input.value = '';
            return;
        }
        
        // Check file size (max 500MB)
        const maxSize = 500 * 1024 * 1024; // 500MB
        if (file.size > maxSize) {
            this.showError('File size must be less than 500MB');
            input.value = '';
            return;
        }
        
        // Show file info
        const fileSize = this.formatFileSize(file.size);
        console.log(`Selected file: ${file.name} (${fileSize})`);
    }
    
    async handleFileUpload() {
        if (this.isProcessing) {
            this.showError('Processing is already in progress');
            return;
        }
        
        const formData = new FormData();
        const fileInput = document.getElementById('fileInput');
        const centerFreq = document.getElementById('centerFreq');
        const bandwidth = document.getElementById('bandwidth');
        const sampleRate = document.getElementById('sampleRate');
        
        // Validate inputs
        if (!fileInput.files[0]) {
            this.showError('Please select a file');
            return;
        }
        
        if (!this.validateParameters(centerFreq.value, bandwidth.value, sampleRate.value)) {
            return;
        }
        
        // Prepare form data
        formData.append('file', fileInput.files[0]);
        formData.append('center_freq', centerFreq.value);
        formData.append('bandwidth', bandwidth.value);
        formData.append('sample_rate', sampleRate.value);
        
        try {
            this.setProcessingState(true);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.currentJobId = result.job_id;
                this.showProcessingStatus(result.job_id);
                this.updateProgress(0, 'Upload completed, starting processing...');
            } else {
                throw new Error(result.error || 'Upload failed');
            }
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Upload failed: ' + error.message);
            this.setProcessingState(false);
        }
    }
    
    validateParameters(centerFreq, bandwidth, sampleRate) {
        const freq = parseFloat(centerFreq);
        const bw = parseFloat(bandwidth);
        const sr = parseFloat(sampleRate);
        
        if (isNaN(freq) || freq <= 0) {
            this.showError('Center frequency must be a positive number');
            return false;
        }
        
        if (isNaN(bw) || bw <= 0) {
            this.showError('Bandwidth must be a positive number');
            return false;
        }
        
        if (isNaN(sr) || sr <= 0) {
            this.showError('Sample rate must be a positive number');
            return false;
        }
        
        if (bw > sr) {
            this.showError('Bandwidth cannot be greater than sample rate');
            return false;
        }
        
        return true;
    }
    
    setProcessingState(processing) {
        this.isProcessing = processing;
        const processBtn = document.getElementById('processBtn');
        const form = document.getElementById('uploadForm');
        
        if (processing) {
            processBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            processBtn.disabled = true;
            form.classList.add('processing');
        } else {
            processBtn.innerHTML = '<i class="fas fa-cogs me-2"></i>Process File';
            processBtn.disabled = false;
            form.classList.remove('processing');
        }
    }
    
    showProcessingStatus(jobId) {
        const statusCard = document.getElementById('statusCard');
        const jobIdElement = document.getElementById('jobId');
        const welcomeCard = document.getElementById('welcomeCard');
        const resultsCard = document.getElementById('resultsCard');
        
        jobIdElement.textContent = jobId;
        statusCard.style.display = 'block';
        statusCard.classList.add('card-appear');
        welcomeCard.style.display = 'none';
        resultsCard.style.display = 'none';
    }
    
    updateProgress(progress, message) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const statusMessage = document.getElementById('statusMessage');
        
        progressBar.style.width = progress + '%';
        progressBar.setAttribute('aria-valuenow', progress);
        progressText.textContent = Math.round(progress) + '%';
        statusMessage.textContent = message;
        
        // Update progress bar color based on progress
        if (progress >= 100) {
            progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
            progressBar.classList.add('bg-success');
        }
    }
    
    handleProcessingComplete(result) {
        this.setProcessingState(false);
        
        if (result.success) {
            this.showResults(result);
            this.loadRecentJobs(); // Refresh job list
        } else {
            this.showError('Processing failed: ' + (result.error || 'Unknown error'));
        }
        
        // Hide status card after a delay
        setTimeout(() => {
            document.getElementById('statusCard').style.display = 'none';
        }, 3000);
    }
    
    showResults(result) {
        const resultsCard = document.getElementById('resultsCard');
        const audioSection = document.getElementById('audioSection');
        const spectrogramSection = document.getElementById('spectrogramSection');
        const statsSection = document.getElementById('statsSection');
        
        // Show results card
        resultsCard.style.display = 'block';
        resultsCard.classList.add('card-appear');
        
        // Setup audio player
        if (result.audio_file) {
            const audioPlayer = document.getElementById('audioPlayer');
            const downloadBtn = document.getElementById('downloadAudioBtn');
            
            audioPlayer.src = `/download/${result.audio_file}`;
            downloadBtn.href = `/download/${result.audio_file}`;
            downloadBtn.download = result.audio_file;
            
            audioSection.style.display = 'block';
        }
        
        // Show spectrogram
        if (result.plot_file) {
            const spectrogramImage = document.getElementById('spectrogramImage');
            spectrogramImage.src = `/plot/${result.plot_file}`;
            spectrogramSection.style.display = 'block';
        }
        
        // Show statistics
        if (result.num_hops !== undefined) {
            document.getElementById('numHops').textContent = result.num_hops;
            document.getElementById('centerFreqDisplay').textContent = 
                (parseFloat(document.getElementById('centerFreq').value)).toFixed(3);
            document.getElementById('bandwidthDisplay').textContent = 
                (parseFloat(document.getElementById('bandwidth').value)).toFixed(3);
            
            statsSection.style.display = 'block';
        }
        
        // Hide welcome card
        document.getElementById('welcomeCard').style.display = 'none';
    }
    
    async loadRecentJobs() {
        try {
            const response = await fetch('/api/jobs');
            const jobs = await response.json();
            
            if (response.ok) {
                this.displayRecentJobs(jobs);
            } else {
                console.warn('Failed to load recent jobs:', jobs.error);
            }
        } catch (error) {
            console.error('Error loading recent jobs:', error);
        }
    }
    
    displayRecentJobs(jobs) {
        const container = document.getElementById('recentJobs');
        
        if (jobs.length === 0) {
            container.innerHTML = `
                <div class="text-muted text-center">
                    <i class="fas fa-clock me-2"></i>No recent jobs
                </div>
            `;
            return;
        }
        
        const jobsHtml = jobs.map(job => {
            const statusClass = job.status === 'completed' ? 'completed' : 
                               job.status === 'failed' ? 'failed' : 'processing';
            
            const timestamp = new Date(job.timestamp).toLocaleString();
            
            return `
                <div class="job-item">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <small class="text-muted">${job.job_id}</small>
                            <div class="small">${timestamp}</div>
                            ${job.num_hops ? `<div class="small text-info">${job.num_hops} hops detected</div>` : ''}
                        </div>
                        <span class="job-status ${statusClass}">${job.status}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = jobsHtml;
    }
    
    showError(message) {
        const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
        document.getElementById('errorMessage').textContent = message;
        errorModal.show();
        
        // Also log to console
        console.error('App Error:', message);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new FrequencyHoppingApp();
    
    // Make app instance globally available for debugging
    window.fhApp = app;
    
    console.log('Frequency Hopping Decoder initialized');
});

// Service Worker registration for offline support (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => {
                console.log('SW registered: ', registration);
            })
            .catch(registrationError => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}