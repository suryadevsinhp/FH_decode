// Frequency Hopping Decoder - Frontend JavaScript (Optimized)

// Performance configuration
const PERF_CONFIG = {
    RECONNECT_DELAY: 3000,       // WebSocket reconnection delay
    MAX_RECONNECT_ATTEMPTS: 5,   // Maximum reconnection attempts
    DOM_CACHE_CLEANUP: 30000,    // DOM cache cleanup interval
    UPDATE_THROTTLE_MS: 100,     // Throttle UI updates
    REQUEST_TIMEOUT: 10000       // Request timeout
};

// Performance: DOM element cache
const DOM_CACHE = {};
function cacheDOMElements() {
    DOM_CACHE.uploadForm = document.getElementById('uploadForm');
    DOM_CACHE.fileInput = document.getElementById('fileInput');
    DOM_CACHE.processBtn = document.getElementById('processBtn');
    DOM_CACHE.progressBar = document.getElementById('progressBar');
    DOM_CACHE.progressText = document.getElementById('progressText');
    DOM_CACHE.statusMessage = document.getElementById('statusMessage');
    DOM_CACHE.connectionStatus = document.getElementById('connectionStatus');
    DOM_CACHE.recentJobs = document.getElementById('recentJobs');
}

// Performance: Throttle function
function createThrottledFunction(func, delay) {
    let lastCall = 0;
    let timeoutId = null;
    return function(...args) {
        const now = Date.now();
        if (now - lastCall >= delay) {
            func.apply(this, args);
            lastCall = now;
        } else {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => {
                func.apply(this, args);
                lastCall = Date.now();
            }, delay - (now - lastCall));
        }
    };
}

class FrequencyHoppingApp {
    constructor() {
        this.socket = null;
        this.currentJobId = null;
        this.isProcessing = false;
        this.reconnectAttempts = 0;
        this.isConnected = false;
        
        // Performance: Create throttled functions
        this.throttledProgressUpdate = createThrottledFunction(
            this.updateProgress.bind(this), 
            PERF_CONFIG.UPDATE_THROTTLE_MS
        );
        
        // Performance: Cache DOM elements
        cacheDOMElements();
        
        this.initializeSocket();
        this.bindEvents();
        this.loadRecentJobs();
        this.setupCleanup();
    }
    
    // Performance: Setup cleanup and memory management
    setupCleanup() {
        // Performance: Periodic cleanup
        setInterval(() => {
            try {
                if (window.gc) window.gc(); // Force garbage collection if available
            } catch (e) {
                console.warn('GC not available');
            }
        }, PERF_CONFIG.DOM_CACHE_CLEANUP);
        
        // Performance: Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }

    // Performance: Optimized socket initialization with reconnection logic
    initializeSocket() {
        try {
            // Initialize Socket.IO connection
            this.socket = io({
                timeout: PERF_CONFIG.REQUEST_TIMEOUT,
                forceNew: true
            });
            
            // Connection events
            this.socket.on('connect', () => {
                console.log('Connected to server');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
            });
            
            this.socket.on('disconnect', (reason) => {
                console.log('Disconnected from server:', reason);
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.handleReconnection();
            });
            
            // Processing events with throttling
            this.socket.on('progress_update', (data) => {
                if (data.job_id === this.currentJobId) {
                    this.throttledProgressUpdate(data.progress, data.message);
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
            
            this.socket.on('connect_error', (error) => {
                console.error('Connection error:', error);
                this.handleReconnection();
            });
            
        } catch (error) {
            console.error('Socket initialization error:', error);
            this.showError('Failed to initialize connection');
        }
    }

    // Performance: Handle reconnection with exponential backoff
    handleReconnection() {
        if (this.reconnectAttempts < PERF_CONFIG.MAX_RECONNECT_ATTEMPTS) {
            this.reconnectAttempts++;
            const delay = Math.min(PERF_CONFIG.RECONNECT_DELAY * Math.pow(2, this.reconnectAttempts - 1), 30000);
            
            console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
            
            setTimeout(() => {
                if (!this.isConnected) {
                    this.initializeSocket();
                }
            }, delay);
        } else {
            this.showError('Connection lost. Please refresh the page.');
        }
    }
    
    // Performance: Optimized event binding with cached DOM elements
    bindEvents() {
        try {
            // Form submission
            const uploadForm = DOM_CACHE.uploadForm || document.getElementById('uploadForm');
            if (uploadForm) {
                uploadForm.addEventListener('submit', (e) => {
                    e.preventDefault();
                    this.handleFileUpload();
                });
            }
            
            // File input change
            const fileInput = DOM_CACHE.fileInput || document.getElementById('fileInput');
            if (fileInput) {
                fileInput.addEventListener('change', (e) => {
                    this.validateFileInput(e.target);
                });
            }
            
            // Window events
            window.addEventListener('beforeunload', (e) => {
                if (this.isProcessing) {
                    e.preventDefault();
                    e.returnValue = 'Processing is in progress. Are you sure you want to leave?';
                }
            });
        } catch (error) {
            console.error('Event binding error:', error);
        }
    }
    
    // Performance: Optimized connection status update with cached DOM
    updateConnectionStatus(connected) {
        try {
            const statusElement = DOM_CACHE.connectionStatus || document.getElementById('connectionStatus');
            if (statusElement) {
                if (connected) {
                    statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
                    statusElement.className = 'badge bg-success connected';
                } else {
                    statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
                    statusElement.className = 'badge bg-danger disconnected';
                }
            }
        } catch (error) {
            console.warn('Connection status update error:', error);
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

    // Performance: Cleanup method
    cleanup() {
        try {
            // Disconnect socket
            if (this.socket) {
                this.socket.disconnect();
                this.socket = null;
            }
            
            // Clear intervals and timeouts
            this.isProcessing = false;
            this.currentJobId = null;
            this.reconnectAttempts = 0;
            this.isConnected = false;
            
            // Clear DOM cache
            Object.keys(DOM_CACHE).forEach(key => {
                DOM_CACHE[key] = null;
            });
            
            console.log('Application cleanup completed');
        } catch (error) {
            console.warn('Cleanup error:', error);
        }
    }
}

// Performance: Optimized application initialization
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Performance: Cache DOM elements first
        cacheDOMElements();
        
        const app = new FrequencyHoppingApp();
        
        // Make app instance globally available for debugging
        window.fhApp = app;
        
        // Performance: Setup global error handling
        window.addEventListener('error', (event) => {
            console.error('Global error:', event.error);
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
        });
        
        console.log('Frequency Hopping Decoder initialized (Optimized WebSocket version)');
    } catch (error) {
        console.error('Initialization error:', error);
        alert('Application failed to initialize. Please refresh the page.');
    }
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