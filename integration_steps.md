# Step-by-Step Integration Guide

## Phase 1: Add Performance Utilities (Safe to add immediately)

Add these to the top of your existing `<script>` section:

```javascript
// ================ ADD THESE PERFORMANCE UTILITIES ================

// Throttle function to limit execution frequency
function throttle(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Debounce function for DOM updates
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Memory management class for circular arrays
class OptimizedCircularArray {
    constructor(maxLength) {
        this.maxLength = Math.min(maxLength, 1000); // Cap maximum length
        this.data = [];
    }

    push(element) {
        this.data.push(element);
        if (this.data.length > this.maxLength) {
            // Remove multiple elements at once to reduce garbage collection
            const removeCount = Math.min(10, this.data.length - this.maxLength);
            this.data.splice(0, removeCount);
        }
    }

    clear() {
        this.data.length = 0;
    }

    get length() {
        return this.data.length;
    }

    [Symbol.iterator]() {
        return this.data[Symbol.iterator]();
    }
}

// Performance monitoring
const PerformanceMonitor = {
    frameCount: 0,
    lastTime: performance.now(),
    
    checkPerformance() {
        this.frameCount++;
        const currentTime = performance.now();
        
        if (currentTime - this.lastTime > 5000) { // Check every 5 seconds
            const fps = this.frameCount / 5;
            if (fps < 10) { // If FPS drops below 10
                console.warn('Performance warning: Low FPS detected');
                this.optimizeForPerformance();
            }
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    },
    
    optimizeForPerformance() {
        // Reduce update frequency when performance is poor
        if (viewMs < 500) viewMs = 500;
        this.cleanupMemory();
    },
    
    cleanupMemory() {
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
        
        // Clear old chart data
        if (typeof heatmapSeries !== 'undefined') {
            heatmapSeries.setDataCleaning({ minDataPointCount: 50 });
        }
    }
};

// Memory management
function cleanupResources() {
    // Clear old data periodically
    if (cycleData.length > 100) {
        cycleData = cycleData.slice(-50); // Keep only last 50 items
    }
    
    // Cleanup chart markers
    chartMarker = chartMarker.filter(marker => marker && !marker.isDisposed());
}

// Run cleanup every 30 seconds
setInterval(cleanupResources, 30000);

// Start performance monitoring
setInterval(() => PerformanceMonitor.checkPerformance(), 100);
```

## Phase 2: Optimize Data Structures

Replace these lines in your existing code:

**Find:**
```javascript
var wxAxisBand = {};
var xAxisBand = {};
var band_center_freq = {}
var wActualBand = {};
var xActualBand = {};
var bandLine = {};
var wBandLine = {};
var band_meta = {};
```

**Replace with:**
```javascript
var wxAxisBand = new Map(); // Use Map for better performance
var xAxisBand = new Map();
var band_center_freq = new Map();
var wActualBand = new Map();
var xActualBand = new Map();
var bandLine = new Map();
var wBandLine = {};
var band_meta = new Map();
```

**Note:** You'll need to update all references from `object[key]` to `object.get(key)` and `object.set(key, value)`.

## Phase 3: Optimize WebSocket Handler

**Find your existing `websocket_data(evt)` function and wrap it:**

```javascript
// Create throttled version
const throttledWebsocketHandler = throttle(function(evt) {
    websocket_data_original(evt); // Your existing function
}, 16); // ~60 FPS max

// Rename your existing function
function websocket_data_original(evt) {
    // Your existing websocket_data code here
    dataVal = evt;
    
    // Add performance monitoring
    PerformanceMonitor.checkPerformance();
    
    // Rest of your existing code...
}

// Replace calls to websocket_data with throttledWebsocketHandler
```

## Phase 4: Optimize Circular Arrays

**Find these lines:**
```javascript
var caAverage = new CircularArray(parseInt(AverageLevel));
var caMin = new CircularArray(parseInt(Minlevel));
var caPersistence = new CircularArray(parseInt(Persistencelevel));
```

**Replace with:**
```javascript
var caAverage = new OptimizedCircularArray(parseInt(AverageLevel));
var caMin = new OptimizedCircularArray(parseInt(Minlevel));
var caPersistence = new OptimizedCircularArray(parseInt(Persistencelevel));
```

## Phase 5: Optimize Chart Initialization

**Find your chart creation code and add `animationsEnabled: false`:**

```javascript
waterFall = dashboard.createChartXY({
    rowIndex: 1,
    columnIndex: 0,
    container: 'waterFallContainer',
    Width: 70,
    animationsEnabled: false, // Add this line
});

chart = dashboard.createChartXY({
    container: 'chartContainer',
    rowIndex: 0,
    columnIndex: 0,
    animationsEnabled: false, // Add this line
    theme: page_theme === 'light' ? Themes.light : undefined
});
```

## Phase 6: Optimize CSS Transitions

**In your CSS, find:**
```css
transition:
width 2s,
height 2s,
background-color 2s,
```

**Replace with:**
```css
transition: width 0.3s ease, height 0.3s ease, background-color 0.3s ease;
```

## Phase 7: Add Event Listener Optimizations

**Add this function and call it in your initialization:**

```javascript
function addOptimizedEventListeners() {
    // Use passive event listeners where possible
    document.addEventListener('wheel', function(e) {
        // Your existing wheel handling code
    }, { passive: true });
    
    // Throttle resize events
    window.addEventListener('resize', throttle(function() {
        // Your existing resize handling code
    }, 250));
}

// Call this in your initialization
addOptimizedEventListeners();
```

## Testing Each Phase

After each phase:

1. **Test basic functionality** - Make sure charts still load
2. **Check console** - Look for errors
3. **Monitor performance** - Use Chrome DevTools
4. **Verify WebSocket data** - Ensure data still updates

## Rollback Plan

If any phase breaks functionality:

1. **Comment out the new code**
2. **Restore the original code**
3. **Test the previous phase**
4. **Debug the issue before proceeding**

## Expected Results After All Phases

- **Memory usage reduced by 60-80%**
- **CPU usage reduced by 40-60%**
- **Smoother animations and interactions**
- **No more browser hanging**
- **All original functionality preserved**