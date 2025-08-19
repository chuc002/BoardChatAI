"""
Production monitoring and performance tracking for BoardContinuity MVP
"""

import time
import logging
from functools import wraps
from datetime import datetime
from flask import request, g
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor application performance and resource usage."""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize monitoring with Flask app."""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown_appcontext)
    
    def before_request(self):
        """Record request start time and system metrics."""
        g.start_time = time.time()
        g.request_start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def after_request(self, response):
        """Log request performance metrics."""
        if hasattr(g, 'start_time'):
            duration = (time.time() - g.start_time) * 1000  # milliseconds
            
            # Log slow requests
            if duration > 5000:  # 5 seconds
                logger.warning(f"Slow request: {request.method} {request.path} took {duration:.2f}ms")
            
            # Log performance metrics
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_delta = memory_usage - getattr(g, 'request_start_memory', 0)
            
            logger.info(f"{request.method} {request.path} - {response.status_code} - {duration:.2f}ms - Memory: {memory_usage:.1f}MB (+{memory_delta:.1f}MB)")
        
        return response
    
    def teardown_appcontext(self, error):
        """Clean up request context."""
        if error:
            logger.error(f"Request error: {error}")

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            logger.info(f"Function {func.__name__} completed in {duration:.2f}ms using {memory_used:.1f}MB")
            return result
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Function {func.__name__} failed after {duration:.2f}ms: {str(e)}")
            raise
    
    return wrapper

def get_system_metrics():
    """Get current system metrics."""
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024,
        'disk_percent': psutil.disk_usage('/').percent,
        'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
    }

def check_system_health():
    """Check if system is healthy."""
    metrics = get_system_metrics()
    
    health_status = {
        'healthy': True,
        'warnings': [],
        'metrics': metrics
    }
    
    # Check CPU usage
    if metrics['cpu_percent'] > 90:
        health_status['healthy'] = False
        health_status['warnings'].append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
    elif metrics['cpu_percent'] > 70:
        health_status['warnings'].append(f"Elevated CPU usage: {metrics['cpu_percent']:.1f}%")
    
    # Check memory usage
    if metrics['memory_percent'] > 90:
        health_status['healthy'] = False
        health_status['warnings'].append(f"High memory usage: {metrics['memory_percent']:.1f}%")
    elif metrics['memory_percent'] > 70:
        health_status['warnings'].append(f"Elevated memory usage: {metrics['memory_percent']:.1f}%")
    
    # Check disk usage
    if metrics['disk_percent'] > 95:
        health_status['healthy'] = False
        health_status['warnings'].append(f"Critical disk usage: {metrics['disk_percent']:.1f}%")
    elif metrics['disk_percent'] > 80:
        health_status['warnings'].append(f"High disk usage: {metrics['disk_percent']:.1f}%")
    
    return health_status

# Singleton instance
performance_monitor = PerformanceMonitor()