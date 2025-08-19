"""
Enterprise Monitoring and Performance Optimization
Provides comprehensive monitoring, alerting, and optimization for enterprise-scale operations.
"""

import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os

class EnterpriseMonitor:
    def __init__(self):
        self.performance_history = []
        self.alert_thresholds = {
            'response_time_ms': 5000,
            'memory_increase_mb': 200,
            'error_rate_percent': 5,
            'concurrent_requests': 10
        }
        self.logger = logging.getLogger(__name__)
        
    def monitor_performance(self, func):
        """Decorator to monitor function performance with enterprise metrics"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Calculate metrics
                execution_time = (end_time - start_time) * 1000  # ms
                memory_delta = end_memory - start_memory
                
                # Record performance data
                performance_data = {
                    'timestamp': datetime.now().isoformat(),
                    'function_name': func.__name__,
                    'execution_time_ms': execution_time,
                    'memory_delta_mb': memory_delta,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'success': True,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                self.record_performance(performance_data)
                
                # Check for performance alerts
                self.check_performance_alerts(performance_data)
                
                # Log performance metrics
                self.logger.info(f"⚡ {func.__name__} - Time: {execution_time:.0f}ms, Memory: {memory_delta:+.1f}MB")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                # Record error performance data
                error_data = {
                    'timestamp': datetime.now().isoformat(),
                    'function_name': func.__name__,
                    'execution_time_ms': execution_time,
                    'success': False,
                    'error': str(e),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                self.record_performance(error_data)
                self.logger.error(f"❌ {func.__name__} failed after {execution_time:.0f}ms: {str(e)}")
                
                raise
        
        return wrapper
    
    def record_performance(self, performance_data: Dict[str, Any]):
        """Record performance data for analysis"""
        self.performance_history.append(performance_data)
        
        # Keep only last 1000 records to prevent memory bloat
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def check_performance_alerts(self, performance_data: Dict[str, Any]):
        """Check if performance data triggers any alerts"""
        alerts = []
        
        # Response time alert
        if performance_data['execution_time_ms'] > self.alert_thresholds['response_time_ms']:
            alerts.append({
                'type': 'HIGH_RESPONSE_TIME',
                'message': f"Function {performance_data['function_name']} took {performance_data['execution_time_ms']:.0f}ms",
                'severity': 'WARNING'
            })
        
        # Memory usage alert
        if performance_data.get('memory_delta_mb', 0) > self.alert_thresholds['memory_increase_mb']:
            alerts.append({
                'type': 'HIGH_MEMORY_USAGE',
                'message': f"Function {performance_data['function_name']} used {performance_data['memory_delta_mb']:.1f}MB memory",
                'severity': 'WARNING'
            })
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"🚨 ALERT [{alert['severity']}]: {alert['message']}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            data for data in self.performance_history
            if datetime.fromisoformat(data['timestamp']) > cutoff_time
        ]
        
        if not recent_data:
            return {'error': 'No performance data available'}
        
        # Calculate summary statistics
        successful_calls = [d for d in recent_data if d.get('success', True)]
        failed_calls = [d for d in recent_data if not d.get('success', True)]
        
        if successful_calls:
            avg_response_time = sum(d['execution_time_ms'] for d in successful_calls) / len(successful_calls)
            max_response_time = max(d['execution_time_ms'] for d in successful_calls)
            min_response_time = min(d['execution_time_ms'] for d in successful_calls)
            
            avg_memory_delta = sum(d.get('memory_delta_mb', 0) for d in successful_calls) / len(successful_calls)
            max_memory_delta = max(d.get('memory_delta_mb', 0) for d in successful_calls)
        else:
            avg_response_time = max_response_time = min_response_time = 0
            avg_memory_delta = max_memory_delta = 0
        
        error_rate = (len(failed_calls) / len(recent_data)) * 100 if recent_data else 0
        
        # Function performance breakdown
        function_stats = {}
        for data in successful_calls:
            func_name = data['function_name']
            if func_name not in function_stats:
                function_stats[func_name] = {
                    'call_count': 0,
                    'total_time_ms': 0,
                    'total_memory_mb': 0
                }
            
            function_stats[func_name]['call_count'] += 1
            function_stats[func_name]['total_time_ms'] += data['execution_time_ms']
            function_stats[func_name]['total_memory_mb'] += data.get('memory_delta_mb', 0)
        
        # Calculate averages for each function
        for func_name, stats in function_stats.items():
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['call_count']
            stats['avg_memory_mb'] = stats['total_memory_mb'] / stats['call_count']
        
        return {
            'period_hours': hours,
            'total_calls': len(recent_data),
            'successful_calls': len(successful_calls),
            'failed_calls': len(failed_calls),
            'error_rate_percent': error_rate,
            'avg_response_time_ms': avg_response_time,
            'min_response_time_ms': min_response_time,
            'max_response_time_ms': max_response_time,
            'avg_memory_delta_mb': avg_memory_delta,
            'max_memory_delta_mb': max_memory_delta,
            'function_breakdown': function_stats,
            'enterprise_health': self._assess_enterprise_health(avg_response_time, error_rate, max_memory_delta)
        }
    
    def _assess_enterprise_health(self, avg_response_time: float, error_rate: float, max_memory_delta: float) -> Dict[str, Any]:
        """Assess overall enterprise system health"""
        
        health_score = 100
        issues = []
        
        # Response time assessment
        if avg_response_time > self.alert_thresholds['response_time_ms']:
            health_score -= 30
            issues.append(f"High average response time: {avg_response_time:.0f}ms")
        
        # Error rate assessment
        if error_rate > self.alert_thresholds['error_rate_percent']:
            health_score -= 40
            issues.append(f"High error rate: {error_rate:.1f}%")
        
        # Memory usage assessment
        if max_memory_delta > self.alert_thresholds['memory_increase_mb']:
            health_score -= 20
            issues.append(f"High memory usage: {max_memory_delta:.1f}MB peak")
        
        health_status = 'EXCELLENT' if health_score >= 90 else \
                       'GOOD' if health_score >= 70 else \
                       'WARNING' if health_score >= 50 else 'CRITICAL'
        
        return {
            'health_score': max(0, health_score),
            'status': health_status,
            'issues': issues,
            'enterprise_ready': health_score >= 70
        }
    
    def export_performance_data(self, filepath: str):
        """Export performance data to file for analysis"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            self.logger.info(f"📊 Performance data exported to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to export performance data: {str(e)}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate optimization recommendations based on performance data"""
        
        if len(self.performance_history) < 10:
            return [{'recommendation': 'Insufficient data for optimization recommendations', 'priority': 'INFO'}]
        
        recommendations = []
        
        # Analyze recent performance
        recent_data = self.performance_history[-100:]  # Last 100 calls
        successful_calls = [d for d in recent_data if d.get('success', True)]
        
        if successful_calls:
            avg_response_time = sum(d['execution_time_ms'] for d in successful_calls) / len(successful_calls)
            avg_memory_delta = sum(d.get('memory_delta_mb', 0) for d in successful_calls) / len(successful_calls)
            
            # Response time recommendations
            if avg_response_time > 3000:
                recommendations.append({
                    'recommendation': 'Consider implementing response caching to reduce query times',
                    'priority': 'HIGH',
                    'metric': f'Average response time: {avg_response_time:.0f}ms'
                })
            
            if avg_response_time > 5000:
                recommendations.append({
                    'recommendation': 'Database query optimization needed - add indexes or optimize retrieval logic',
                    'priority': 'CRITICAL',
                    'metric': f'Average response time: {avg_response_time:.0f}ms'
                })
            
            # Memory recommendations
            if avg_memory_delta > 50:
                recommendations.append({
                    'recommendation': 'Implement memory management optimizations to reduce per-query memory usage',
                    'priority': 'MEDIUM',
                    'metric': f'Average memory increase: {avg_memory_delta:.1f}MB'
                })
            
            if avg_memory_delta > 100:
                recommendations.append({
                    'recommendation': 'Critical memory optimization needed - consider context limiting and garbage collection',
                    'priority': 'HIGH',
                    'metric': f'Average memory increase: {avg_memory_delta:.1f}MB'
                })
        
        # Error rate recommendations
        error_rate = len([d for d in recent_data if not d.get('success', True)]) / len(recent_data) * 100
        if error_rate > 2:
            recommendations.append({
                'recommendation': 'Implement better error handling and retry logic',
                'priority': 'HIGH',
                'metric': f'Error rate: {error_rate:.1f}%'
            })
        
        # Function-specific recommendations
        function_performance = {}
        for data in successful_calls:
            func_name = data['function_name']
            if func_name not in function_performance:
                function_performance[func_name] = []
            function_performance[func_name].append(data['execution_time_ms'])
        
        for func_name, times in function_performance.items():
            avg_time = sum(times) / len(times)
            if avg_time > 2000 and len(times) > 5:  # Function called multiple times with high response time
                recommendations.append({
                    'recommendation': f'Optimize {func_name} function - consistently high execution time',
                    'priority': 'MEDIUM',
                    'metric': f'{func_name} average: {avg_time:.0f}ms'
                })
        
        if not recommendations:
            recommendations.append({
                'recommendation': 'System performance is within acceptable parameters',
                'priority': 'INFO',
                'metric': 'All metrics healthy'
            })
        
        return recommendations

# Global monitor instance
enterprise_monitor = EnterpriseMonitor()

# Decorator for easy use
def monitor_performance(func):
    """Convenience decorator for monitoring function performance"""
    return enterprise_monitor.monitor_performance(func)

# System health check function
def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    
    # Current system metrics
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    # Disk usage
    disk_usage = psutil.disk_usage('/')
    
    system_health = {
        'timestamp': datetime.now().isoformat(),
        'memory_usage_mb': memory_info.rss / 1024 / 1024,
        'memory_percent': psutil.virtual_memory().percent,
        'cpu_percent': cpu_percent,
        'disk_usage_percent': (disk_usage.used / disk_usage.total) * 100,
        'disk_free_gb': disk_usage.free / 1024 / 1024 / 1024,
        'system_healthy': True
    }
    
    # Health assessment
    health_issues = []
    if system_health['memory_percent'] > 85:
        health_issues.append("High memory usage")
        system_health['system_healthy'] = False
    
    if system_health['cpu_percent'] > 80:
        health_issues.append("High CPU usage")
        system_health['system_healthy'] = False
    
    if system_health['disk_usage_percent'] > 90:
        health_issues.append("Low disk space")
        system_health['system_healthy'] = False
    
    system_health['health_issues'] = health_issues
    system_health['health_status'] = 'HEALTHY' if system_health['system_healthy'] else 'WARNING'
    
    return system_health