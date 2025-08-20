import time
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

class AgentPerformanceMonitor:
    def __init__(self):
        self.performance_data = defaultdict(list)
        self.guardrail_stats = defaultdict(int)
        self.intervention_stats = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def log_interaction(self, interaction_data: Dict[str, Any]):
        """Log agent interaction for performance analysis"""
        
        timestamp = datetime.now()
        
        # Store performance metrics
        self.performance_data['response_times'].append({
            'timestamp': timestamp,
            'response_time_ms': interaction_data.get('response_time_ms', 0),
            'query_type': interaction_data.get('agent_type', 'unknown'),
            'confidence': interaction_data.get('confidence', 0)
        })
        
        # Track guardrail performance
        guardrails_passed = interaction_data.get('guardrails_passed', True)
        self.guardrail_stats['total'] += 1
        if guardrails_passed:
            self.guardrail_stats['passed'] += 1
        else:
            self.guardrail_stats['failed'] += 1
            
        # Track interventions
        if interaction_data.get('intervention_triggered', False):
            trigger_type = interaction_data.get('trigger_type', 'unknown')
            self.intervention_stats[trigger_type] += 1
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent data
        recent_responses = [
            r for r in self.performance_data['response_times']
            if r['timestamp'] > cutoff_time
        ]
        
        if not recent_responses:
            return {'status': 'no_recent_data'}
        
        # Calculate metrics
        response_times = [r['response_time_ms'] for r in recent_responses]
        confidences = [r['confidence'] for r in recent_responses]
        
        summary = {
            'time_period_hours': hours,
            'total_interactions': len(recent_responses),
            'performance_metrics': {
                'avg_response_time_ms': sum(response_times) / len(response_times),
                'max_response_time_ms': max(response_times),
                'min_response_time_ms': min(response_times),
                'avg_confidence': sum(confidences) / len(confidences),
                'responses_under_3s': sum(1 for t in response_times if t < 3000),
                'responses_under_5s': sum(1 for t in response_times if t < 5000)
            },
            'guardrail_metrics': {
                'total_checks': self.guardrail_stats['total'],
                'pass_rate': self.guardrail_stats['passed'] / max(self.guardrail_stats['total'], 1),
                'failure_count': self.guardrail_stats['failed']
            },
            'intervention_metrics': dict(self.intervention_stats),
            'agent_types': self._analyze_agent_types(recent_responses),
            'enterprise_readiness': self._assess_enterprise_readiness(recent_responses)
        }
        
        return summary
    
    def _analyze_agent_types(self, responses: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of agent types used"""
        
        type_counts = defaultdict(int)
        for response in responses:
            agent_type = response.get('query_type', 'unknown')
            type_counts[agent_type] += 1
        
        total = len(responses)
        return {
            'single_agent': type_counts.get('single_agent', 0),
            'committee_consultation': type_counts.get('committee_consultation', 0),
            'percentages': {
                'single_agent': type_counts.get('single_agent', 0) / total * 100,
                'committee_consultation': type_counts.get('committee_consultation', 0) / total * 100
            }
        }
    
    def _assess_enterprise_readiness(self, responses: List[Dict]) -> Dict[str, Any]:
        """Assess if system meets enterprise standards"""
        
        response_times = [r['response_time_ms'] for r in responses]
        confidences = [r['confidence'] for r in responses]
        
        avg_response_time = sum(response_times) / len(response_times)
        avg_confidence = sum(confidences) / len(confidences)
        fast_response_rate = sum(1 for t in response_times if t < 3000) / len(response_times)
        guardrail_pass_rate = self.guardrail_stats['passed'] / max(self.guardrail_stats['total'], 1)
        
        criteria = {
            'avg_response_under_3s': avg_response_time < 3000,
            'fast_response_rate_80pct': fast_response_rate >= 0.8,
            'avg_confidence_70pct': avg_confidence >= 0.7,
            'guardrail_pass_rate_95pct': guardrail_pass_rate >= 0.95
        }
        
        enterprise_ready = all(criteria.values())
        
        return {
            'overall_ready': enterprise_ready,
            'criteria_met': criteria,
            'scores': {
                'avg_response_time_ms': avg_response_time,
                'fast_response_rate': fast_response_rate,
                'avg_confidence': avg_confidence,
                'guardrail_pass_rate': guardrail_pass_rate
            },
            'recommendations': self._generate_recommendations(criteria)
        }
    
    def _generate_recommendations(self, criteria: Dict[str, bool]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        if not criteria['avg_response_under_3s']:
            recommendations.append("Optimize response time - consider model size reduction or caching")
        
        if not criteria['fast_response_rate_80pct']:
            recommendations.append("Improve consistency - 80% of responses should be under 3 seconds")
        
        if not criteria['avg_confidence_70pct']:
            recommendations.append("Enhance context retrieval to improve confidence scores")
        
        if not criteria['guardrail_pass_rate_95pct']:
            recommendations.append("Review and tune guardrails for better reliability")
        
        if not recommendations:
            recommendations.append("System meets enterprise standards - ready for production")
        
        return recommendations

def create_agent_performance_monitor():
    """Factory function to create agent performance monitor"""
    return AgentPerformanceMonitor()