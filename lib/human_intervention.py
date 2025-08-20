"""
Human Intervention System for BoardContinuity AI
Intelligent detection and escalation for high-risk governance scenarios requiring human oversight
"""

from typing import Dict, Any, Optional, List
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class InterventionTrigger(Enum):
    HIGH_RISK_ACTION = "high_risk_action"
    LOW_CONFIDENCE = "low_confidence"
    GUARDRAIL_FAILURE = "guardrail_failure"
    COMPLEX_EDGE_CASE = "complex_edge_case"
    USER_REQUEST = "user_request"
    FINANCIAL_THRESHOLD = "financial_threshold"
    LEGAL_CONCERN = "legal_concern"

class HumanInterventionManager:
    def __init__(self):
        # High-risk keywords that trigger immediate escalation
        self.high_risk_keywords = [
            'terminate', 'fire', 'dismiss', 'legal action', 'lawsuit', 'litigation',
            'emergency fund', 'large expenditure', 'major contract', 'acquisition',
            'membership termination', 'bylaw change', 'constitutional amendment',
            'discrimination', 'harassment', 'safety violation', 'emergency response',
            'member expulsion', 'board removal', 'ethics violation', 'audit finding'
        ]
        
        # Legal concern keywords
        self.legal_keywords = [
            'liability', 'insurance claim', 'regulatory compliance', 'tax issue',
            'employment law', 'ada compliance', 'discrimination', 'harassment',
            'intellectual property', 'contract dispute', 'zoning violation'
        ]
        
        # Financial threshold keywords
        self.financial_keywords = [
            'major renovation', 'capital project', 'large contract', 'bond issue',
            'significant expenditure', 'major investment', 'budget deficit',
            'emergency funding', 'loan approval', 'debt restructuring'
        ]
        
        # User request patterns for human assistance
        self.human_request_patterns = [
            'speak to human', 'human help', 'escalate', 'transfer to person',
            'real person', 'not ai', 'human expert', 'board member',
            'specialist', 'professional advice', 'legal counsel'
        ]
        
        # Intervention thresholds
        self.intervention_thresholds = {
            'confidence_threshold': 0.65,
            'financial_threshold': 50000,  # $50K+ decisions
            'risk_score_threshold': 0.8,
            'guardrail_failure_threshold': 2  # Multiple guardrail failures
        }
        
        logger.info("Human intervention manager initialized with enterprise thresholds")
    
    def should_intervene(self, query: str, response_data: Dict[str, Any]) -> Optional[InterventionTrigger]:
        """Determine if human intervention is needed based on comprehensive risk assessment"""
        
        # Check for explicit user requests first
        if self._user_requests_human(query):
            logger.info("Human intervention triggered: User explicit request")
            return InterventionTrigger.USER_REQUEST
        
        # Check for high-risk actions
        if self._is_high_risk_action(query):
            logger.warning("Human intervention triggered: High-risk action detected")
            return InterventionTrigger.HIGH_RISK_ACTION
        
        # Check for legal concerns
        if self._involves_legal_concern(query):
            logger.warning("Human intervention triggered: Legal concern detected")
            return InterventionTrigger.LEGAL_CONCERN
        
        # Check confidence levels
        confidence = response_data.get('confidence', 1.0)
        if confidence < self.intervention_thresholds['confidence_threshold']:
            logger.info(f"Human intervention triggered: Low confidence ({confidence:.2f})")
            return InterventionTrigger.LOW_CONFIDENCE
        
        # Check for guardrail failures
        guardrail_flags = response_data.get('guardrail_flags', {})
        failed_checks = response_data.get('failed_checks', [])
        if (not response_data.get('guardrails_passed', True) or 
            len(failed_checks) >= self.intervention_thresholds['guardrail_failure_threshold']):
            logger.warning("Human intervention triggered: Guardrail failures")
            return InterventionTrigger.GUARDRAIL_FAILURE
        
        # Check for financial thresholds
        if self._involves_large_financial_decision(query):
            logger.info("Human intervention triggered: Large financial decision")
            return InterventionTrigger.FINANCIAL_THRESHOLD
        
        # Check for complex edge cases (multiple committees, synthesis failures)
        if self._is_complex_edge_case(response_data):
            logger.info("Human intervention triggered: Complex edge case")
            return InterventionTrigger.COMPLEX_EDGE_CASE
        
        return None
    
    def _is_high_risk_action(self, query: str) -> bool:
        """Check if query involves high-risk governance actions"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.high_risk_keywords)
    
    def _involves_legal_concern(self, query: str) -> bool:
        """Check if query involves legal concerns requiring counsel"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.legal_keywords)
    
    def _involves_large_financial_decision(self, query: str) -> bool:
        """Check if query involves large financial decisions above threshold"""
        
        # Look for explicit dollar amounts
        amounts = re.findall(r'\$([0-9,]+)', query)
        for amount in amounts:
            try:
                value = int(amount.replace(',', ''))
                if value >= self.intervention_thresholds['financial_threshold']:
                    return True
            except ValueError:
                continue
        
        # Look for keywords indicating large financial decisions
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.financial_keywords):
            return True
        
        # Look for percentage-based major decisions
        percentages = re.findall(r'(\d+)%', query)
        for percentage in percentages:
            if int(percentage) >= 20:  # 20%+ budget changes
                return True
        
        return False
    
    def _user_requests_human(self, query: str) -> bool:
        """Check if user explicitly requests human assistance"""
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.human_request_patterns)
    
    def _is_complex_edge_case(self, response_data: Dict[str, Any]) -> bool:
        """Check if response indicates complex edge case requiring human oversight"""
        
        # Multiple committee consultation with low synthesis confidence
        committees_consulted = response_data.get('committees_consulted', [])
        if len(committees_consulted) >= 3:
            confidence = response_data.get('confidence', 1.0)
            if confidence < 0.8:
                return True
        
        # Error in committee synthesis or agent processing
        if response_data.get('error') and 'synthesis' in str(response_data.get('error')):
            return True
        
        # Strategy indicates fallback or error handling
        strategy = response_data.get('strategy', '')
        if strategy in ['fallback_single_agent', 'error_fallback']:
            return True
        
        return False
    
    def create_intervention_response(self, trigger: InterventionTrigger, query: str, response_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create appropriate intervention response with detailed context"""
        
        intervention_messages = {
            InterventionTrigger.HIGH_RISK_ACTION: {
                'message': "This request involves a high-risk governance decision that requires human oversight. I'm connecting you with a board governance specialist who can provide expert guidance while ensuring all proper procedures are followed.",
                'next_steps': "A qualified governance professional will review your situation and provide comprehensive guidance.",
                'urgency': 'high'
            },
            InterventionTrigger.LEGAL_CONCERN: {
                'message': "This matter involves legal considerations that require professional legal counsel. I'm connecting you with a governance specialist who can coordinate with appropriate legal resources.",
                'next_steps': "A specialist will review the legal implications and ensure you receive proper counsel.",
                'urgency': 'high'
            },
            InterventionTrigger.FINANCIAL_THRESHOLD: {
                'message': "This involves a significant financial decision that warrants human review. I'm connecting you with a governance specialist who can ensure all fiduciary responsibilities are properly addressed.",
                'next_steps': "A financial governance expert will review the proposed decision and provide detailed guidance.",
                'urgency': 'medium'
            },
            InterventionTrigger.LOW_CONFIDENCE: {
                'message': "I want to ensure you receive the most accurate guidance for this complex situation. Let me connect you with a governance expert who can provide detailed, personalized assistance.",
                'next_steps': "A specialist will analyze your question thoroughly and provide comprehensive guidance.",
                'urgency': 'medium'
            },
            InterventionTrigger.GUARDRAIL_FAILURE: {
                'message': "I need to ensure this request receives appropriate review to provide you with proper guidance. I'm connecting you with a qualified governance specialist.",
                'next_steps': "Your request will be reviewed by a professional who can address any sensitive aspects appropriately.",
                'urgency': 'medium'
            },
            InterventionTrigger.COMPLEX_EDGE_CASE: {
                'message': "This situation involves unique circumstances that would benefit from human expertise and institutional knowledge. I'm connecting you with a governance specialist.",
                'next_steps': "A specialist will analyze the specific circumstances and provide tailored guidance based on institutional experience.",
                'urgency': 'low'
            },
            InterventionTrigger.USER_REQUEST: {
                'message': "I understand you'd prefer to speak with a human governance specialist. I'm connecting you with a qualified professional who can provide personalized assistance.",
                'next_steps': "Please hold while we connect you with a board governance expert.",
                'urgency': 'low'
            }
        }
        
        intervention_info = intervention_messages[trigger]
        
        # Build context summary for human specialist
        context_summary = self._build_context_summary(query, response_data, trigger)
        
        return {
            'response': intervention_info['message'],
            'intervention_triggered': True,
            'trigger_type': trigger.value,
            'urgency_level': intervention_info['urgency'],
            'next_steps': intervention_info['next_steps'],
            'escalation_reason': trigger.value,
            'original_query': query,
            'context_summary': context_summary,
            'human_assistance_required': True,
            'estimated_response_time': self._estimate_response_time(intervention_info['urgency']),
            'specialist_type': self._determine_specialist_type(trigger, query)
        }
    
    def _build_context_summary(self, query: str, response_data: Dict[str, Any], trigger: InterventionTrigger) -> Dict[str, Any]:
        """Build comprehensive context summary for human specialist"""
        
        context = {
            'query_analysis': {
                'high_risk_detected': self._is_high_risk_action(query),
                'legal_concerns': self._involves_legal_concern(query),
                'financial_threshold': self._involves_large_financial_decision(query),
                'complexity_indicators': []
            },
            'ai_processing_results': {},
            'recommended_specialist_focus': []
        }
        
        if response_data:
            context['ai_processing_results'] = {
                'confidence_level': response_data.get('confidence', 0),
                'strategy_used': response_data.get('strategy', 'unknown'),
                'committees_consulted': response_data.get('committees_consulted', []),
                'guardrails_status': response_data.get('guardrails_passed', True),
                'failed_checks': response_data.get('failed_checks', [])
            }
        
        # Add specialist focus recommendations
        if trigger == InterventionTrigger.LEGAL_CONCERN:
            context['recommended_specialist_focus'].append('legal_compliance')
        if trigger == InterventionTrigger.FINANCIAL_THRESHOLD:
            context['recommended_specialist_focus'].append('financial_governance')
        if trigger == InterventionTrigger.HIGH_RISK_ACTION:
            context['recommended_specialist_focus'].append('risk_management')
        
        return context
    
    def _estimate_response_time(self, urgency: str) -> str:
        """Estimate response time based on urgency level"""
        time_estimates = {
            'high': '15-30 minutes',
            'medium': '1-2 hours',
            'low': '4-6 hours'
        }
        return time_estimates.get(urgency, '2-4 hours')
    
    def _determine_specialist_type(self, trigger: InterventionTrigger, query: str) -> str:
        """Determine the most appropriate specialist type"""
        
        if trigger == InterventionTrigger.LEGAL_CONCERN:
            return 'Legal Governance Specialist'
        elif trigger == InterventionTrigger.FINANCIAL_THRESHOLD:
            return 'Financial Governance Specialist'
        elif trigger == InterventionTrigger.HIGH_RISK_ACTION:
            return 'Senior Governance Specialist'
        elif 'membership' in query.lower():
            return 'Membership Governance Specialist'
        elif any(word in query.lower() for word in ['golf', 'course', 'grounds']):
            return 'Operations Governance Specialist'
        else:
            return 'Board Governance Specialist'
    
    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get statistics about intervention patterns (for monitoring/analytics)"""
        return {
            'thresholds': self.intervention_thresholds,
            'high_risk_keywords_count': len(self.high_risk_keywords),
            'legal_keywords_count': len(self.legal_keywords),
            'financial_keywords_count': len(self.financial_keywords),
            'intervention_triggers': [trigger.value for trigger in InterventionTrigger]
        }

# Factory function for easy integration
def create_human_intervention_manager() -> HumanInterventionManager:
    """Factory function to create human intervention manager"""
    return HumanInterventionManager()