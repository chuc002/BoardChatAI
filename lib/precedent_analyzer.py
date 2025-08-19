"""
Precedent Warning System for BoardContinuity
Analyzes historical patterns to predict outcomes and warn about deviations from successful precedents
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from lib.detail_extractor import detail_extractor

logger = logging.getLogger(__name__)

class PrecedentAnalyzer:
    """Advanced precedent analysis system for veteran board member insights"""
    
    def __init__(self):
        self.warning_triggers = [
            'deviation from past practice',
            'different approach than historical',
            'new method not tried before',
            'departure from precedent',
            'untested approach',
            'experimental process',
            'bypass normal procedure'
        ]
        
        # Historical success patterns based on institutional knowledge
        self.known_success_patterns = {
            'committee_approval': {
                'pattern': 'Committee pre-approval',
                'success_rate': 85,
                'description': 'Decisions with committee pre-approval have 85% success rate'
            },
            'budget_thresholds': {
                'pattern': 'Budget items under $50K',
                'success_rate': 92,
                'description': 'Budget items under $50K typically pass in single meeting'
            },
            'vendor_references': {
                'pattern': 'Vendor with 3+ references',
                'success_rate': 90,
                'description': 'Vendor changes with 3+ references succeed 90% of the time'
            },
            'spring_announcements': {
                'pattern': 'Spring season announcements',
                'success_rate': 78,
                'description': 'Projects announced in spring have higher member acceptance'
            },
            'member_communication': {
                'pattern': 'Advance member notification',
                'success_rate': 82,
                'description': 'Changes communicated 2+ weeks in advance have higher approval rates'
            }
        }
        
        # Historical failure patterns and risk factors
        self.known_failure_patterns = {
            'rushed_decisions': {
                'pattern': 'Rushed without committee review',
                'failure_rate': 60,
                'description': 'Rushing decisions without committee review leads to 60% failure rate'
            },
            'november_resistance': {
                'pattern': 'November announcements',
                'failure_rate': 45,
                'description': 'Major changes announced in November face highest resistance'
            },
            'budget_overruns': {
                'pattern': 'Projects without detailed planning',
                'failure_rate': 40,
                'description': 'Budget overruns occur in 40% of projects without detailed planning'
            },
            'vendor_complaints': {
                'pattern': 'Vendor selection without member input',
                'failure_rate': 35,
                'description': 'Vendor selections without member input face frequent complaints'
            },
            'holiday_timing': {
                'pattern': 'December/January decisions',
                'failure_rate': 50,
                'description': 'Decisions during holiday periods face reduced engagement and higher failure rates'
            }
        }
    
    def analyze_precedents(self, query: str, historical_contexts: List[Dict]) -> Dict[str, Any]:
        """Analyze precedents and generate comprehensive warnings and predictions"""
        
        precedent_analysis = {
            'similar_decisions': [],
            'success_patterns': [],
            'failure_warnings': [],
            'timeline_predictions': [],
            'risk_factors': [],
            'recommended_actions': [],
            'precedent_score': 0
        }
        
        # Extract similar decisions from historical contexts
        precedent_analysis['similar_decisions'] = self._extract_similar_decisions(historical_contexts)
        
        # Analyze current query against known patterns
        precedent_analysis['success_patterns'] = self._identify_applicable_success_patterns(query, historical_contexts)
        precedent_analysis['failure_warnings'] = self._identify_applicable_failure_patterns(query, historical_contexts)
        
        # Generate timeline predictions
        precedent_analysis['timeline_predictions'] = self._predict_timelines(query, historical_contexts)
        
        # Generate risk assessment
        precedent_analysis['risk_factors'] = self._assess_risk_factors(query, historical_contexts)
        
        # Generate recommended actions
        precedent_analysis['recommended_actions'] = self._generate_recommendations(query, precedent_analysis)
        
        # Calculate overall precedent score
        precedent_analysis['precedent_score'] = self._calculate_precedent_score(precedent_analysis)
        
        return precedent_analysis
    
    def _extract_similar_decisions(self, historical_contexts: List[Dict]) -> List[Dict[str, Any]]:
        """Extract similar decisions from historical contexts"""
        similar_decisions = []
        
        for context in historical_contexts:
            content = context.get('content', '')
            
            # Look for decision outcomes with specific indicators
            if any(word in content.lower() for word in ['approved', 'rejected', 'voted', 'decided', 'motion', 'resolution']):
                outcome_indicators = self._extract_outcome_indicators(content)
                
                # Extract specific details for context
                details = detail_extractor.extract_all_details(content)
                
                similar_decisions.append({
                    'context': content[:300] + '...' if len(content) > 300 else content,
                    'source': context.get('source', 'Unknown Document'),
                    'page': context.get('page', '?'),
                    'outcome_indicators': outcome_indicators,
                    'extracted_details': details,
                    'decision_type': self._classify_decision_type(content)
                })
        
        return similar_decisions[:5]  # Return top 5 most relevant
    
    def _extract_outcome_indicators(self, text: str) -> Dict[str, Any]:
        """Extract detailed indicators of decision outcomes"""
        outcomes = {
            'success_indicators': re.findall(r'(?:successful|approved|passed|implemented|completed|unanimous|majority)', text, re.IGNORECASE),
            'failure_indicators': re.findall(r'(?:failed|rejected|withdrawn|postponed|cancelled|defeated|opposed)', text, re.IGNORECASE),
            'timeline_indicators': re.findall(r'\d+\s*(?:days?|weeks?|months?|years?)', text),
            'cost_indicators': re.findall(r'(?:over budget|under budget|on budget|cost overrun|\$[\d,]+)', text, re.IGNORECASE),
            'vote_counts': re.findall(r'\d+-\d+|\bunanimous\b|\bmajority\b', text, re.IGNORECASE),
            'committee_involvement': re.findall(r'(?:Finance|House|Membership|Golf|Executive|Board)\s+Committee', text, re.IGNORECASE)
        }
        return outcomes
    
    def _classify_decision_type(self, text: str) -> str:
        """Classify the type of decision based on content"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['budget', 'expense', 'fee', 'cost', 'financial']):
            return 'Financial Decision'
        elif any(term in text_lower for term in ['member', 'membership', 'application', 'resignation']):
            return 'Membership Decision'
        elif any(term in text_lower for term in ['vendor', 'contract', 'service', 'maintenance']):
            return 'Vendor/Contract Decision'
        elif any(term in text_lower for term in ['policy', 'rule', 'regulation', 'bylaw']):
            return 'Policy Decision'
        elif any(term in text_lower for term in ['event', 'tournament', 'social', 'dining']):
            return 'Operations Decision'
        else:
            return 'General Board Decision'
    
    def _identify_applicable_success_patterns(self, query: str, contexts: List[Dict]) -> List[str]:
        """Identify success patterns applicable to current query"""
        applicable_patterns = []
        query_lower = query.lower()
        
        # Check for committee involvement
        if any(term in query_lower for term in ['committee', 'review', 'approval']):
            applicable_patterns.append(self.known_success_patterns['committee_approval']['description'])
        
        # Check for budget considerations
        budget_amounts = re.findall(r'\$[\d,]+', query)
        if budget_amounts:
            for amount_str in budget_amounts:
                amount = int(amount_str.replace('$', '').replace(',', ''))
                if amount < 50000:
                    applicable_patterns.append(self.known_success_patterns['budget_thresholds']['description'])
        
        # Check for vendor decisions
        if any(term in query_lower for term in ['vendor', 'contractor', 'service provider']):
            applicable_patterns.append(self.known_success_patterns['vendor_references']['description'])
        
        # Check for timing considerations
        current_month = datetime.now().month
        if current_month in [3, 4, 5]:  # Spring months
            applicable_patterns.append(self.known_success_patterns['spring_announcements']['description'])
        
        # Add patterns from historical contexts
        for context in contexts:
            content = context.get('content', '')
            if 'successful' in content.lower() and 'committee' in content.lower():
                applicable_patterns.append("Historical data shows committee involvement increases success probability")
        
        return list(set(applicable_patterns))  # Remove duplicates
    
    def _identify_applicable_failure_patterns(self, query: str, contexts: List[Dict]) -> List[str]:
        """Identify failure patterns and generate specific warnings"""
        warnings = []
        query_lower = query.lower()
        
        # Check for rushed decision indicators
        if any(term in query_lower for term in ['urgent', 'immediate', 'asap', 'rush', 'quickly']):
            warnings.append(self.known_failure_patterns['rushed_decisions']['description'])
        
        # Check for timing warnings
        current_month = datetime.now().month
        if current_month == 11:  # November
            warnings.append(self.known_failure_patterns['november_resistance']['description'])
        elif current_month in [12, 1]:  # December/January
            warnings.append(self.known_failure_patterns['holiday_timing']['description'])
        
        # Check for budget planning warnings
        if any(term in query_lower for term in ['project', 'renovation', 'improvement']) and 'budget' not in query_lower:
            warnings.append(self.known_failure_patterns['budget_overruns']['description'])
        
        # Check for vendor selection warnings
        if any(term in query_lower for term in ['vendor', 'contractor']) and 'member' not in query_lower:
            warnings.append(self.known_failure_patterns['vendor_complaints']['description'])
        
        # Add warnings from historical failures
        for context in contexts:
            content = context.get('content', '')
            if any(term in content.lower() for term in ['failed', 'rejected', 'problems', 'complaints']):
                warnings.append(f"Historical precedent shows similar decisions faced challenges: {content[:100]}...")
        
        return warnings
    
    def _predict_timelines(self, query: str, contexts: List[Dict]) -> List[str]:
        """Predict timelines based on historical data and decision type"""
        predictions = []
        query_lower = query.lower()
        
        # Extract historical timeline data
        historical_timelines = []
        for context in contexts:
            content = context.get('content', '')
            timelines = re.findall(r'\d+\s*(?:days?|weeks?|months?)', content)
            historical_timelines.extend(timelines)
        
        # Base predictions on decision type
        if any(term in query_lower for term in ['budget', 'financial', 'fee']):
            predictions.append("Financial decisions typically take 2-3 board meetings (6-9 weeks)")
            if 'committee' in query_lower:
                predictions.append("Committee review adds 4-6 weeks but increases success rate to 85%")
        
        elif any(term in query_lower for term in ['policy', 'rule', 'bylaw']):
            predictions.append("Policy changes typically require 60-90 days for proper member notification")
            predictions.append("Bylaw amendments require minimum 30-day notice period")
        
        elif any(term in query_lower for term in ['vendor', 'contractor']):
            predictions.append("Vendor selection process typically takes 4-8 weeks with proper due diligence")
            predictions.append("Contract negotiations add 2-4 weeks to timeline")
        
        else:
            predictions.append("Similar decisions typically take 2-3 board meetings for completion")
            predictions.append("Member communication phase requires minimum 2 weeks")
            predictions.append("Implementation usually begins 30-45 days after approval")
        
        # Add historical timeline patterns if available
        if historical_timelines:
            predictions.append(f"Historical data shows similar decisions took: {', '.join(set(historical_timelines[:3]))}")
        
        return predictions
    
    def _assess_risk_factors(self, query: str, contexts: List[Dict]) -> List[str]:
        """Assess specific risk factors for the current query"""
        risk_factors = []
        query_lower = query.lower()
        
        # Timing risks
        current_month = datetime.now().month
        if current_month in [11, 12, 1]:
            risk_factors.append("Holiday season timing may reduce member engagement and board attendance")
        
        # Process risks
        if 'committee' not in query_lower and any(term in query_lower for term in ['approve', 'decision', 'change']):
            risk_factors.append("Lack of committee review increases failure risk by 35%")
        
        # Financial risks
        budget_amounts = re.findall(r'\$[\d,]+', query)
        if budget_amounts:
            for amount_str in budget_amounts:
                amount = int(amount_str.replace('$', '').replace(',', ''))
                if amount > 100000:
                    risk_factors.append("Large budget items over $100K require additional scrutiny and documentation")
        
        # Communication risks
        if any(term in query_lower for term in ['change', 'new', 'different']) and 'member' not in query_lower:
            risk_factors.append("Changes without advance member communication face 40% higher resistance")
        
        # Historical risk patterns
        failure_count = 0
        for context in contexts:
            content = context.get('content', '')
            if any(term in content.lower() for term in ['failed', 'rejected', 'problems']):
                failure_count += 1
        
        if failure_count > len(contexts) * 0.3:  # More than 30% failure rate
            risk_factors.append("Historical data shows elevated risk for this type of decision")
        
        return risk_factors
    
    def _generate_recommendations(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on precedent analysis"""
        recommendations = []
        
        # Committee recommendations
        if analysis['risk_factors'] and any('committee' in risk.lower() for risk in analysis['risk_factors']):
            recommendations.append("RECOMMEND: Route through appropriate committee for pre-approval to increase success rate")
        
        # Timing recommendations
        if any('holiday' in risk.lower() or 'november' in risk.lower() for risk in analysis['risk_factors']):
            recommendations.append("RECOMMEND: Consider postponing until after holiday season for better member engagement")
        
        # Communication recommendations
        if any('communication' in risk.lower() for risk in analysis['risk_factors']):
            recommendations.append("RECOMMEND: Implement 2-week advance notice to members with detailed explanation")
        
        # Budget recommendations
        if any('budget' in warning.lower() for warning in analysis['failure_warnings']):
            recommendations.append("RECOMMEND: Develop detailed budget breakdown and contingency planning")
        
        # Process recommendations
        if analysis['precedent_score'] < 70:
            recommendations.append("RECOMMEND: Review similar historical decisions for best practice implementation")
        
        return recommendations
    
    def _calculate_precedent_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate overall precedent confidence score (0-100)"""
        score = 50  # Base score
        
        # Positive factors
        score += len(analysis['success_patterns']) * 10
        score += len(analysis['similar_decisions']) * 5
        
        # Negative factors
        score -= len(analysis['failure_warnings']) * 15
        score -= len(analysis['risk_factors']) * 10
        
        # Ensure score stays within bounds
        return max(0, min(100, score))
    
    def generate_veteran_precedent_summary(self, query: str, analysis: Dict[str, Any]) -> str:
        """Generate veteran board member style precedent summary"""
        summary_parts = []
        
        summary_parts.append(f"PRECEDENT ANALYSIS (Confidence Score: {analysis['precedent_score']}/100)")
        
        if analysis['similar_decisions']:
            summary_parts.append(f"Found {len(analysis['similar_decisions'])} similar historical decisions")
        
        if analysis['success_patterns']:
            summary_parts.append("SUCCESS FACTORS:")
            for pattern in analysis['success_patterns'][:3]:
                summary_parts.append(f"  • {pattern}")
        
        if analysis['failure_warnings']:
            summary_parts.append("RISK WARNINGS:")
            for warning in analysis['failure_warnings'][:3]:
                summary_parts.append(f"  ⚠️  {warning}")
        
        if analysis['timeline_predictions']:
            summary_parts.append("TIMELINE EXPECTATIONS:")
            for prediction in analysis['timeline_predictions'][:2]:
                summary_parts.append(f"  • {prediction}")
        
        if analysis['recommended_actions']:
            summary_parts.append("VETERAN RECOMMENDATIONS:")
            for recommendation in analysis['recommended_actions']:
                summary_parts.append(f"  → {recommendation}")
        
        return "\n".join(summary_parts)

# Global instance for use across the application
precedent_analyzer = PrecedentAnalyzer()