"""
Governance Intelligence System - Complete historical context and predictive analysis.

This module provides comprehensive governance intelligence by analyzing historical patterns,
predicting outcomes, and providing detailed context for board decisions.
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

from lib.supa import supa
from lib.pattern_recognition import GovernancePatternEngine
from lib.knowledge_graph import InstitutionalKnowledgeGraph

logger = logging.getLogger(__name__)

@dataclass
class DecisionAnalysis:
    """Complete analysis of a decision with historical context."""
    precedents: List[Dict[str, Any]]
    success_probability: float
    typical_timeline: Dict[str, Any]
    common_amendments: List[str]
    risk_factors: List[str]
    member_sentiment_prediction: Dict[str, Any]
    financial_impact_forecast: Dict[str, Any]
    patterns: Dict[str, Any]
    confidence_level: str
    recommendations: List[str]

@dataclass
class OutcomePrediction:
    """Prediction for a specific proposal."""
    likely_outcome: str
    confidence: float
    key_factors: List[str]
    recommended_modifications: List[str]
    optimal_timing: Dict[str, Any]
    success_indicators: List[str]
    failure_indicators: List[str]
    member_support_forecast: Dict[str, float]

class GovernanceIntelligence:
    """
    Main governance intelligence system providing complete historical context
    and predictive analysis for board decisions.
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.pattern_engine = GovernancePatternEngine(org_id)
        self.knowledge_graph = InstitutionalKnowledgeGraph(org_id)
        
        # Cache for frequently accessed data
        self.decision_patterns = {}
        self.member_patterns = {}
        self.financial_patterns = {}
        self.last_cache_update = None
        
        # Initialize intelligence caches
        self._initialize_intelligence_cache()
    
    def analyze_decision(self, new_decision: Dict[str, Any]) -> DecisionAnalysis:
        """Provide complete historical context for any decision."""
        logger.info(f"Analyzing decision: {new_decision.get('title', 'Untitled')}")
        
        # Find similar decisions with comprehensive matching
        similar_decisions = self._find_similar_decisions(new_decision)
        
        # Build comprehensive analysis
        analysis = DecisionAnalysis(
            precedents=self._analyze_precedents(similar_decisions),
            success_probability=self._calculate_success_probability(similar_decisions),
            typical_timeline=self._analyze_timeline_patterns(similar_decisions),
            common_amendments=self._identify_common_amendments(similar_decisions),
            risk_factors=self._identify_risk_factors(new_decision, similar_decisions),
            member_sentiment_prediction=self._predict_member_sentiment(new_decision),
            financial_impact_forecast=self._forecast_financial_impact(new_decision, similar_decisions),
            patterns=self._identify_comprehensive_patterns(similar_decisions),
            confidence_level=self._assess_analysis_confidence(similar_decisions),
            recommendations=self._generate_strategic_recommendations(new_decision, similar_decisions)
        )
        
        return analysis
    
    def predict_outcome(self, proposal: Dict[str, Any]) -> OutcomePrediction:
        """Predict outcome with comprehensive reasoning."""
        logger.info(f"Predicting outcome for: {proposal.get('title', 'Untitled')}")
        
        # Extract decision features
        features = self._extract_comprehensive_features(proposal)
        
        # Get historical outcomes for similar features
        historical_outcomes = self._get_historical_outcomes(features)
        
        # Build comprehensive prediction
        prediction = OutcomePrediction(
            likely_outcome=self._calculate_likely_outcome(historical_outcomes),
            confidence=self._calculate_prediction_confidence(historical_outcomes),
            key_factors=self._identify_key_success_factors(historical_outcomes),
            recommended_modifications=self._suggest_strategic_improvements(proposal, historical_outcomes),
            optimal_timing=self._calculate_optimal_timing(proposal, historical_outcomes),
            success_indicators=self._identify_success_indicators(historical_outcomes),
            failure_indicators=self._identify_failure_indicators(historical_outcomes),
            member_support_forecast=self._forecast_member_support(proposal)
        )
        
        return prediction
    
    def get_complete_context(self, decision_id: str) -> Dict[str, Any]:
        """Get complete institutional context for a specific decision."""
        try:
            # Get decision details
            decision_result = supa.table('decision_registry').select('*').eq('id', decision_id).eq('org_id', self.org_id).execute()
            
            if not decision_result.data:
                return {"error": f"Decision {decision_id} not found"}
            
            decision = decision_result.data[0]
            
            # Get complete context using all systems
            context = {
                'decision_details': decision,
                'historical_analysis': self.analyze_decision(decision),
                'ripple_effects': self.knowledge_graph.find_decision_ripple_effects(decision_id),
                'related_patterns': self._get_related_patterns(decision),
                'institutional_wisdom': self._get_institutional_wisdom(decision),
                'member_perspectives': self._analyze_member_perspectives(decision),
                'implementation_insights': self._get_implementation_insights(decision),
                'lessons_learned': self._extract_lessons_learned(decision_id)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get complete context for {decision_id}: {e}")
            return {"error": str(e)}
    
    def analyze_governance_trends(self, time_period_months: int = 24) -> Dict[str, Any]:
        """Analyze governance trends over specified time period."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_period_months * 30)
        
        trends = {
            'decision_velocity': self._analyze_decision_velocity(start_date, end_date),
            'topic_evolution': self._analyze_topic_evolution(start_date, end_date),
            'member_engagement_trends': self._analyze_member_engagement_trends(start_date, end_date),
            'financial_decision_trends': self._analyze_financial_trends(start_date, end_date),
            'success_rate_trends': self._analyze_success_rate_trends(start_date, end_date),
            'complexity_trends': self._analyze_complexity_trends(start_date, end_date),
            'seasonal_patterns': self._identify_seasonal_governance_patterns(start_date, end_date)
        }
        
        return trends
    
    def generate_board_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights about board performance and patterns."""
        insights = {
            'overall_performance': self._assess_overall_board_performance(),
            'decision_making_efficiency': self._analyze_decision_efficiency(),
            'member_contribution_analysis': self._analyze_member_contributions(),
            'governance_health_score': self._calculate_governance_health_score(),
            'improvement_opportunities': self._identify_improvement_opportunities(),
            'best_practices_identified': self._identify_best_practices(),
            'risk_assessment': self._assess_governance_risks(),
            'strategic_recommendations': self._generate_strategic_recommendations_board()
        }
        
        return insights
    
    # Helper methods for decision analysis
    
    def _find_similar_decisions(self, new_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar decisions using multiple similarity metrics."""
        try:
            # Get all historical decisions
            all_decisions = supa.table('decision_registry').select('*').eq('org_id', self.org_id).execute()
            
            if not all_decisions.data:
                return []
            
            # Calculate similarity scores
            similar_decisions = []
            for decision in all_decisions.data:
                similarity_score = self._calculate_similarity_score(new_decision, decision)
                
                if similarity_score > 0.3:  # Threshold for similarity
                    decision['similarity_score'] = similarity_score
                    similar_decisions.append(decision)
            
            # Sort by similarity and return top matches
            similar_decisions.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_decisions[:15]  # Return top 15 matches
            
        except Exception as e:
            logger.error(f"Failed to find similar decisions: {e}")
            return []
    
    def _calculate_similarity_score(self, decision1: Dict, decision2: Dict) -> float:
        """Calculate comprehensive similarity score between decisions."""
        score = 0.0
        
        # Type similarity (highest weight)
        if decision1.get('decision_type') == decision2.get('decision_type'):
            score += 0.4
        
        # Amount similarity
        amount1 = decision1.get('amount_involved', 0) or 0
        amount2 = decision2.get('amount_involved', 0) or 0
        
        if amount1 > 0 and amount2 > 0:
            ratio = min(amount1, amount2) / max(amount1, amount2)
            score += ratio * 0.2
        
        # Tag similarity
        tags1 = set(decision1.get('tags', []))
        tags2 = set(decision2.get('tags', []))
        
        if tags1 and tags2:
            tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            score += tag_similarity * 0.2
        
        # Temporal proximity (recent decisions are more relevant)
        try:
            date1 = datetime.fromisoformat(decision1.get('date', '2020-01-01'))
            date2 = datetime.fromisoformat(decision2.get('date', '2020-01-01'))
            
            days_diff = abs((date1 - date2).days)
            temporal_score = max(0, 1 - (days_diff / 1095))  # 3 years max relevance
            score += temporal_score * 0.2
        except:
            pass
        
        return min(1.0, score)
    
    def _analyze_precedents(self, similar_decisions: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze precedent decisions with comprehensive details."""
        precedents = []
        
        for decision in similar_decisions[:10]:  # Top 10 precedents
            precedent = {
                'date': decision.get('date'),
                'title': decision.get('title'),
                'outcome': decision.get('outcome'),
                'vote_breakdown': {
                    'for': decision.get('vote_count_for', 0),
                    'against': decision.get('vote_count_against', 0),
                    'abstain': decision.get('vote_count_abstain', 0)
                },
                'implementation_timeline': decision.get('implementation_timeline'),
                'actual_vs_projected_cost': self._calculate_cost_variance(decision),
                'member_feedback_summary': decision.get('member_feedback'),
                'lessons_learned': decision.get('lessons_learned'),
                'success_factors': self._extract_success_factors(decision),
                'complications': self._extract_complications(decision),
                'similarity_score': decision.get('similarity_score', 0)
            }
            
            precedents.append(precedent)
        
        return precedents
    
    def _calculate_success_probability(self, similar_decisions: List[Dict]) -> float:
        """Calculate success probability based on historical outcomes."""
        if not similar_decisions:
            return 0.5  # Default 50% when no data
        
        successful_outcomes = ['approved', 'passed', 'implemented', 'successful']
        
        # Weight by similarity score
        weighted_success = 0
        total_weight = 0
        
        for decision in similar_decisions:
            weight = decision.get('similarity_score', 0.5)
            outcome = decision.get('outcome', '').lower()
            
            if outcome in successful_outcomes:
                weighted_success += weight
            
            total_weight += weight
        
        return weighted_success / total_weight if total_weight > 0 else 0.5
    
    def _analyze_timeline_patterns(self, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze typical timeline patterns for similar decisions."""
        if not similar_decisions:
            return {}
        
        # Extract timeline data
        approval_times = []
        implementation_times = []
        
        for decision in similar_decisions:
            # Calculate approval time (proposal to decision)
            if decision.get('proposal_date') and decision.get('decision_date'):
                try:
                    proposal_date = datetime.fromisoformat(decision['proposal_date'])
                    decision_date = datetime.fromisoformat(decision['decision_date'])
                    approval_days = (decision_date - proposal_date).days
                    if approval_days >= 0:
                        approval_times.append(approval_days)
                except:
                    pass
            
            # Calculate implementation time
            if decision.get('decision_date') and decision.get('implementation_date'):
                try:
                    decision_date = datetime.fromisoformat(decision['decision_date'])
                    impl_date = datetime.fromisoformat(decision['implementation_date'])
                    impl_days = (impl_date - decision_date).days
                    if impl_days >= 0:
                        implementation_times.append(impl_days)
                except:
                    pass
        
        timeline_analysis = {}
        
        if approval_times:
            timeline_analysis['approval_timeline'] = {
                'average_days': statistics.mean(approval_times),
                'median_days': statistics.median(approval_times),
                'range': {'min': min(approval_times), 'max': max(approval_times)},
                'typical_range': f"{int(statistics.median(approval_times))} - {int(statistics.mean(approval_times) + statistics.stdev(approval_times))}" if len(approval_times) > 1 else f"{int(statistics.median(approval_times))}"
            }
        
        if implementation_times:
            timeline_analysis['implementation_timeline'] = {
                'average_days': statistics.mean(implementation_times),
                'median_days': statistics.median(implementation_times),
                'range': {'min': min(implementation_times), 'max': max(implementation_times)}
            }
        
        return timeline_analysis
    
    def _identify_common_amendments(self, similar_decisions: List[Dict]) -> List[str]:
        """Identify commonly proposed amendments for similar decisions."""
        amendments = []
        
        for decision in similar_decisions:
            decision_amendments = decision.get('amendments', [])
            if isinstance(decision_amendments, list):
                amendments.extend(decision_amendments)
            elif isinstance(decision_amendments, str):
                amendments.append(decision_amendments)
        
        # Count frequency of amendment types
        amendment_frequency = defaultdict(int)
        for amendment in amendments:
            if amendment:
                amendment_frequency[amendment.lower()] += 1
        
        # Return most common amendments
        common_amendments = sorted(amendment_frequency.items(), key=lambda x: x[1], reverse=True)
        return [amendment for amendment, freq in common_amendments[:5] if freq > 1]
    
    def _identify_risk_factors(self, new_decision: Dict, similar_decisions: List[Dict]) -> List[str]:
        """Identify potential risk factors for the new decision."""
        risk_factors = []
        
        # Amount-based risks
        amount = new_decision.get('amount_involved', 0) or 0
        if amount > 0:
            historical_amounts = [d.get('amount_involved', 0) for d in similar_decisions if d.get('amount_involved')]
            if historical_amounts:
                avg_amount = statistics.mean(historical_amounts)
                if amount > avg_amount * 1.5:
                    risk_factors.append(f"Amount ${amount:,.2f} is {amount/avg_amount:.1f}x higher than historical average")
        
        # Timeline risks
        if new_decision.get('urgency') == 'high':
            rushed_decisions = [d for d in similar_decisions if d.get('urgency') == 'high']
            if rushed_decisions:
                rushed_success_rate = len([d for d in rushed_decisions if d.get('outcome') in ['approved', 'passed']]) / len(rushed_decisions)
                if rushed_success_rate < 0.6:
                    risk_factors.append(f"High urgency decisions have only {rushed_success_rate:.0%} success rate historically")
        
        # Precedent deviation risks
        if not similar_decisions:
            risk_factors.append("No historical precedent found for this type of decision")
        
        # Seasonal risks
        decision_month = datetime.now().month
        if decision_month in [6, 7, 8, 12]:  # Summer and December
            risk_factors.append("Proposed during period of typically lower member engagement")
        
        return risk_factors
    
    def _predict_member_sentiment(self, new_decision: Dict) -> Dict[str, Any]:
        """Predict member sentiment based on historical voting patterns."""
        try:
            # Get member voting patterns
            member_insights = supa.table('board_member_insights').select('*').eq('org_id', self.org_id).execute()
            
            sentiment_prediction = {
                'overall_support_likelihood': 0.5,
                'key_supporters': [],
                'potential_opposition': [],
                'swing_votes': [],
                'engagement_forecast': 'medium'
            }
            
            if member_insights.data:
                # Analyze member patterns for decision type
                decision_type = new_decision.get('decision_type', 'general')
                
                for member in member_insights.data:
                    voting_pattern = member.get('voting_pattern_summary', {})
                    type_support = voting_pattern.get(decision_type, {}).get('support_rate', 0.5)
                    
                    member_data = {
                        'name': member.get('member_name'),
                        'predicted_support': type_support,
                        'historical_engagement': member.get('meeting_attendance_rate', 0.8),
                        'influence_level': member.get('influence_score', 0.5)
                    }
                    
                    if type_support > 0.7:
                        sentiment_prediction['key_supporters'].append(member_data)
                    elif type_support < 0.3:
                        sentiment_prediction['potential_opposition'].append(member_data)
                    else:
                        sentiment_prediction['swing_votes'].append(member_data)
                
                # Calculate overall support
                total_members = len(member_insights.data)
                if total_members > 0:
                    avg_support = sum(m.get('voting_pattern_summary', {}).get(decision_type, {}).get('support_rate', 0.5) 
                                    for m in member_insights.data) / total_members
                    sentiment_prediction['overall_support_likelihood'] = avg_support
            
            return sentiment_prediction
            
        except Exception as e:
            logger.error(f"Failed to predict member sentiment: {e}")
            return {'overall_support_likelihood': 0.5, 'note': 'Limited data available'}
    
    def _forecast_financial_impact(self, new_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Forecast financial impact based on historical patterns."""
        forecast = {
            'projected_cost': new_decision.get('amount_involved', 0),
            'cost_variance_range': {'low': 0, 'high': 0},
            'historical_overrun_rate': 0,
            'budget_risk_assessment': 'medium',
            'cost_factors': []
        }
        
        # Analyze historical cost variances
        cost_variances = []
        for decision in similar_decisions:
            initial_cost = decision.get('amount_involved', 0)
            actual_cost = decision.get('actual_cost')
            
            if initial_cost > 0 and actual_cost:
                variance = (actual_cost - initial_cost) / initial_cost
                cost_variances.append(variance)
        
        if cost_variances:
            avg_variance = statistics.mean(cost_variances)
            std_variance = statistics.stdev(cost_variances) if len(cost_variances) > 1 else 0
            
            projected_cost = new_decision.get('amount_involved', 0)
            
            forecast.update({
                'projected_cost': projected_cost,
                'cost_variance_range': {
                    'low': projected_cost * (1 + avg_variance - std_variance),
                    'high': projected_cost * (1 + avg_variance + std_variance)
                },
                'historical_overrun_rate': len([v for v in cost_variances if v > 0]) / len(cost_variances),
                'budget_risk_assessment': 'high' if avg_variance > 0.2 else 'medium' if avg_variance > 0.1 else 'low'
            })
        
        return forecast
    
    def _identify_comprehensive_patterns(self, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Identify comprehensive patterns in similar decisions."""
        if not similar_decisions:
            return {}
        
        patterns = {
            'voting_patterns': self._analyze_voting_patterns(similar_decisions),
            'seasonal_patterns': self._analyze_seasonal_patterns(similar_decisions),
            'financial_patterns': self._analyze_financial_patterns(similar_decisions),
            'implementation_patterns': self._analyze_implementation_patterns(similar_decisions),
            'amendment_patterns': self._analyze_amendment_patterns(similar_decisions)
        }
        
        return patterns
    
    def _assess_analysis_confidence(self, similar_decisions: List[Dict]) -> str:
        """Assess confidence level of the analysis."""
        if len(similar_decisions) >= 10:
            return 'high'
        elif len(similar_decisions) >= 5:
            return 'medium'
        elif len(similar_decisions) >= 2:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_strategic_recommendations(self, new_decision: Dict, similar_decisions: List[Dict]) -> List[str]:
        """Generate strategic recommendations based on analysis."""
        recommendations = []
        
        # Based on success probability
        success_prob = self._calculate_success_probability(similar_decisions)
        if success_prob < 0.6:
            recommendations.append("Consider preliminary stakeholder consultation to build support")
            recommendations.append("Review similar failed proposals to identify potential issues")
        
        # Based on amount
        amount = new_decision.get('amount_involved', 0)
        if amount > 0:
            historical_amounts = [d.get('amount_involved', 0) for d in similar_decisions if d.get('amount_involved')]
            if historical_amounts and amount > statistics.mean(historical_amounts) * 1.5:
                recommendations.append("Consider phased implementation to reduce financial risk")
                recommendations.append("Prepare detailed cost-benefit analysis for the higher amount")
        
        # Based on timing
        if datetime.now().month in [6, 7, 8]:
            recommendations.append("Consider timing - summer months typically have lower engagement")
        
        # Based on patterns
        risk_factors = self._identify_risk_factors(new_decision, similar_decisions)
        if risk_factors:
            recommendations.append("Address identified risk factors before proceeding")
        
        return recommendations
    
    # Additional helper methods would continue here...
    # (Implementation of remaining methods for comprehensive governance intelligence)
    
    def _initialize_intelligence_cache(self):
        """Initialize intelligence cache with current data."""
        self.last_cache_update = datetime.now()
        
        # Cache will be populated as methods are called
        logger.info("Governance intelligence cache initialized")
    
    def _extract_comprehensive_features(self, proposal: Dict) -> Dict[str, Any]:
        """Extract comprehensive features from a proposal for analysis."""
        return {
            'decision_type': proposal.get('decision_type'),
            'amount_range': self._categorize_amount(proposal.get('amount_involved', 0)),
            'urgency': proposal.get('urgency', 'normal'),
            'seasonality': datetime.now().month,
            'complexity': self._assess_complexity(proposal),
            'stakeholder_impact': self._assess_stakeholder_impact(proposal)
        }
    
    def _categorize_amount(self, amount: float) -> str:
        """Categorize amount into ranges."""
        if amount == 0:
            return 'no_cost'
        elif amount < 10000:
            return 'low'
        elif amount < 50000:
            return 'medium'
        elif amount < 100000:
            return 'high'
        else:
            return 'very_high'
    
    def _assess_complexity(self, proposal: Dict) -> str:
        """Assess proposal complexity."""
        # Simple heuristic based on description length and keywords
        description = proposal.get('description', '')
        
        complexity_indicators = ['multiple', 'phases', 'committee', 'coordination', 'approval', 'review']
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in description.lower())
        
        if indicator_count >= 3:
            return 'high'
        elif indicator_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_stakeholder_impact(self, proposal: Dict) -> str:
        """Assess stakeholder impact level."""
        # Based on keywords and amount
        description = proposal.get('description', '').lower()
        amount = proposal.get('amount_involved', 0)
        
        high_impact_keywords = ['all members', 'membership', 'dues', 'fees', 'facility']
        
        if any(keyword in description for keyword in high_impact_keywords) or amount > 50000:
            return 'high'
        elif amount > 10000:
            return 'medium'
        else:
            return 'low'

# Main API functions

def analyze_decision_comprehensive(org_id: str, decision_data: Dict[str, Any]) -> Dict[str, Any]:
    """Provide comprehensive analysis for a decision."""
    intelligence = GovernanceIntelligence(org_id)
    analysis = intelligence.analyze_decision(decision_data)
    return asdict(analysis)

def predict_decision_outcome(org_id: str, proposal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict outcome for a specific proposal."""
    intelligence = GovernanceIntelligence(org_id)
    prediction = intelligence.predict_outcome(proposal_data)
    return asdict(prediction)

def get_decision_context(org_id: str, decision_id: str) -> Dict[str, Any]:
    """Get complete context for a specific decision."""
    intelligence = GovernanceIntelligence(org_id)
    return intelligence.get_complete_context(decision_id)

def analyze_governance_trends(org_id: str, months: int = 24) -> Dict[str, Any]:
    """Analyze governance trends over time period."""
    intelligence = GovernanceIntelligence(org_id)
    return intelligence.analyze_governance_trends(months)

def generate_board_insights(org_id: str) -> Dict[str, Any]:
    """Generate comprehensive board performance insights."""
    intelligence = GovernanceIntelligence(org_id)
    return intelligence.generate_board_insights()