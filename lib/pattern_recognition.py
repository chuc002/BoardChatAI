"""
Pattern Recognition Engine - Advanced Governance Pattern Analysis

This module implements comprehensive pattern recognition for governance decisions,
member behavior, financial patterns, and organizational trends.
"""

import json
import uuid
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import statistics
import numpy as np
from collections import defaultdict

from lib.supa import supa

logger = logging.getLogger(__name__)

@dataclass
class DecisionPattern:
    """Represents a recognized decision pattern."""
    pattern_name: str
    pattern_type: str
    trigger_conditions: Dict
    success_rate: float
    approval_rate: float
    typical_timeline_days: int
    average_cost: Optional[float] = None
    risk_factors: List[str] = None
    success_factors: List[str] = None
    examples: List[str] = None
    confidence_score: float = 0.5

@dataclass
class GovernanceTrend:
    """Represents a governance trend analysis."""
    trend_name: str
    trend_type: str
    trend_direction: str
    start_date: date
    magnitude: float
    confidence_level: float
    supporting_evidence: List[str] = None
    predictive_indicators: List[str] = None

@dataclass
class VotingPattern:
    """Represents member voting behavior patterns."""
    member_name: str
    voting_tendencies: Dict[str, float]
    influence_score: float
    consistency_score: float
    predictability_score: float
    policy_preferences: Dict = None
    alignment_patterns: Dict = None

class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine for governance intelligence.
    Identifies patterns in decisions, trends, voting, and organizational behavior.
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self._ensure_pattern_tables()
        logger.info("Pattern Recognition Engine initialized")
    
    def _ensure_pattern_tables(self):
        """Ensure pattern recognition tables exist."""
        try:
            # Test if pattern tables exist
            supa.table('decision_patterns').select('id').limit(1).execute()
            logger.info("Pattern recognition tables available")
        except Exception as e:
            logger.warning(f"Pattern tables may need creation: {e}")
    
    def analyze_decision_patterns(self, 
                                 pattern_type: Optional[str] = None,
                                 min_examples: int = 3) -> List[DecisionPattern]:
        """Analyze and identify decision patterns."""
        patterns = []
        
        try:
            # Get decisions from the database
            decisions_query = supa.table('decision_complete').select('*').eq('org_id', self.org_id)
            if pattern_type:
                decisions_query = decisions_query.eq('decision_type', pattern_type)
            
            decisions_result = decisions_query.execute()
            decisions = decisions_result.data or []
            
            if len(decisions) < min_examples:
                logger.info(f"Insufficient decisions ({len(decisions)}) for pattern analysis")
                return patterns
            
            # Group decisions by similar characteristics
            pattern_groups = self._group_decisions_by_pattern(decisions)
            
            for group_key, group_decisions in pattern_groups.items():
                if len(group_decisions) >= min_examples:
                    pattern = self._analyze_decision_group(group_key, group_decisions)
                    if pattern:
                        patterns.append(pattern)
                        # Store pattern in database
                        self._store_decision_pattern(pattern)
            
            logger.info(f"Identified {len(patterns)} decision patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze decision patterns: {e}")
            return patterns
    
    def identify_governance_trends(self, 
                                  trend_types: List[str] = None,
                                  time_window_days: int = 365) -> List[GovernanceTrend]:
        """Identify long-term governance trends."""
        trends = []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            
            # Get recent decisions and records
            decisions = supa.table('decision_complete').select('*').eq('org_id', self.org_id).gte('proposed_date', cutoff_date.date().isoformat()).execute()
            
            decisions_data = decisions.data or []
            
            # Analyze trends by type
            trend_types = trend_types or ['financial', 'membership', 'governance', 'facilities']
            
            for trend_type in trend_types:
                trend = self._analyze_trend_for_type(trend_type, decisions_data, time_window_days)
                if trend:
                    trends.append(trend)
                    # Store trend in database
                    self._store_governance_trend(trend)
            
            logger.info(f"Identified {len(trends)} governance trends")
            return trends
            
        except Exception as e:
            logger.error(f"Failed to identify governance trends: {e}")
            return trends
    
    def analyze_voting_patterns(self, member_name: Optional[str] = None) -> List[VotingPattern]:
        """Analyze member voting patterns and behaviors."""
        patterns = []
        
        try:
            # Get voting data from decisions
            decisions = supa.table('decision_complete').select('*').eq('org_id', self.org_id).execute()
            decisions_data = decisions.data or []
            
            # Extract voting information
            member_votes = self._extract_voting_data(decisions_data)
            
            if member_name:
                if member_name in member_votes:
                    pattern = self._analyze_member_voting_pattern(member_name, member_votes[member_name])
                    if pattern:
                        patterns.append(pattern)
            else:
                # Analyze all members
                for member, votes in member_votes.items():
                    if len(votes) >= 5:  # Minimum votes for pattern analysis
                        pattern = self._analyze_member_voting_pattern(member, votes)
                        if pattern:
                            patterns.append(pattern)
                            # Store pattern in database
                            self._store_voting_pattern(pattern)
            
            logger.info(f"Analyzed voting patterns for {len(patterns)} members")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze voting patterns: {e}")
            return patterns
    
    def predict_decision_outcome(self, 
                                decision_data: Dict,
                                use_patterns: bool = True) -> Dict[str, Any]:
        """Predict outcome of a proposed decision based on patterns."""
        prediction = {
            'success_probability': 0.5,
            'approval_probability': 0.5,
            'estimated_timeline_days': 30,
            'risk_factors': [],
            'success_factors': [],
            'confidence_level': 'medium',
            'similar_patterns': [],
            'recommendations': []
        }
        
        try:
            if use_patterns:
                # Find similar patterns
                similar_patterns = self._find_similar_patterns(decision_data)
                prediction['similar_patterns'] = similar_patterns
                
                if similar_patterns:
                    # Calculate weighted predictions based on similar patterns
                    success_rates = [p['success_rate'] for p in similar_patterns]
                    approval_rates = [p['approval_rate'] for p in similar_patterns]
                    timelines = [p['typical_timeline_days'] for p in similar_patterns]
                    
                    prediction['success_probability'] = statistics.mean(success_rates)
                    prediction['approval_probability'] = statistics.mean(approval_rates)
                    prediction['estimated_timeline_days'] = int(statistics.mean(timelines))
                    
                    # Aggregate risk and success factors
                    all_risk_factors = []
                    all_success_factors = []
                    
                    for pattern in similar_patterns:
                        all_risk_factors.extend(pattern.get('risk_factors', []))
                        all_success_factors.extend(pattern.get('success_factors', []))
                    
                    prediction['risk_factors'] = list(set(all_risk_factors))
                    prediction['success_factors'] = list(set(all_success_factors))
                    
                    # Set confidence based on pattern match quality
                    if len(similar_patterns) >= 3:
                        prediction['confidence_level'] = 'high'
                    elif len(similar_patterns) >= 1:
                        prediction['confidence_level'] = 'medium'
                    else:
                        prediction['confidence_level'] = 'low'
            
            # Add voting pattern analysis
            voting_prediction = self._predict_voting_outcome(decision_data)
            prediction.update(voting_prediction)
            
            # Generate recommendations
            prediction['recommendations'] = self._generate_outcome_recommendations(prediction, decision_data)
            
            logger.info(f"Generated decision prediction with {prediction['confidence_level']} confidence")
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to predict decision outcome: {e}")
            return prediction
    
    def get_pattern_insights(self, insight_type: str = 'comprehensive') -> Dict[str, Any]:
        """Get comprehensive insights from pattern analysis."""
        insights = {
            'decision_patterns': {},
            'governance_trends': {},
            'voting_patterns': {},
            'risk_analysis': {},
            'recommendations': [],
            'key_findings': []
        }
        
        try:
            # Decision pattern insights
            if insight_type in ['comprehensive', 'decisions']:
                decision_patterns = supa.table('decision_patterns').select('*').eq('org_id', self.org_id).execute()
                insights['decision_patterns'] = self._analyze_decision_pattern_insights(decision_patterns.data or [])
            
            # Governance trend insights
            if insight_type in ['comprehensive', 'trends']:
                trends = supa.table('governance_trends').select('*').eq('org_id', self.org_id).execute()
                insights['governance_trends'] = self._analyze_trend_insights(trends.data or [])
            
            # Voting pattern insights
            if insight_type in ['comprehensive', 'voting']:
                voting_patterns = supa.table('voting_patterns').select('*').eq('org_id', self.org_id).execute()
                insights['voting_patterns'] = self._analyze_voting_insights(voting_patterns.data or [])
            
            # Generate strategic recommendations
            insights['recommendations'] = self._generate_strategic_recommendations(insights)
            insights['key_findings'] = self._extract_key_findings(insights)
            
            logger.info(f"Generated {insight_type} pattern insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get pattern insights: {e}")
            return insights
    
    # Helper methods for pattern analysis
    
    def _group_decisions_by_pattern(self, decisions: List[Dict]) -> Dict[str, List[Dict]]:
        """Group decisions by similar patterns."""
        pattern_groups = defaultdict(list)
        
        for decision in decisions:
            # Create pattern key based on decision characteristics
            pattern_key = self._create_pattern_key(decision)
            pattern_groups[pattern_key].append(decision)
        
        return pattern_groups
    
    def _create_pattern_key(self, decision: Dict) -> str:
        """Create a pattern key for grouping similar decisions."""
        # Combine key characteristics into a pattern signature
        decision_type = decision.get('decision_type', 'general')
        
        # Categorize by financial impact
        cost = decision.get('budget_projected', 0) or 0
        cost_category = 'low' if cost < 5000 else 'medium' if cost < 25000 else 'high'
        
        # Categorize by complexity
        complexity = decision.get('complexity_score', 0.5)
        complexity_category = 'simple' if complexity < 0.3 else 'moderate' if complexity < 0.7 else 'complex'
        
        # Categorize by urgency
        urgency = decision.get('urgency_level', 'normal')
        
        return f"{decision_type}_{cost_category}_{complexity_category}_{urgency}"
    
    def _analyze_decision_group(self, pattern_key: str, decisions: List[Dict]) -> Optional[DecisionPattern]:
        """Analyze a group of similar decisions to identify patterns with enhanced analysis."""
        if len(decisions) < 2:
            return None
        
        try:
            # Enhanced approval analysis using your specification methods
            approval_analysis = self._analyze_approval_patterns_enhanced(decisions)
            
            # Calculate success rate (decisions marked as would_repeat)
            successful_decisions = [d for d in decisions if d.get('would_repeat') == True]
            success_rate = len(successful_decisions) / len(decisions)
            
            # Calculate approval rate with enhanced logic
            approved_decisions = [d for d in decisions if self._is_approved(d)]
            approval_rate = len(approved_decisions) / len(decisions)
            
            # Calculate average timeline with multiple methods
            avg_timeline = self._calculate_average_timeline(decisions)
            
            # Enhanced cost analysis
            cost_analysis = self._analyze_cost_patterns(decisions)
            avg_cost = cost_analysis.get('average_cost')
            
            # Extract enhanced risk and success factors
            risk_factors = self._extract_enhanced_factors(decisions, 'risk_assessment')
            success_factors = self._extract_enhanced_factors(decisions, 'success_metrics')
            
            # Add governance insights
            governance_insights = self._calculate_governance_insights(decisions)
            
            # Create enhanced pattern
            pattern = DecisionPattern(
                pattern_name=f"Pattern: {pattern_key.replace('_', ' ').title()}",
                pattern_type=pattern_key.split('_')[0],
                trigger_conditions={
                    'pattern_key': pattern_key,
                    'approval_thresholds': approval_analysis.get('thresholds', {}),
                    'seasonal_effects': approval_analysis.get('seasonal_patterns', {}),
                    'sponsor_influence': approval_analysis.get('sponsor_patterns', {})
                },
                success_rate=success_rate,
                approval_rate=approval_rate,
                typical_timeline_days=avg_timeline,
                average_cost=avg_cost,
                risk_factors=risk_factors,
                success_factors=success_factors,
                examples=[d.get('id') for d in decisions],
                confidence_score=self._calculate_enhanced_confidence(decisions, approval_analysis)
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to analyze decision group: {e}")
            return None
    
    def _extract_common_factors(self, decisions: List[Dict], field: str) -> List[str]:
        """Extract common factors from a field across decisions."""
        all_factors = []
        
        for decision in decisions:
            factors_data = decision.get(field, {})
            if isinstance(factors_data, dict):
                all_factors.extend(factors_data.keys())
            elif isinstance(factors_data, list):
                all_factors.extend(factors_data)
        
        # Count frequency and return most common
        factor_counts = defaultdict(int)
        for factor in all_factors:
            factor_counts[factor] += 1
        
        # Return factors that appear in at least 50% of decisions
        threshold = len(decisions) * 0.5
        common_factors = [factor for factor, count in factor_counts.items() if count >= threshold]
        
        return common_factors[:5]  # Return top 5
    
    def _analyze_trend_for_type(self, trend_type: str, decisions: List[Dict], time_window_days: int) -> Optional[GovernanceTrend]:
        """Analyze trend for a specific type."""
        try:
            # Filter decisions relevant to this trend type
            relevant_decisions = self._filter_decisions_by_trend_type(decisions, trend_type)
            
            if len(relevant_decisions) < 3:
                return None
            
            # Sort by date
            relevant_decisions.sort(key=lambda x: x.get('proposed_date', ''))
            
            # Analyze trend direction and magnitude
            trend_analysis = self._calculate_trend_metrics(relevant_decisions, trend_type)
            
            if trend_analysis:
                trend = GovernanceTrend(
                    trend_name=f"{trend_type.title()} Trend",
                    trend_type=trend_type,
                    trend_direction=trend_analysis['direction'],
                    start_date=datetime.fromisoformat(relevant_decisions[0]['proposed_date']).date(),
                    magnitude=trend_analysis['magnitude'],
                    confidence_level=trend_analysis['confidence'],
                    supporting_evidence=[d.get('id') for d in relevant_decisions],
                    predictive_indicators=trend_analysis.get('predictive_indicators', [])
                )
                
                return trend
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze trend for {trend_type}: {e}")
            return None
    
    def _filter_decisions_by_trend_type(self, decisions: List[Dict], trend_type: str) -> List[Dict]:
        """Filter decisions relevant to a specific trend type."""
        relevant_decisions = []
        
        for decision in decisions:
            decision_type = decision.get('decision_type', '').lower()
            description = decision.get('description', '').lower()
            title = decision.get('decision_title', '').lower()
            
            # Check if decision is relevant to the trend type
            if trend_type == 'financial':
                if ('fee' in description or 'cost' in description or 'budget' in description or 
                    'financial' in decision_type or decision.get('budget_projected')):
                    relevant_decisions.append(decision)
            elif trend_type == 'membership':
                if ('member' in description or 'admission' in description or 'category' in description or
                    'membership' in decision_type):
                    relevant_decisions.append(decision)
            elif trend_type == 'governance':
                if ('policy' in description or 'rule' in description or 'governance' in description or
                    'procedure' in description or 'governance' in decision_type):
                    relevant_decisions.append(decision)
            elif trend_type == 'facilities':
                if ('facility' in description or 'building' in description or 'maintenance' in description or
                    'improvement' in description or 'facilities' in decision_type):
                    relevant_decisions.append(decision)
        
        return relevant_decisions
    
    def _calculate_trend_metrics(self, decisions: List[Dict], trend_type: str) -> Optional[Dict]:
        """Calculate trend metrics for a set of decisions."""
        try:
            if len(decisions) < 2:
                return None
            
            # Extract relevant metrics based on trend type
            if trend_type == 'financial':
                values = []
                for decision in decisions:
                    cost = decision.get('budget_actual', decision.get('budget_projected', 0)) or 0
                    if cost > 0:
                        values.append(cost)
                
                if len(values) >= 2:
                    # Calculate trend direction
                    first_half = values[:len(values)//2]
                    second_half = values[len(values)//2:]
                    
                    avg_first = statistics.mean(first_half)
                    avg_second = statistics.mean(second_half)
                    
                    if avg_second > avg_first * 1.1:
                        direction = 'increasing'
                        magnitude = (avg_second - avg_first) / avg_first
                    elif avg_second < avg_first * 0.9:
                        direction = 'decreasing'
                        magnitude = (avg_first - avg_second) / avg_first
                    else:
                        direction = 'stable'
                        magnitude = 0.1
                    
                    confidence = min(1.0, len(values) / 10)
                    
                    return {
                        'direction': direction,
                        'magnitude': magnitude,
                        'confidence': confidence,
                        'predictive_indicators': ['budget_trends', 'cost_inflation']
                    }
            
            # Add similar logic for other trend types
            return {
                'direction': 'stable',
                'magnitude': 0.1,
                'confidence': 0.5,
                'predictive_indicators': []
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate trend metrics: {e}")
            return None
    
    def _extract_voting_data(self, decisions: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract voting data for each member."""
        member_votes = defaultdict(list)
        
        for decision in decisions:
            vote_details = decision.get('vote_details', {})
            if isinstance(vote_details, dict) and 'details' in vote_details:
                for vote_record in vote_details['details']:
                    if isinstance(vote_record, dict):
                        member = vote_record.get('member')
                        vote = vote_record.get('vote')
                        if member and vote:
                            member_votes[member].append({
                                'decision_id': decision.get('id'),
                                'vote': vote,
                                'decision_type': decision.get('decision_type'),
                                'proposed_date': decision.get('proposed_date'),
                                'complexity': decision.get('complexity_score', 0.5),
                                'cost': decision.get('budget_projected', 0) or 0
                            })
        
        return dict(member_votes)
    
    def _analyze_member_voting_pattern(self, member_name: str, votes: List[Dict]) -> Optional[VotingPattern]:
        """Analyze voting pattern for a specific member."""
        try:
            if len(votes) < 3:
                return None
            
            # Calculate voting tendencies
            vote_counts = defaultdict(int)
            for vote in votes:
                vote_counts[vote['vote']] += 1
            
            total_votes = len(votes)
            tendencies = {
                'approval_rate': vote_counts.get('for', 0) / total_votes,
                'opposition_rate': vote_counts.get('against', 0) / total_votes,
                'abstention_rate': vote_counts.get('abstain', 0) / total_votes
            }
            
            # Calculate consistency (how often they vote the same way)
            most_common_vote = max(vote_counts.keys(), key=lambda x: vote_counts[x])
            consistency_score = vote_counts[most_common_vote] / total_votes
            
            # Calculate predictability (variance in voting behavior)
            predictability_score = consistency_score  # Simplified
            
            # Estimate influence (placeholder - would need more complex analysis)
            influence_score = 0.5  # Default influence
            
            # Analyze policy preferences
            policy_preferences = self._analyze_policy_preferences(votes)
            
            pattern = VotingPattern(
                member_name=member_name,
                voting_tendencies=tendencies,
                influence_score=influence_score,
                consistency_score=consistency_score,
                predictability_score=predictability_score,
                policy_preferences=policy_preferences
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to analyze voting pattern for {member_name}: {e}")
            return None
    
    def _analyze_policy_preferences(self, votes: List[Dict]) -> Dict[str, float]:
        """Analyze policy preferences based on voting history."""
        preferences = defaultdict(list)
        
        for vote in votes:
            decision_type = vote.get('decision_type', 'general')
            vote_value = 1.0 if vote['vote'] == 'for' else -1.0 if vote['vote'] == 'against' else 0.0
            preferences[decision_type].append(vote_value)
        
        # Calculate average preference for each policy area
        policy_preferences = {}
        for policy_type, vote_values in preferences.items():
            if vote_values:
                avg_preference = statistics.mean(vote_values)
                policy_preferences[policy_type] = avg_preference
        
        return policy_preferences
    
    def _find_similar_patterns(self, decision_data: Dict) -> List[Dict]:
        """Find similar patterns for a proposed decision."""
        try:
            patterns = supa.table('decision_patterns').select('*').eq('org_id', self.org_id).execute()
            similar_patterns = []
            
            decision_type = decision_data.get('decision_type', 'general')
            
            for pattern in (patterns.data or []):
                # Check similarity based on decision type
                if pattern.get('pattern_type') == decision_type:
                    similar_patterns.append(pattern)
            
            # Sort by success rate and confidence
            similar_patterns.sort(key=lambda x: (x.get('success_rate', 0), x.get('confidence_score', 0)), reverse=True)
            
            return similar_patterns[:5]  # Return top 5 similar patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            return []
    
    def _predict_voting_outcome(self, decision_data: Dict) -> Dict[str, Any]:
        """Predict voting outcome based on member patterns."""
        voting_prediction = {
            'expected_for_votes': 0,
            'expected_against_votes': 0,
            'expected_abstentions': 0,
            'swing_voters': [],
            'strong_supporters': [],
            'likely_opposition': []
        }
        
        try:
            # Get voting patterns
            patterns = supa.table('voting_patterns').select('*').eq('org_id', self.org_id).execute()
            
            decision_type = decision_data.get('decision_type', 'general')
            
            for pattern in (patterns.data or []):
                member_name = pattern.get('member_name')
                tendencies = pattern.get('voting_tendencies', {})
                
                approval_rate = tendencies.get('approval_rate', 0.5)
                
                # Predict vote based on approval rate and decision type
                if approval_rate > 0.7:
                    voting_prediction['expected_for_votes'] += 1
                    voting_prediction['strong_supporters'].append(member_name)
                elif approval_rate < 0.3:
                    voting_prediction['expected_against_votes'] += 1
                    voting_prediction['likely_opposition'].append(member_name)
                else:
                    voting_prediction['swing_voters'].append(member_name)
                    # Assume swing voters split evenly
                    if len(voting_prediction['swing_voters']) % 2:
                        voting_prediction['expected_for_votes'] += 0.5
                    else:
                        voting_prediction['expected_against_votes'] += 0.5
            
            return voting_prediction
            
        except Exception as e:
            logger.error(f"Failed to predict voting outcome: {e}")
            return voting_prediction
    
    def _generate_outcome_recommendations(self, prediction: Dict, decision_data: Dict) -> List[str]:
        """Generate recommendations based on outcome prediction."""
        recommendations = []
        
        success_prob = prediction.get('success_probability', 0.5)
        approval_prob = prediction.get('approval_probability', 0.5)
        
        if success_prob < 0.5:
            recommendations.append("Consider revising the proposal to address identified risk factors")
        
        if approval_prob < 0.6:
            recommendations.append("Build stronger consensus before bringing to vote")
            recommendations.append("Address concerns of likely opposition members")
        
        if prediction.get('swing_voters'):
            recommendations.append(f"Focus on convincing swing voters: {', '.join(prediction['swing_voters'][:3])}")
        
        if prediction.get('estimated_timeline_days', 0) > 45:
            recommendations.append("Allow extra time for decision process - similar decisions take longer")
        
        return recommendations
    
    # Database storage methods
    
    def _store_decision_pattern(self, pattern: DecisionPattern):
        """Store decision pattern in database."""
        try:
            pattern_data = {
                'org_id': self.org_id,
                'pattern_name': pattern.pattern_name,
                'pattern_type': pattern.pattern_type,
                'trigger_conditions': pattern.trigger_conditions,
                'typical_timeline_days': pattern.typical_timeline_days,
                'success_rate': pattern.success_rate,
                'approval_rate': pattern.approval_rate,
                'average_cost': pattern.average_cost,
                'risk_factors': pattern.risk_factors or [],
                'success_factors': pattern.success_factors or [],
                'examples': pattern.examples or [],
                'confidence_score': pattern.confidence_score
            }
            
            # Check if pattern already exists
            existing = supa.table('decision_patterns').select('id').eq('org_id', self.org_id).eq('pattern_name', pattern.pattern_name).execute()
            
            if existing.data:
                # Update existing
                result = supa.table('decision_patterns').update(pattern_data).eq('org_id', self.org_id).eq('pattern_name', pattern.pattern_name).execute()
            else:
                # Insert new
                result = supa.table('decision_patterns').insert(pattern_data).execute()
            
            logger.info(f"Stored decision pattern: {pattern.pattern_name}")
            
        except Exception as e:
            logger.error(f"Failed to store decision pattern: {e}")
    
    def _store_governance_trend(self, trend: GovernanceTrend):
        """Store governance trend in database."""
        try:
            trend_data = {
                'org_id': self.org_id,
                'trend_name': trend.trend_name,
                'trend_type': trend.trend_type,
                'trend_direction': trend.trend_direction,
                'start_date': trend.start_date.isoformat(),
                'magnitude': trend.magnitude,
                'confidence_level': trend.confidence_level,
                'supporting_decisions': trend.supporting_evidence or [],
                'predictive_indicators': trend.predictive_indicators or []
            }
            
            result = supa.table('governance_trends').insert(trend_data).execute()
            logger.info(f"Stored governance trend: {trend.trend_name}")
            
        except Exception as e:
            logger.error(f"Failed to store governance trend: {e}")
    
    def _store_voting_pattern(self, pattern: VotingPattern):
        """Store voting pattern in database."""
        try:
            pattern_data = {
                'org_id': self.org_id,
                'member_name': pattern.member_name,
                'voting_tendencies': pattern.voting_tendencies,
                'influence_score': pattern.influence_score,
                'consistency_score': pattern.consistency_score,
                'predictability_score': pattern.predictability_score,
                'policy_preferences': pattern.policy_preferences or {},
                'alignment_patterns': pattern.alignment_patterns or {}
            }
            
            # Check if pattern already exists
            existing = supa.table('voting_patterns').select('id').eq('org_id', self.org_id).eq('member_name', pattern.member_name).execute()
            
            if existing.data:
                # Update existing
                result = supa.table('voting_patterns').update(pattern_data).eq('org_id', self.org_id).eq('member_name', pattern.member_name).execute()
            else:
                # Insert new
                result = supa.table('voting_patterns').insert(pattern_data).execute()
            
            logger.info(f"Stored voting pattern: {pattern.member_name}")
            
        except Exception as e:
            logger.error(f"Failed to store voting pattern: {e}")
    
    # Insight analysis methods
    
    def _analyze_decision_pattern_insights(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze insights from decision patterns."""
        insights = {
            'total_patterns': len(patterns),
            'average_success_rate': 0.0,
            'most_successful_patterns': [],
            'risk_factors': [],
            'recommendations': []
        }
        
        if patterns:
            success_rates = [p.get('success_rate', 0) for p in patterns]
            insights['average_success_rate'] = statistics.mean(success_rates)
            
            # Find most successful patterns
            sorted_patterns = sorted(patterns, key=lambda x: x.get('success_rate', 0), reverse=True)
            insights['most_successful_patterns'] = sorted_patterns[:3]
            
            # Aggregate risk factors
            all_risks = []
            for pattern in patterns:
                all_risks.extend(pattern.get('risk_factors', []))
            
            risk_counts = defaultdict(int)
            for risk in all_risks:
                risk_counts[risk] += 1
            
            insights['risk_factors'] = [{'risk': risk, 'frequency': count} 
                                      for risk, count in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        return insights
    
    def _analyze_trend_insights(self, trends: List[Dict]) -> Dict[str, Any]:
        """Analyze insights from governance trends."""
        insights = {
            'total_trends': len(trends),
            'trend_directions': {},
            'high_confidence_trends': [],
            'emerging_patterns': []
        }
        
        if trends:
            # Count trend directions
            direction_counts = defaultdict(int)
            for trend in trends:
                direction_counts[trend.get('trend_direction', 'unknown')] += 1
            
            insights['trend_directions'] = dict(direction_counts)
            
            # High confidence trends
            high_confidence = [t for t in trends if t.get('confidence_level', 0) > 0.7]
            insights['high_confidence_trends'] = high_confidence
        
        return insights
    
    def _analyze_voting_insights(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze insights from voting patterns."""
        insights = {
            'total_members_analyzed': len(patterns),
            'average_consistency': 0.0,
            'most_influential_members': [],
            'voting_tendencies_summary': {}
        }
        
        if patterns:
            # Average consistency
            consistency_scores = [p.get('consistency_score', 0) for p in patterns]
            insights['average_consistency'] = statistics.mean(consistency_scores)
            
            # Most influential members
            sorted_by_influence = sorted(patterns, key=lambda x: x.get('influence_score', 0), reverse=True)
            insights['most_influential_members'] = sorted_by_influence[:5]
            
            # Voting tendencies summary
            all_tendencies = [p.get('voting_tendencies', {}) for p in patterns]
            if all_tendencies:
                avg_approval = statistics.mean([t.get('approval_rate', 0.5) for t in all_tendencies])
                avg_opposition = statistics.mean([t.get('opposition_rate', 0.5) for t in all_tendencies])
                
                insights['voting_tendencies_summary'] = {
                    'average_approval_rate': avg_approval,
                    'average_opposition_rate': avg_opposition
                }
        
        return insights
    
    def _generate_strategic_recommendations(self, insights: Dict) -> List[str]:
        """Generate strategic recommendations based on insights."""
        recommendations = []
        
        # Decision pattern recommendations
        decision_insights = insights.get('decision_patterns', {})
        if decision_insights.get('average_success_rate', 0) < 0.6:
            recommendations.append("Review decision-making process - success rate below optimal threshold")
        
        # Trend recommendations
        trend_insights = insights.get('governance_trends', {})
        trend_directions = trend_insights.get('trend_directions', {})
        if trend_directions.get('decreasing', 0) > trend_directions.get('increasing', 0):
            recommendations.append("Monitor declining trends - consider proactive interventions")
        
        # Voting pattern recommendations
        voting_insights = insights.get('voting_patterns', {})
        if voting_insights.get('average_consistency', 0) < 0.6:
            recommendations.append("Improve communication to increase voting consistency")
        
        return recommendations
    
    def _extract_key_findings(self, insights: Dict) -> List[str]:
        """Extract key findings from comprehensive insights."""
        findings = []
        
        # Decision patterns
        decision_insights = insights.get('decision_patterns', {})
        total_patterns = decision_insights.get('total_patterns', 0)
        if total_patterns > 0:
            findings.append(f"Identified {total_patterns} distinct decision patterns")
            
            avg_success = decision_insights.get('average_success_rate', 0)
            findings.append(f"Average decision success rate: {avg_success:.1%}")
        
        # Governance trends
        trend_insights = insights.get('governance_trends', {})
        total_trends = trend_insights.get('total_trends', 0)
        if total_trends > 0:
            findings.append(f"Tracking {total_trends} governance trends")
        
        # Voting patterns
        voting_insights = insights.get('voting_patterns', {})
        members_analyzed = voting_insights.get('total_members_analyzed', 0)
        if members_analyzed > 0:
            findings.append(f"Analyzed voting patterns for {members_analyzed} members")
            
            avg_consistency = voting_insights.get('average_consistency', 0)
            findings.append(f"Average voting consistency: {avg_consistency:.1%}")
        
        return findings
    
    # Enhanced analysis methods based on your specification
    
    def _analyze_approval_patterns_enhanced(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Enhanced approval pattern analysis with detailed categorization."""
        approval_analysis = {
            'by_amount': defaultdict(list),
            'by_type': defaultdict(list),
            'by_committee': defaultdict(list),
            'by_season': defaultdict(list),
            'by_sponsor': defaultdict(list),
            'thresholds': {},
            'seasonal_patterns': {},
            'sponsor_patterns': {}
        }
        
        for decision in decisions:
            outcome = self._get_decision_outcome(decision)
            amount = decision.get('budget_projected', 0) or decision.get('budget_actual', 0) or 0
            decision_type = decision.get('decision_type', 'general')
            committee = decision.get('committee', 'board')
            date = decision.get('proposed_date')
            sponsor = decision.get('proposed_by')
            
            # Amount-based patterns
            amount_range = self._get_amount_range(amount)
            approval_analysis['by_amount'][amount_range].append({
                'approved': outcome == 'approved',
                'amount': amount,
                'type': decision_type,
                'vote_margin': self._calculate_vote_margin(decision)
            })
            
            # Type-based patterns
            approval_analysis['by_type'][decision_type].append({
                'approved': outcome == 'approved',
                'amount': amount,
                'timeline': self._calculate_decision_timeline(decision),
                'amendments': len(decision.get('modifications', []))
            })
            
            # Committee patterns
            approval_analysis['by_committee'][committee].append({
                'approved': outcome == 'approved',
                'amount': amount,
                'type': decision_type,
                'preparation_time': self._calculate_prep_time(decision)
            })
            
            # Seasonal patterns
            if date:
                season = self._get_season(date)
                approval_analysis['by_season'][season].append({
                    'approved': outcome == 'approved',
                    'type': decision_type,
                    'amount': amount
                })
                
            # Sponsor patterns
            if sponsor:
                approval_analysis['by_sponsor'][sponsor].append({
                    'approved': outcome == 'approved',
                    'amount': amount,
                    'type': decision_type
                })
        
        # Calculate success rates and identify patterns
        for category, data in approval_analysis.items():
            if category in ['thresholds', 'seasonal_patterns', 'sponsor_patterns']:
                continue
                
            for subcategory, items in data.items():
                if items:
                    total = len(items)
                    approved = sum(1 for item in items if item['approved'])
                    approval_rate = approved / total
                    
                    # Store calculated patterns
                    if category == 'by_amount':
                        approval_analysis['thresholds'][subcategory] = approval_rate
                    elif category == 'by_season':
                        approval_analysis['seasonal_patterns'][subcategory] = approval_rate
                    elif category == 'by_sponsor':
                        approval_analysis['sponsor_patterns'][subcategory] = approval_rate
        
        return approval_analysis
    
    def _get_decision_outcome(self, decision: Dict) -> str:
        """Extract decision outcome with multiple fallback methods."""
        # Check multiple fields for outcome
        if decision.get('would_repeat') == True:
            return 'approved'
        elif decision.get('would_repeat') == False:
            return 'rejected'
        
        vote_details = decision.get('vote_details', {})
        if isinstance(vote_details, dict):
            if vote_details.get('approved') == True:
                return 'approved'
            elif vote_details.get('approved') == False:
                return 'rejected'
        
        # Check implementation status
        if decision.get('implementation_completion_date'):
            return 'approved'
        
        return 'unknown'
    
    def _get_amount_range(self, amount: float) -> str:
        """Categorize amounts into ranges."""
        if amount < 1000:
            return 'under_1k'
        elif amount < 5000:
            return '1k_5k'
        elif amount < 25000:
            return '5k_25k'
        elif amount < 100000:
            return '25k_100k'
        else:
            return 'over_100k'
    
    def _get_season(self, date_str: str) -> str:
        """Get season from date string."""
        try:
            if isinstance(date_str, str):
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date = date_str
            
            month = date.month
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        except:
            return 'unknown'
    
    def _calculate_vote_margin(self, decision: Dict) -> float:
        """Calculate vote margin from decision data."""
        vote_details = decision.get('vote_details', {})
        
        if isinstance(vote_details, dict):
            vote_for = vote_details.get('for', 0)
            vote_against = vote_details.get('against', 0)
            total = vote_for + vote_against
            
            if total > 0:
                return (vote_for - vote_against) / total
        
        # Try vote_margin field
        vote_margin = decision.get('vote_margin')
        if vote_margin is not None:
            return vote_margin
        
        return 0.0
    
    def _calculate_decision_timeline(self, decision: Dict) -> int:
        """Calculate timeline for a decision."""
        try:
            proposed_date = decision.get('proposed_date')
            vote_date = decision.get('vote_date')
            
            if proposed_date and vote_date:
                proposed = datetime.fromisoformat(proposed_date)
                voted = datetime.fromisoformat(vote_date)
                return (voted - proposed).days
        except:
            pass
        
        return 0
    
    def _calculate_prep_time(self, decision: Dict) -> int:
        """Calculate preparation time for a decision."""
        # This would need to be calculated based on when preparation started
        # For now, return a default or calculate from available data
        timeline = self._calculate_decision_timeline(decision)
        return max(timeline - 7, 0)  # Assume 7 days for voting, rest is prep
    
    def _is_approved(self, decision: Dict) -> bool:
        """Check if a decision was approved using multiple methods."""
        return self._get_decision_outcome(decision) == 'approved'
    
    def _calculate_average_timeline(self, decisions: List[Dict]) -> int:
        """Calculate average timeline across decisions."""
        timelines = []
        
        for decision in decisions:
            timeline = self._calculate_decision_timeline(decision)
            if timeline > 0:
                timelines.append(timeline)
        
        return int(statistics.mean(timelines)) if timelines else 30
    
    def _analyze_cost_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze cost patterns in decisions."""
        costs = []
        actual_costs = []
        variances = []
        
        for decision in decisions:
            projected = decision.get('budget_projected', 0) or 0
            actual = decision.get('budget_actual', 0) or 0
            
            if projected > 0:
                costs.append(projected)
                
                if actual > 0:
                    actual_costs.append(actual)
                    variance = (actual - projected) / projected
                    variances.append(variance)
        
        analysis = {
            'average_cost': statistics.mean(costs) if costs else 0,
            'median_cost': statistics.median(costs) if costs else 0,
            'cost_variance': statistics.mean(variances) if variances else 0,
            'cost_accuracy': 1 - abs(statistics.mean(variances)) if variances else 0.8
        }
        
        return analysis
    
    def _extract_enhanced_factors(self, decisions: List[Dict], field: str) -> List[str]:
        """Enhanced factor extraction with better analysis."""
        all_factors = []
        
        for decision in decisions:
            factors_data = decision.get(field, {})
            
            if isinstance(factors_data, dict):
                # Extract keys and values
                all_factors.extend(factors_data.keys())
                for value in factors_data.values():
                    if isinstance(value, str):
                        all_factors.append(value)
                    elif isinstance(value, list):
                        all_factors.extend(value)
            elif isinstance(factors_data, list):
                all_factors.extend(factors_data)
            elif isinstance(factors_data, str):
                all_factors.append(factors_data)
        
        # Count frequency and return most common
        factor_counts = defaultdict(int)
        for factor in all_factors:
            if factor and isinstance(factor, str):
                factor_counts[factor] += 1
        
        # Return factors that appear in at least 30% of decisions
        threshold = max(1, len(decisions) * 0.3)
        common_factors = [factor for factor, count in factor_counts.items() if count >= threshold]
        
        return sorted(common_factors, key=lambda x: factor_counts[x], reverse=True)[:10]
    
    def _calculate_governance_insights(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate governance insights for decision group."""
        insights = {
            'decision_velocity': len(decisions),
            'average_complexity': statistics.mean([d.get('complexity_score', 0.5) for d in decisions]),
            'consensus_building': self._analyze_consensus_patterns(decisions),
            'implementation_success': self._analyze_implementation_success(decisions)
        }
        
        return insights
    
    def _analyze_consensus_patterns(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze consensus building patterns."""
        unanimous_count = 0
        total_with_votes = 0
        
        for decision in decisions:
            if decision.get('unanimous'):
                unanimous_count += 1
                total_with_votes += 1
            elif decision.get('vote_details'):
                total_with_votes += 1
        
        consensus_rate = unanimous_count / total_with_votes if total_with_votes > 0 else 0
        
        return {
            'unanimous_rate': consensus_rate,
            'consensus_building_needed': 1 - consensus_rate,
            'average_vote_margin': statistics.mean([
                abs(self._calculate_vote_margin(d)) for d in decisions
            ])
        }
    
    def _analyze_implementation_success(self, decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze implementation success patterns."""
        implemented = 0
        total_approved = 0
        
        for decision in decisions:
            if self._is_approved(decision):
                total_approved += 1
                if decision.get('implementation_completion_date'):
                    implemented += 1
        
        implementation_rate = implemented / total_approved if total_approved > 0 else 0
        
        return {
            'implementation_rate': implementation_rate,
            'completion_success': implementation_rate,
            'average_implementation_time': self._calculate_avg_implementation_time(decisions)
        }
    
    def _calculate_avg_implementation_time(self, decisions: List[Dict]) -> float:
        """Calculate average implementation time."""
        implementation_times = []
        
        for decision in decisions:
            start_date = decision.get('implementation_start_date')
            end_date = decision.get('implementation_completion_date')
            
            if start_date and end_date:
                try:
                    start = datetime.fromisoformat(start_date)
                    end = datetime.fromisoformat(end_date)
                    implementation_times.append((end - start).days)
                except:
                    continue
        
        return statistics.mean(implementation_times) if implementation_times else 90
    
    def _calculate_enhanced_confidence(self, decisions: List[Dict], approval_analysis: Dict) -> float:
        """Calculate enhanced confidence score."""
        base_confidence = min(1.0, len(decisions) / 10)
        
        # Boost confidence based on data quality
        quality_factors = []
        
        # Check for complete voting data
        complete_votes = sum(1 for d in decisions if d.get('vote_details'))
        vote_completeness = complete_votes / len(decisions)
        quality_factors.append(vote_completeness)
        
        # Check for financial data completeness
        complete_budgets = sum(1 for d in decisions if d.get('budget_projected') or d.get('budget_actual'))
        budget_completeness = complete_budgets / len(decisions)
        quality_factors.append(budget_completeness)
        
        # Check for outcome tracking
        tracked_outcomes = sum(1 for d in decisions if d.get('would_repeat') is not None)
        outcome_completeness = tracked_outcomes / len(decisions)
        quality_factors.append(outcome_completeness)
        
        quality_score = statistics.mean(quality_factors)
        
        return min(1.0, base_confidence * (0.5 + 0.5 * quality_score))


# API Functions

def analyze_governance_patterns(org_id: str) -> Dict[str, Any]:
    """Analyze governance patterns for an organization."""
    engine = PatternRecognitionEngine(org_id)
    
    # Analyze all pattern types
    decision_patterns = engine.analyze_decision_patterns()
    governance_trends = engine.identify_governance_trends()
    voting_patterns = engine.analyze_voting_patterns()
    
    return {
        'decision_patterns': [asdict(p) for p in decision_patterns],
        'governance_trends': [asdict(t) for t in governance_trends],
        'voting_patterns': [asdict(v) for v in voting_patterns],
        'analysis_timestamp': datetime.now().isoformat()
    }

def predict_proposal_outcome(org_id: str, proposal_data: Dict) -> Dict[str, Any]:
    """Predict outcome of a proposed decision."""
    engine = PatternRecognitionEngine(org_id)
    return engine.predict_decision_outcome(proposal_data)

def get_pattern_insights(org_id: str, insight_type: str = 'comprehensive') -> Dict[str, Any]:
    """Get comprehensive pattern insights."""
    engine = PatternRecognitionEngine(org_id)
    return engine.get_pattern_insights(insight_type)