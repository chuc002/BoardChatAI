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
        """Analyze a group of similar decisions to identify patterns."""
        if len(decisions) < 2:
            return None
        
        try:
            # Calculate success rate (decisions marked as would_repeat)
            successful_decisions = [d for d in decisions if d.get('would_repeat') == True]
            success_rate = len(successful_decisions) / len(decisions)
            
            # Calculate approval rate (decisions that were approved/implemented)
            approved_decisions = [d for d in decisions if d.get('vote_details', {}).get('approved') == True]
            approval_rate = len(approved_decisions) / len(decisions)
            
            # Calculate average timeline
            timelines = []
            for decision in decisions:
                if decision.get('proposed_date') and decision.get('vote_date'):
                    proposed = datetime.fromisoformat(decision['proposed_date'])
                    voted = datetime.fromisoformat(decision['vote_date'])
                    timeline = (voted - proposed).days
                    timelines.append(timeline)
            
            avg_timeline = int(statistics.mean(timelines)) if timelines else 30
            
            # Calculate average cost
            costs = [d.get('budget_actual', d.get('budget_projected', 0)) or 0 for d in decisions]
            avg_cost = statistics.mean(costs) if costs else None
            
            # Extract common risk and success factors
            risk_factors = self._extract_common_factors(decisions, 'risk_assessment')
            success_factors = self._extract_common_factors(decisions, 'success_metrics')
            
            # Create pattern
            pattern = DecisionPattern(
                pattern_name=f"Pattern: {pattern_key.replace('_', ' ').title()}",
                pattern_type=pattern_key.split('_')[0],
                trigger_conditions={'pattern_key': pattern_key},
                success_rate=success_rate,
                approval_rate=approval_rate,
                typical_timeline_days=avg_timeline,
                average_cost=avg_cost,
                risk_factors=risk_factors,
                success_factors=success_factors,
                examples=[d.get('id') for d in decisions],
                confidence_score=min(1.0, len(decisions) / 10)  # Higher confidence with more examples
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