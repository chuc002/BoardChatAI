"""
Pattern Recognition System for Board Governance Analysis

This module identifies and tracks governance patterns to provide predictive insights
for board decisions, financial planning, and risk management.
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

from lib.supa import supa

logger = logging.getLogger(__name__)

@dataclass
class DecisionPattern:
    """Represents a identified decision pattern."""
    pattern_id: str
    pattern_type: str
    decision_category: str
    approval_rate: float
    avg_processing_days: float
    success_factors: List[str]
    failure_factors: List[str]
    typical_cost_range: Tuple[float, float]
    historical_instances: List[str]
    confidence_score: float

@dataclass
class PredictionResult:
    """Represents a prediction for a new proposal."""
    success_probability: float
    estimated_processing_time: float
    predicted_cost_overrun: float
    similar_decisions_count: int
    risk_factors: List[str]
    precedent_matches: List[Dict]
    recommendation: str
    confidence_level: str

class DecisionPatternAnalyzer:
    """Analyzes recurring decision patterns and success factors."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.patterns_cache = {}
        self.last_analysis = None
    
    def analyze_decision_patterns(self) -> Dict[str, DecisionPattern]:
        """Analyze all decision patterns from historical data."""
        logger.info("Starting decision pattern analysis")
        
        # Get all decisions with outcomes
        decisions = self._get_all_decisions()
        
        if not decisions:
            return {}
        
        # Group decisions by type
        pattern_groups = defaultdict(list)
        for decision in decisions:
            decision_type = decision.get('decision_type', 'general')
            pattern_groups[decision_type].append(decision)
        
        patterns = {}
        for decision_type, group_decisions in pattern_groups.items():
            pattern = self._analyze_pattern_group(decision_type, group_decisions)
            if pattern:
                patterns[decision_type] = pattern
        
        self.patterns_cache = patterns
        self.last_analysis = datetime.now()
        
        # Store patterns in database
        self._store_patterns(patterns)
        
        return patterns
    
    def _analyze_pattern_group(self, decision_type: str, decisions: List[Dict]) -> Optional[DecisionPattern]:
        """Analyze a group of similar decisions."""
        if len(decisions) < 3:  # Need minimum sample size
            return None
        
        # Calculate approval rate
        approved_count = sum(1 for d in decisions if d.get('outcome') in ['approved', 'passed'])
        approval_rate = approved_count / len(decisions) * 100
        
        # Calculate average processing time
        processing_times = []
        for decision in decisions:
            created_date = self._parse_date(decision.get('created_at'))
            decision_date = self._parse_date(decision.get('date'))
            if created_date and decision_date:
                processing_days = (decision_date - created_date).days
                if processing_days >= 0:  # Sanity check
                    processing_times.append(processing_days)
        
        avg_processing_days = statistics.mean(processing_times) if processing_times else 0
        
        # Identify success and failure factors
        success_factors = self._identify_success_factors(
            [d for d in decisions if d.get('outcome') in ['approved', 'passed']]
        )
        failure_factors = self._identify_failure_factors(
            [d for d in decisions if d.get('outcome') in ['rejected', 'failed']]
        )
        
        # Calculate typical cost range
        amounts = [d.get('amount_involved', 0) for d in decisions if d.get('amount_involved')]
        if amounts:
            cost_range = (min(amounts), max(amounts))
        else:
            cost_range = (0, 0)
        
        # Calculate confidence score
        confidence_score = min(1.0, len(decisions) / 10)  # Higher confidence with more data
        
        return DecisionPattern(
            pattern_id=f"{decision_type}_{len(decisions)}_{hash(str(decisions))%10000}",
            pattern_type=decision_type,
            decision_category=decision_type,
            approval_rate=approval_rate,
            avg_processing_days=avg_processing_days,
            success_factors=success_factors,
            failure_factors=failure_factors,
            typical_cost_range=cost_range,
            historical_instances=[d.get('id') for d in decisions],
            confidence_score=confidence_score
        )
    
    def _identify_success_factors(self, successful_decisions: List[Dict]) -> List[str]:
        """Identify common factors in successful decisions."""
        factors = []
        
        if not successful_decisions:
            return factors
        
        # Check for common patterns
        vote_margins = []
        for decision in successful_decisions:
            votes_for = decision.get('vote_count_for', 0)
            votes_against = decision.get('vote_count_against', 0)
            if votes_for > 0:
                margin = votes_for / (votes_for + votes_against) if (votes_for + votes_against) > 0 else 0
                vote_margins.append(margin)
        
        if vote_margins:
            avg_margin = statistics.mean(vote_margins)
            if avg_margin > 0.7:
                factors.append("Strong consensus (>70% support)")
            elif avg_margin > 0.6:
                factors.append("Good majority support")
        
        # Check financial patterns
        amounts = [d.get('amount_involved') for d in successful_decisions if d.get('amount_involved')]
        if amounts:
            median_amount = statistics.median(amounts)
            if median_amount < 10000:
                factors.append("Lower financial impact (<$10K)")
            elif median_amount < 50000:
                factors.append("Moderate financial impact ($10K-$50K)")
        
        # Check timing patterns
        tags_frequency = defaultdict(int)
        for decision in successful_decisions:
            tags = decision.get('tags', [])
            for tag in tags:
                tags_frequency[tag] += 1
        
        common_tags = [tag for tag, freq in tags_frequency.items() if freq > len(successful_decisions) / 2]
        if common_tags:
            factors.extend([f"Common in {tag} decisions" for tag in common_tags[:3]])
        
        return factors
    
    def _identify_failure_factors(self, failed_decisions: List[Dict]) -> List[str]:
        """Identify common factors in failed decisions."""
        factors = []
        
        if not failed_decisions:
            return factors
        
        # Similar analysis as success factors but for failures
        amounts = [d.get('amount_involved') for d in failed_decisions if d.get('amount_involved')]
        if amounts:
            median_amount = statistics.median(amounts)
            if median_amount > 100000:
                factors.append("High financial impact (>$100K)")
        
        # Check for rushed decisions (quick processing time)
        quick_decisions = sum(1 for d in failed_decisions if d.get('priority_level') == 'high')
        if quick_decisions > len(failed_decisions) / 2:
            factors.append("Often marked as high priority/rushed")
        
        return factors
    
    def _get_all_decisions(self) -> List[Dict]:
        """Get all decisions from the database."""
        try:
            result = supa.table('decision_registry').select('*').eq('org_id', self.org_id).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get decisions: {e}")
            return []
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Handle different date formats
            formats = ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f']
            for fmt in formats:
                try:
                    return datetime.strptime(date_str[:len(fmt)], fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None
    
    def _store_patterns(self, patterns: Dict[str, DecisionPattern]):
        """Store identified patterns in the database."""
        try:
            for pattern_type, pattern in patterns.items():
                pattern_data = {
                    'org_id': self.org_id,
                    'pattern_type': pattern.decision_category,
                    'pattern_name': f"{pattern.decision_category.title()} Decision Pattern",
                    'frequency_count': len(pattern.historical_instances),
                    'success_rate': pattern.approval_rate,
                    'average_duration_days': int(pattern.avg_processing_days),
                    'typical_amount': (pattern.typical_cost_range[0] + pattern.typical_cost_range[1]) / 2,
                    'amount_range_min': pattern.typical_cost_range[0],
                    'amount_range_max': pattern.typical_cost_range[1],
                    'success_factors': pattern.success_factors,
                    'common_failures': pattern.failure_factors,
                    'decision_instances': pattern.historical_instances,
                    'confidence_score': pattern.confidence_score,
                    'last_occurrence': datetime.now().date()
                }
                
                # Upsert pattern
                existing = supa.table('historical_patterns').select('id').eq('org_id', self.org_id).eq('pattern_type', pattern_type).execute()
                
                if existing.data:
                    supa.table('historical_patterns').update(pattern_data).eq('id', existing.data[0]['id']).execute()
                else:
                    supa.table('historical_patterns').insert(pattern_data).execute()
                    
        except Exception as e:
            logger.error(f"Failed to store patterns: {e}")

class FinancialPatternTracker:
    """Tracks financial patterns and cost predictions."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
    
    def analyze_fee_increase_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in fee increases and member reactions."""
        try:
            # Get decisions involving fee increases
            fee_decisions = self._get_fee_decisions()
            
            increases = []
            member_reactions = {}
            
            for decision in fee_decisions:
                if self._is_fee_increase(decision):
                    amount = decision.get('amount_involved', 0)
                    outcome = decision.get('outcome', '')
                    
                    increases.append({
                        'amount': amount,
                        'year': self._extract_year(decision.get('date')),
                        'outcome': outcome,
                        'vote_margin': self._calculate_vote_margin(decision)
                    })
            
            # Calculate patterns
            if increases:
                successful_increases = [i for i in increases if i['outcome'] in ['approved', 'passed']]
                avg_increase = statistics.mean([i['amount'] for i in successful_increases]) if successful_increases else 0
                success_rate = len(successful_increases) / len(increases) * 100
                
                return {
                    'total_fee_increase_attempts': len(increases),
                    'successful_increases': len(successful_increases),
                    'success_rate': success_rate,
                    'average_successful_increase': avg_increase,
                    'pattern_confidence': min(1.0, len(increases) / 5)
                }
            
            return {'message': 'Insufficient fee increase data'}
            
        except Exception as e:
            logger.error(f"Fee pattern analysis failed: {e}")
            return {'error': str(e)}
    
    def predict_cost_overrun(self, decision_type: str, initial_amount: float) -> Dict[str, Any]:
        """Predict cost overruns based on historical patterns."""
        try:
            # Get historical decisions of similar type with final costs
            historical_decisions = self._get_decisions_with_costs(decision_type)
            
            if not historical_decisions:
                return {
                    'predicted_overrun_percentage': 15.0,  # Default conservative estimate
                    'confidence': 'low',
                    'basis': 'industry_average'
                }
            
            overruns = []
            for decision in historical_decisions:
                initial = decision.get('amount_involved', 0)
                final = decision.get('final_cost', initial)  # Assuming final_cost field exists
                
                if initial > 0:
                    overrun_pct = ((final - initial) / initial) * 100
                    overruns.append(overrun_pct)
            
            if overruns:
                avg_overrun = statistics.mean(overruns)
                std_overrun = statistics.stdev(overruns) if len(overruns) > 1 else 0
                
                predicted_final_cost = initial_amount * (1 + avg_overrun / 100)
                
                return {
                    'predicted_overrun_percentage': avg_overrun,
                    'predicted_final_cost': predicted_final_cost,
                    'confidence': 'high' if len(overruns) >= 5 else 'medium',
                    'historical_range': (min(overruns), max(overruns)),
                    'sample_size': len(overruns)
                }
            
            return {'message': 'No historical cost data available'}
            
        except Exception as e:
            logger.error(f"Cost overrun prediction failed: {e}")
            return {'error': str(e)}
    
    def _get_fee_decisions(self) -> List[Dict]:
        """Get decisions related to fees."""
        try:
            result = supa.table('decision_registry').select('*').eq('org_id', self.org_id).contains('tags', ['financial', 'fee']).execute()
            return result.data if result.data else []
        except Exception:
            return []
    
    def _is_fee_increase(self, decision: Dict) -> bool:
        """Check if decision represents a fee increase."""
        description = decision.get('description', '').lower()
        title = decision.get('title', '').lower()
        
        increase_indicators = ['increase', 'raise', 'higher', 'additional']
        fee_indicators = ['fee', 'dues', 'cost', 'charge']
        
        return (any(ind in description or ind in title for ind in increase_indicators) and
                any(ind in description or ind in title for ind in fee_indicators))
    
    def _extract_year(self, date_str: str) -> int:
        """Extract year from date string."""
        try:
            if date_str:
                return int(date_str[:4])
        except (ValueError, TypeError):
            pass
        return datetime.now().year
    
    def _calculate_vote_margin(self, decision: Dict) -> float:
        """Calculate voting margin."""
        votes_for = decision.get('vote_count_for', 0)
        votes_against = decision.get('vote_count_against', 0)
        total_votes = votes_for + votes_against
        
        if total_votes > 0:
            return votes_for / total_votes
        return 0.0
    
    def _get_decisions_with_costs(self, decision_type: str) -> List[Dict]:
        """Get decisions with cost information."""
        try:
            result = supa.table('decision_registry').select('*').eq('org_id', self.org_id).eq('decision_type', decision_type).not_.is_('amount_involved', 'null').execute()
            return result.data if result.data else []
        except Exception:
            return []

class PrecedentMatcher:
    """Finds and matches similar historical decisions."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
    
    def find_similar_decisions(self, proposal_text: str, decision_type: str = None, amount: float = None) -> List[Dict]:
        """Find all similar decisions in history."""
        try:
            # Get all relevant decisions
            query = supa.table('decision_registry').select('*').eq('org_id', self.org_id)
            
            if decision_type:
                query = query.eq('decision_type', decision_type)
            
            result = query.execute()
            all_decisions = result.data if result.data else []
            
            # Score similarity
            similar_decisions = []
            for decision in all_decisions:
                similarity_score = self._calculate_similarity(
                    proposal_text, decision, amount
                )
                
                if similarity_score > 0.3:  # Threshold for similarity
                    decision['similarity_score'] = similarity_score
                    similar_decisions.append(decision)
            
            # Sort by similarity
            similar_decisions.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_decisions[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.error(f"Precedent matching failed: {e}")
            return []
    
    def identify_precedent_deviations(self, proposal: Dict, similar_decisions: List[Dict]) -> List[str]:
        """Identify how this proposal deviates from precedent."""
        deviations = []
        
        if not similar_decisions:
            deviations.append("No historical precedent found")
            return deviations
        
        # Check amount deviation
        proposal_amount = proposal.get('amount_involved', 0)
        if proposal_amount > 0:
            historical_amounts = [d.get('amount_involved', 0) for d in similar_decisions if d.get('amount_involved')]
            if historical_amounts:
                avg_historical = statistics.mean(historical_amounts)
                if proposal_amount > avg_historical * 1.5:
                    deviations.append(f"Proposed amount (${proposal_amount:,.2f}) is {proposal_amount/avg_historical:.1f}x higher than historical average")
                elif proposal_amount < avg_historical * 0.5:
                    deviations.append(f"Proposed amount (${proposal_amount:,.2f}) is significantly lower than historical precedent")
        
        # Check processing urgency
        proposal_priority = proposal.get('priority_level', 'normal')
        historical_priorities = [d.get('priority_level', 'normal') for d in similar_decisions]
        urgent_historical = sum(1 for p in historical_priorities if p == 'high')
        
        if proposal_priority == 'high' and urgent_historical < len(similar_decisions) * 0.3:
            deviations.append("Marked as high priority while similar decisions were typically routine")
        
        return deviations
    
    def predict_outcome_based_on_precedent(self, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Predict outcome based on similar historical decisions."""
        if not similar_decisions:
            return {
                'predicted_outcome': 'unknown',
                'confidence': 0.0,
                'basis': 'no_precedent'
            }
        
        # Calculate outcome probabilities
        outcomes = [d.get('outcome', '') for d in similar_decisions]
        approved_count = sum(1 for o in outcomes if o in ['approved', 'passed'])
        
        success_probability = approved_count / len(outcomes)
        
        # Weigh by similarity scores
        weighted_success = 0
        total_weight = 0
        for decision in similar_decisions:
            weight = decision.get('similarity_score', 0.5)
            if decision.get('outcome') in ['approved', 'passed']:
                weighted_success += weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_probability = weighted_success / total_weight
        else:
            weighted_probability = success_probability
        
        return {
            'predicted_outcome': 'approved' if weighted_probability > 0.5 else 'rejected',
            'success_probability': weighted_probability,
            'confidence': min(1.0, len(similar_decisions) / 5),
            'basis': f"{len(similar_decisions)} similar decisions",
            'historical_success_rate': success_probability
        }
    
    def _calculate_similarity(self, proposal_text: str, historical_decision: Dict, proposal_amount: float = None) -> float:
        """Calculate similarity score between proposal and historical decision."""
        score = 0.0
        
        # Text similarity (basic keyword matching)
        proposal_words = set(proposal_text.lower().split())
        decision_words = set((historical_decision.get('description', '') + ' ' + 
                            historical_decision.get('title', '')).lower().split())
        
        if proposal_words and decision_words:
            common_words = proposal_words.intersection(decision_words)
            text_similarity = len(common_words) / len(proposal_words.union(decision_words))
            score += text_similarity * 0.4
        
        # Amount similarity
        if proposal_amount and historical_decision.get('amount_involved'):
            hist_amount = historical_decision.get('amount_involved')
            if hist_amount > 0:
                ratio = min(proposal_amount, hist_amount) / max(proposal_amount, hist_amount)
                score += ratio * 0.3
        
        # Type similarity
        if historical_decision.get('decision_type'):
            # This would be enhanced with better type matching
            score += 0.3  # Base score for having a type
        
        return min(1.0, score)

class RiskPatternIdentifier:
    """Identifies risk patterns and predicts potential problems."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
    
    def identify_high_risk_patterns(self) -> Dict[str, Any]:
        """Identify patterns that historically led to problems."""
        try:
            # Get decisions with known negative outcomes or subsequent problems
            problematic_decisions = self._get_problematic_decisions()
            
            risk_patterns = []
            
            if problematic_decisions:
                # Analyze common characteristics
                common_factors = self._extract_risk_factors(problematic_decisions)
                
                risk_patterns = [
                    {
                        'risk_factor': factor,
                        'frequency': frequency,
                        'risk_level': self._assess_risk_level(frequency, len(problematic_decisions))
                    }
                    for factor, frequency in common_factors.items()
                ]
            
            return {
                'risk_patterns': risk_patterns,
                'total_problematic_decisions': len(problematic_decisions),
                'analysis_confidence': min(1.0, len(problematic_decisions) / 10)
            }
            
        except Exception as e:
            logger.error(f"Risk pattern identification failed: {e}")
            return {'error': str(e)}
    
    def assess_proposal_risk(self, proposal: Dict) -> Dict[str, Any]:
        """Assess risk level of a new proposal."""
        risk_score = 0.0
        risk_factors = []
        
        # Check amount-based risk
        amount = proposal.get('amount_involved', 0)
        if amount > 100000:
            risk_score += 0.3
            risk_factors.append("High financial impact (>$100K)")
        
        # Check urgency-based risk
        if proposal.get('priority_level') == 'high':
            risk_score += 0.2
            risk_factors.append("High priority/rushed decision")
        
        # Check precedent deviation risk
        similar_decisions = PrecedentMatcher(self.org_id).find_similar_decisions(
            proposal.get('description', ''),
            proposal.get('decision_type'),
            amount
        )
        
        if not similar_decisions:
            risk_score += 0.3
            risk_factors.append("No historical precedent")
        else:
            deviations = PrecedentMatcher(self.org_id).identify_precedent_deviations(proposal, similar_decisions)
            if deviations:
                risk_score += 0.2 * len(deviations)
                risk_factors.extend(deviations[:2])  # Add top 2 deviations
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        risk_level = 'low'
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'medium'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': self._generate_risk_recommendations(risk_factors)
        }
    
    def _get_problematic_decisions(self) -> List[Dict]:
        """Get decisions that led to problems."""
        try:
            # This would ideally identify decisions with subsequent issues
            # For now, use rejected decisions and those with low vote margins
            result = supa.table('decision_registry').select('*').eq('org_id', self.org_id).execute()
            all_decisions = result.data if result.data else []
            
            problematic = []
            for decision in all_decisions:
                if (decision.get('outcome') in ['rejected', 'failed'] or
                    self._had_implementation_problems(decision)):
                    problematic.append(decision)
            
            return problematic
            
        except Exception:
            return []
    
    def _had_implementation_problems(self, decision: Dict) -> bool:
        """Check if decision had implementation problems."""
        # This would check for subsequent issues, complaints, or reversals
        # For now, use heuristics
        votes_for = decision.get('vote_count_for', 0)
        votes_against = decision.get('vote_count_against', 0)
        
        if votes_for + votes_against > 0:
            margin = votes_for / (votes_for + votes_against)
            return margin < 0.6  # Narrow margin might indicate problems
        
        return False
    
    def _extract_risk_factors(self, problematic_decisions: List[Dict]) -> Dict[str, int]:
        """Extract common factors from problematic decisions."""
        factors = defaultdict(int)
        
        for decision in problematic_decisions:
            # High amount
            if decision.get('amount_involved', 0) > 50000:
                factors['high_financial_impact'] += 1
            
            # High priority
            if decision.get('priority_level') == 'high':
                factors['rushed_decision'] += 1
            
            # Specific tags
            tags = decision.get('tags', [])
            if 'emergency' in tags:
                factors['emergency_decision'] += 1
        
        return dict(factors)
    
    def _assess_risk_level(self, frequency: int, total_decisions: int) -> str:
        """Assess risk level based on frequency."""
        rate = frequency / total_decisions if total_decisions > 0 else 0
        
        if rate > 0.7:
            return 'high'
        elif rate > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_risk_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk factors."""
        recommendations = []
        
        if any('financial' in factor.lower() for factor in risk_factors):
            recommendations.append("Consider detailed financial impact analysis")
            recommendations.append("Require multiple approval stages for high amounts")
        
        if any('precedent' in factor.lower() for factor in risk_factors):
            recommendations.append("Document reasons for deviation from precedent")
            recommendations.append("Consider extended discussion period")
        
        if any('rushed' in factor.lower() for factor in risk_factors):
            recommendations.append("Allow additional time for member review")
            recommendations.append("Consider postponing to next regular meeting")
        
        return recommendations

class GovernancePatternEngine:
    """Main engine that coordinates all pattern recognition systems."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.decision_analyzer = DecisionPatternAnalyzer(org_id)
        self.financial_tracker = FinancialPatternTracker(org_id)
        self.precedent_matcher = PrecedentMatcher(org_id)
        self.risk_identifier = RiskPatternIdentifier(org_id)
    
    def analyze_proposal(self, proposal: Dict) -> PredictionResult:
        """Comprehensive analysis of a new proposal."""
        logger.info(f"Analyzing proposal: {proposal.get('title', 'Untitled')}")
        
        # Find similar decisions
        similar_decisions = self.precedent_matcher.find_similar_decisions(
            proposal.get('description', ''),
            proposal.get('decision_type'),
            proposal.get('amount_involved')
        )
        
        # Predict outcome based on precedent
        outcome_prediction = self.precedent_matcher.predict_outcome_based_on_precedent(similar_decisions)
        
        # Assess risks
        risk_assessment = self.risk_identifier.assess_proposal_risk(proposal)
        
        # Predict cost overrun
        cost_prediction = self.financial_tracker.predict_cost_overrun(
            proposal.get('decision_type', 'general'),
            proposal.get('amount_involved', 0)
        )
        
        # Get decision pattern
        decision_patterns = self.decision_analyzer.patterns_cache
        pattern = decision_patterns.get(proposal.get('decision_type', 'general'))
        
        # Estimate processing time
        if pattern:
            estimated_processing_time = pattern.avg_processing_days
        else:
            estimated_processing_time = 21.0  # Default 3 weeks
        
        # Generate comprehensive prediction
        prediction = PredictionResult(
            success_probability=outcome_prediction.get('success_probability', 0.5),
            estimated_processing_time=estimated_processing_time,
            predicted_cost_overrun=cost_prediction.get('predicted_overrun_percentage', 15.0),
            similar_decisions_count=len(similar_decisions),
            risk_factors=risk_assessment.get('risk_factors', []),
            precedent_matches=[
                {
                    'decision_id': d.get('decision_id'),
                    'title': d.get('title'),
                    'outcome': d.get('outcome'),
                    'similarity_score': d.get('similarity_score', 0)
                }
                for d in similar_decisions[:5]
            ],
            recommendation=self._generate_recommendation(outcome_prediction, risk_assessment, pattern),
            confidence_level=self._assess_confidence_level(similar_decisions, pattern)
        )
        
        return prediction
    
    def _generate_recommendation(self, outcome_prediction: Dict, risk_assessment: Dict, pattern: Optional[DecisionPattern]) -> str:
        """Generate a recommendation based on analysis."""
        success_prob = outcome_prediction.get('success_probability', 0.5)
        risk_level = risk_assessment.get('risk_level', 'medium')
        
        if success_prob > 0.7 and risk_level == 'low':
            return "RECOMMENDED: High probability of success with low risk"
        elif success_prob > 0.5 and risk_level in ['low', 'medium']:
            return "PROCEED WITH CAUTION: Moderate success probability"
        elif risk_level == 'high':
            return "HIGH RISK: Consider additional safeguards or postponement"
        else:
            return "REQUIRES CAREFUL CONSIDERATION: Mixed indicators"
    
    def _assess_confidence_level(self, similar_decisions: List[Dict], pattern: Optional[DecisionPattern]) -> str:
        """Assess confidence level of the prediction."""
        factors = 0
        
        if len(similar_decisions) >= 5:
            factors += 1
        if pattern and pattern.confidence_score > 0.7:
            factors += 1
        if similar_decisions and similar_decisions[0].get('similarity_score', 0) > 0.7:
            factors += 1
        
        if factors >= 2:
            return 'high'
        elif factors == 1:
            return 'medium'
        else:
            return 'low'

def analyze_governance_patterns(org_id: str) -> Dict[str, Any]:
    """Main entry point for governance pattern analysis."""
    engine = GovernancePatternEngine(org_id)
    
    # Run full analysis
    decision_patterns = engine.decision_analyzer.analyze_decision_patterns()
    financial_patterns = engine.financial_tracker.analyze_fee_increase_patterns()
    risk_patterns = engine.risk_identifier.identify_high_risk_patterns()
    
    return {
        'decision_patterns': {
            pattern_type: {
                'approval_rate': pattern.approval_rate,
                'avg_processing_days': pattern.avg_processing_days,
                'success_factors': pattern.success_factors,
                'failure_factors': pattern.failure_factors,
                'confidence_score': pattern.confidence_score
            }
            for pattern_type, pattern in decision_patterns.items()
        },
        'financial_patterns': financial_patterns,
        'risk_patterns': risk_patterns,
        'analysis_timestamp': datetime.now().isoformat()
    }

def predict_proposal_outcome(org_id: str, proposal: Dict) -> Dict[str, Any]:
    """Predict outcome for a specific proposal."""
    engine = GovernancePatternEngine(org_id)
    
    # Ensure patterns are analyzed
    engine.decision_analyzer.analyze_decision_patterns()
    
    prediction = engine.analyze_proposal(proposal)
    
    return {
        'success_probability': prediction.success_probability,
        'estimated_processing_time_days': prediction.estimated_processing_time,
        'predicted_cost_overrun_percent': prediction.predicted_cost_overrun,
        'similar_decisions_count': prediction.similar_decisions_count,
        'risk_factors': prediction.risk_factors,
        'precedent_matches': prediction.precedent_matches,
        'recommendation': prediction.recommendation,
        'confidence_level': prediction.confidence_level,
        'analysis_summary': f"Based on {prediction.similar_decisions_count} similar decisions over historical data, "
                          f"this proposal has a {prediction.success_probability:.0%} chance of success, "
                          f"typically takes {prediction.estimated_processing_time:.1f} days to process, "
                          f"and usually results in a {prediction.predicted_cost_overrun:.1f}% cost variance."
    }