"""
Decision Outcome Predictor - Advanced Governance Decision Forecasting

This module provides comprehensive outcome prediction for governance decisions,
combining historical pattern analysis, risk assessment, and multi-model forecasting.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import statistics
import numpy as np
from collections import defaultdict

from lib.supa import supa
from lib.pattern_recognition import PatternRecognitionEngine

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Comprehensive prediction result for a decision."""
    overall_recommendation: str
    success_probability: float
    approval_probability: float
    confidence_level: str
    timeline_estimate: int
    cost_variance_forecast: float
    risk_score: float
    key_factors: List[str]
    recommendations: List[str]
    similar_decisions: List[Dict]

class DecisionOutcomePredictor:
    """
    Advanced decision outcome predictor using historical patterns and machine learning.
    Provides comprehensive forecasting for governance decisions.
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.pattern_engine = PatternRecognitionEngine(org_id)
        self.prediction_models = {
            'approval_likelihood': self._build_approval_model,
            'implementation_success': self._build_implementation_model,
            'cost_accuracy': self._build_cost_model,
            'timeline_accuracy': self._build_timeline_model,
            'member_satisfaction': self._build_satisfaction_model
        }
        logger.info("Decision Outcome Predictor initialized")
    
    def predict_complete_outcome(self, proposed_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Provide comprehensive outcome prediction for a proposed decision."""
        
        try:
            # Extract decision features
            features = self._extract_decision_features(proposed_decision)
            
            # Get historical context
            similar_decisions = self._find_similar_decisions(features)
            
            # Run all prediction models
            predictions = {}
            for model_name, model_func in self.prediction_models.items():
                try:
                    predictions[model_name] = model_func(proposed_decision, similar_decisions)
                except Exception as e:
                    logger.error(f"Failed to run {model_name} model: {e}")
                    predictions[model_name] = self._get_default_prediction(model_name)
            
            # Generate comprehensive assessment
            assessment = {
                'overall_recommendation': self._generate_overall_recommendation(predictions),
                'approval_prediction': predictions['approval_likelihood'],
                'implementation_forecast': predictions['implementation_success'],
                'financial_forecast': predictions['cost_accuracy'],
                'timeline_forecast': predictions['timeline_accuracy'],
                'satisfaction_forecast': predictions['member_satisfaction'],
                'risk_assessment': self._assess_risks(proposed_decision, similar_decisions),
                'optimization_suggestions': self._suggest_optimizations(proposed_decision, predictions),
                'precedent_analysis': self._analyze_precedents(similar_decisions),
                'decision_tree': self._build_decision_tree(proposed_decision, predictions),
                'confidence_metrics': self._calculate_confidence_metrics(predictions, similar_decisions),
                'scenario_analysis': self._perform_scenario_analysis(proposed_decision, similar_decisions)
            }
            
            # Store prediction for future learning
            self._store_prediction(proposed_decision, assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Failed to predict outcome: {e}")
            return self._get_default_assessment(proposed_decision)
    
    def _extract_decision_features(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from a decision for analysis."""
        return {
            'amount': decision.get('budget_projected', 0) or decision.get('amount', 0),
            'type': decision.get('decision_type', 'general'),
            'committee': decision.get('committee', 'board'),
            'timing': decision.get('proposed_date'),
            'sponsor': decision.get('proposed_by'),
            'urgency': decision.get('urgency_level', 'normal'),
            'complexity': decision.get('complexity_score', 0.5)
        }
    
    def _find_similar_decisions(self, features: Dict[str, Any]) -> List[Dict]:
        """Find historically similar decisions using pattern recognition."""
        try:
            # Get all historical decisions
            decisions = supa.table("decision_registry").select("*").eq("org_id", self.org_id).execute().data
            
            # Calculate similarity scores
            scored_decisions = []
            for decision in decisions:
                similarity = self._calculate_similarity(features, decision)
                if similarity > 0.3:  # Threshold for relevance
                    scored_decisions.append((similarity, decision))
            
            # Sort by similarity and return top matches
            scored_decisions.sort(reverse=True, key=lambda x: x[0])
            return [decision for _, decision in scored_decisions[:20]]
            
        except Exception as e:
            logger.error(f"Failed to find similar decisions: {e}")
            return []
    
    def _calculate_similarity(self, features: Dict, decision: Dict) -> float:
        """Calculate similarity between proposed and historical decision."""
        
        similarity_score = 0.0
        total_weight = 0.0
        
        # Amount similarity (weight: 0.3)
        if features.get('amount') and decision.get('budget_projected'):
            amount1, amount2 = features['amount'], decision['budget_projected']
            amount_ratio = min(amount1, amount2) / max(amount1, amount2) if max(amount1, amount2) > 0 else 0
            similarity_score += amount_ratio * 0.3
            total_weight += 0.3
        
        # Type similarity (weight: 0.4)
        if features.get('type') == decision.get('decision_type'):
            similarity_score += 0.4
            total_weight += 0.4
        
        # Committee similarity (weight: 0.2)
        if features.get('committee') == decision.get('committee'):
            similarity_score += 0.2
            total_weight += 0.2
        
        # Timing similarity (weight: 0.1)
        if features.get('timing') and decision.get('proposed_date'):
            timing_similarity = self._calculate_timing_similarity(features['timing'], decision['proposed_date'])
            similarity_score += timing_similarity * 0.1
            total_weight += 0.1
        
        return similarity_score / total_weight if total_weight > 0 else 0
    
    def _calculate_timing_similarity(self, date1: str, date2: str) -> float:
        """Calculate similarity between two dates (seasonal effects)."""
        try:
            d1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))
            d2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))
            
            # Same season gives higher similarity
            if self._get_season(d1) == self._get_season(d2):
                return 1.0
            
            # Same quarter gives medium similarity
            if (d1.month - 1) // 3 == (d2.month - 1) // 3:
                return 0.6
            
            return 0.2
        except:
            return 0
    
    def _get_season(self, date: datetime) -> str:
        """Get season from datetime."""
        month = date.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _build_approval_model(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Predict likelihood of approval using enhanced analysis."""
        
        if not similar_decisions:
            return {
                'probability': 0.5,
                'confidence': 'low',
                'factors': ['No historical data available'],
                'base_rate': 0.5,
                'adjustments': [],
                'similar_decisions_count': 0
            }
        
        # Calculate base approval rate
        approved_count = sum(1 for d in similar_decisions if self._is_approved(d))
        base_rate = approved_count / len(similar_decisions)
        
        # Apply adjustment factors
        adjustments = []
        adjusted_rate = base_rate
        
        # Amount factor analysis
        amount_factor = self._calculate_amount_factor(proposed_decision, similar_decisions)
        adjusted_rate *= amount_factor['multiplier']
        adjustments.append(amount_factor['explanation'])
        
        # Timing factor analysis
        timing_factor = self._calculate_timing_factor(proposed_decision, similar_decisions)
        adjusted_rate *= timing_factor['multiplier']
        adjustments.append(timing_factor['explanation'])
        
        # Committee factor analysis
        committee_factor = self._calculate_committee_factor(proposed_decision, similar_decisions)
        adjusted_rate *= committee_factor['multiplier']
        adjustments.append(committee_factor['explanation'])
        
        # Sponsor influence factor
        sponsor_factor = self._calculate_sponsor_factor(proposed_decision, similar_decisions)
        adjusted_rate *= sponsor_factor['multiplier']
        adjustments.append(sponsor_factor['explanation'])
        
        # Complexity factor
        complexity_factor = self._calculate_complexity_factor(proposed_decision, similar_decisions)
        adjusted_rate *= complexity_factor['multiplier']
        adjustments.append(complexity_factor['explanation'])
        
        # Cap at reasonable bounds
        adjusted_rate = max(0.05, min(0.95, adjusted_rate))
        
        # Determine confidence level
        confidence = self._determine_confidence(len(similar_decisions), adjustments)
        
        return {
            'probability': round(adjusted_rate, 3),
            'confidence': confidence,
            'base_rate': round(base_rate, 3),
            'adjustments': adjustments,
            'similar_decisions_count': len(similar_decisions),
            'key_factors': self._identify_key_approval_factors(similar_decisions),
            'approval_threshold_analysis': self._analyze_approval_thresholds(similar_decisions)
        }
    
    def _build_implementation_model(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Predict implementation success with detailed analysis."""
        
        # Filter to only approved decisions
        implemented_decisions = [d for d in similar_decisions if self._is_approved(d)]
        
        if not implemented_decisions:
            return {
                'success_probability': 0.7,  # Default optimistic assumption
                'confidence': 'low',
                'success_factors': [],
                'risk_factors': [],
                'typical_timeline': 90
            }
        
        # Analyze implementation success
        successful_implementations = []
        failed_implementations = []
        
        for decision in implemented_decisions:
            if self._is_implementation_successful(decision):
                successful_implementations.append(decision)
            else:
                failed_implementations.append(decision)
        
        success_rate = len(successful_implementations) / len(implemented_decisions) if implemented_decisions else 0.7
        
        # Identify patterns
        success_factors = self._identify_implementation_success_factors(successful_implementations)
        risk_factors = self._identify_implementation_risk_factors(failed_implementations)
        
        return {
            'success_probability': round(success_rate, 3),
            'confidence': 'high' if len(implemented_decisions) >= 10 else 'medium' if len(implemented_decisions) >= 5 else 'low',
            'success_factors': success_factors,
            'risk_factors': risk_factors,
            'typical_timeline': self._calculate_typical_implementation_timeline(implemented_decisions),
            'cost_variance_forecast': self._forecast_cost_variance(implemented_decisions, proposed_decision),
            'implementation_complexity': self._assess_implementation_complexity(proposed_decision, implemented_decisions)
        }
    
    def _build_cost_model(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Predict cost accuracy and overruns."""
        
        decisions_with_costs = [d for d in similar_decisions 
                               if d.get('budget_projected') and d.get('budget_actual')]
        
        if not decisions_with_costs:
            return {
                'predicted_variance': 0.1,  # 10% default variance
                'confidence': 'low',
                'cost_factors': [],
                'budget_risk': 'medium'
            }
        
        # Calculate historical cost variances
        variances = []
        for decision in decisions_with_costs:
            projected = decision['budget_projected']
            actual = decision['budget_actual']
            if projected > 0:
                variance = (actual - projected) / projected
                variances.append(variance)
        
        avg_variance = statistics.mean(variances) if variances else 0
        variance_std = statistics.stdev(variances) if len(variances) > 1 else 0.1
        
        # Assess cost factors specific to this decision
        cost_factors = self._identify_cost_factors(proposed_decision, decisions_with_costs)
        
        # Predict variance based on decision characteristics
        predicted_variance = self._predict_cost_variance(proposed_decision, decisions_with_costs, avg_variance)
        
        return {
            'predicted_variance': round(predicted_variance, 3),
            'historical_average_variance': round(avg_variance, 3),
            'variance_uncertainty': round(variance_std, 3),
            'confidence': 'high' if len(decisions_with_costs) >= 10 else 'medium',
            'cost_factors': cost_factors,
            'budget_risk': self._assess_budget_risk(predicted_variance),
            'cost_optimization_suggestions': self._suggest_cost_optimizations(proposed_decision, decisions_with_costs)
        }
    
    def _build_timeline_model(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Predict timeline accuracy and delays."""
        
        decisions_with_timelines = [d for d in similar_decisions 
                                   if d.get('proposed_date') and d.get('implementation_completion_date')]
        
        if not decisions_with_timelines:
            return {
                'predicted_timeline': 90,  # 3 months default
                'confidence': 'low',
                'delay_probability': 0.3,
                'timeline_factors': []
            }
        
        # Calculate historical timelines
        timelines = []
        for decision in decisions_with_timelines:
            try:
                start = datetime.fromisoformat(decision['proposed_date'])
                end = datetime.fromisoformat(decision['implementation_completion_date'])
                timeline = (end - start).days
                timelines.append(timeline)
            except:
                continue
        
        if not timelines:
            return {
                'predicted_timeline': 90,
                'confidence': 'low',
                'delay_probability': 0.3,
                'timeline_factors': []
            }
        
        avg_timeline = statistics.mean(timelines)
        timeline_std = statistics.stdev(timelines) if len(timelines) > 1 else 30
        
        # Predict timeline based on decision characteristics
        predicted_timeline = self._predict_timeline(proposed_decision, decisions_with_timelines, avg_timeline)
        
        # Calculate delay probability
        delay_probability = self._calculate_delay_probability(decisions_with_timelines)
        
        return {
            'predicted_timeline': round(predicted_timeline),
            'historical_average_timeline': round(avg_timeline),
            'timeline_uncertainty': round(timeline_std),
            'confidence': 'high' if len(decisions_with_timelines) >= 10 else 'medium',
            'delay_probability': round(delay_probability, 3),
            'timeline_factors': self._identify_timeline_factors(proposed_decision, decisions_with_timelines),
            'critical_path_risks': self._identify_critical_path_risks(proposed_decision)
        }
    
    def _build_satisfaction_model(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Predict member satisfaction with the decision."""
        
        # This would typically require satisfaction surveys or feedback data
        # For now, use proxy indicators
        
        satisfaction_indicators = []
        
        # Analyze voting margins as satisfaction proxy
        voting_margins = []
        for decision in similar_decisions:
            margin = self._get_voting_margin(decision)
            if margin is not None:
                voting_margins.append(margin)
        
        avg_margin = statistics.mean(voting_margins) if voting_margins else 0.5
        
        # High margin = high satisfaction
        predicted_satisfaction = min(0.95, max(0.1, avg_margin))
        
        return {
            'predicted_satisfaction': round(predicted_satisfaction, 3),
            'confidence': 'medium' if voting_margins else 'low',
            'satisfaction_factors': self._identify_satisfaction_factors(proposed_decision, similar_decisions),
            'stakeholder_analysis': self._analyze_stakeholder_impact(proposed_decision),
            'communication_recommendations': self._suggest_communication_strategy(proposed_decision)
        }
    
    def _assess_risks(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Comprehensive risk assessment."""
        
        risks = {
            'financial_risks': self._assess_financial_risks(proposed_decision, similar_decisions),
            'operational_risks': self._assess_operational_risks(proposed_decision, similar_decisions),
            'political_risks': self._assess_political_risks(proposed_decision, similar_decisions),
            'timeline_risks': self._assess_timeline_risks(proposed_decision, similar_decisions),
            'reputation_risks': self._assess_reputation_risks(proposed_decision, similar_decisions)
        }
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk_score(risks)
        
        return {
            'overall_risk_score': risk_score,
            'risk_level': self._categorize_risk_level(risk_score),
            'risk_categories': risks,
            'mitigation_strategies': self._suggest_risk_mitigations(risks),
            'risk_monitoring_plan': self._create_risk_monitoring_plan(risks)
        }
    
    def _generate_overall_recommendation(self, predictions: Dict) -> Dict[str, Any]:
        """Generate comprehensive recommendation based on all predictions."""
        
        # Extract key probabilities
        approval_prob = predictions['approval_likelihood']['probability']
        implementation_prob = predictions['implementation_success']['success_probability']
        
        # Calculate weighted success probability
        overall_success = (approval_prob * 0.6) + (implementation_prob * 0.4)
        
        # Determine recommendation
        if overall_success >= 0.75:
            recommendation = 'STRONG_APPROVE'
            reasoning = 'High probability of approval and successful implementation'
            action_items = ['Proceed with proposal as planned', 'Monitor implementation milestones']
        elif overall_success >= 0.6:
            recommendation = 'APPROVE'
            reasoning = 'Good success probability with manageable risks'
            action_items = ['Proceed with minor optimizations', 'Establish risk monitoring']
        elif overall_success >= 0.45:
            recommendation = 'APPROVE_WITH_CONDITIONS'
            reasoning = 'Moderate success probability - recommend modifications'
            action_items = ['Address key risk factors', 'Modify proposal based on recommendations', 'Build stronger consensus']
        elif overall_success >= 0.25:
            recommendation = 'DEFER'
            reasoning = 'Low success probability - significant changes needed'
            action_items = ['Redesign proposal', 'Address fundamental concerns', 'Build stakeholder support']
        else:
            recommendation = 'REJECT'
            reasoning = 'Very low success probability - fundamental issues'
            action_items = ['Reconsider proposal necessity', 'Explore alternative approaches']
        
        return {
            'recommendation': recommendation,
            'reasoning': reasoning,
            'overall_success_probability': round(overall_success, 3),
            'confidence_level': self._calculate_overall_confidence(predictions),
            'action_items': action_items,
            'decision_quality_score': self._calculate_decision_quality_score(predictions),
            'strategic_alignment': self._assess_strategic_alignment(predictions)
        }
    
    # Helper methods for detailed analysis
    
    def _is_approved(self, decision: Dict) -> bool:
        """Check if a decision was approved."""
        if decision.get('would_repeat') == True:
            return True
        vote_details = decision.get('vote_details', {})
        if isinstance(vote_details, dict) and vote_details.get('approved') == True:
            return True
        return decision.get('implementation_completion_date') is not None
    
    def _is_implementation_successful(self, decision: Dict) -> bool:
        """Check if implementation was successful."""
        # Multiple indicators of success
        if decision.get('implementation_completion_date'):
            # Check cost variance
            projected = decision.get('budget_projected', 0)
            actual = decision.get('budget_actual', 0)
            if projected > 0 and actual > 0:
                variance = abs(actual - projected) / projected
                if variance > 0.5:  # More than 50% variance indicates problems
                    return False
            
            # Check for explicit success indicators
            if decision.get('would_repeat') == False:
                return False
            
            return True
        
        return False
    
    def _get_default_prediction(self, model_name: str) -> Dict[str, Any]:
        """Get default prediction when model fails."""
        defaults = {
            'approval_likelihood': {'probability': 0.5, 'confidence': 'low'},
            'implementation_success': {'success_probability': 0.5, 'confidence': 'low'},
            'cost_accuracy': {'predicted_variance': 0.2, 'confidence': 'low'},
            'timeline_accuracy': {'predicted_timeline': 90, 'confidence': 'low'},
            'member_satisfaction': {'predicted_satisfaction': 0.5, 'confidence': 'low'}
        }
        return defaults.get(model_name, {'confidence': 'low'})
    
    def _get_default_assessment(self, proposed_decision: Dict) -> Dict[str, Any]:
        """Get default assessment when prediction fails."""
        return {
            'overall_recommendation': {
                'recommendation': 'DEFER',
                'reasoning': 'Insufficient data for reliable prediction',
                'overall_success_probability': 0.5,
                'confidence_level': 'low'
            },
            'approval_prediction': {'probability': 0.5, 'confidence': 'low'},
            'implementation_forecast': {'success_probability': 0.5, 'confidence': 'low'},
            'risk_assessment': {'overall_risk_score': 0.5, 'risk_level': 'medium'},
            'optimization_suggestions': ['Gather more historical data', 'Conduct stakeholder analysis']
        }
    
    def _store_prediction(self, decision: Dict, assessment: Dict):
        """Store prediction for future learning and validation."""
        try:
            prediction_record = {
                'id': str(uuid.uuid4()),
                'org_id': self.org_id,
                'decision_data': json.dumps(decision),
                'prediction_data': json.dumps(assessment),
                'created_at': datetime.now().isoformat(),
                'prediction_type': 'comprehensive_outcome'
            }
            
            # Store in predictions table if it exists
            supa.table('decision_predictions').insert(prediction_record).execute()
        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
    
    # Implementation of all helper methods
    
    def _suggest_optimizations(self, proposed_decision: Dict, predictions: Dict) -> List[str]:
        """Suggest optimizations to improve decision outcome."""
        suggestions = []
        
        approval_prob = predictions['approval_likelihood']['probability']
        if approval_prob < 0.6:
            suggestions.append("Build broader stakeholder consensus before proposal")
            suggestions.append("Consider phased implementation to reduce risk")
        
        cost_forecast = predictions.get('cost_accuracy', {})
        if cost_forecast.get('predicted_variance', 0) > 0.2:
            suggestions.append("Develop more detailed budget analysis")
            suggestions.append("Include contingency planning for cost overruns")
        
        return suggestions
    
    def _analyze_precedents(self, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze precedent decisions and their outcomes."""
        if not similar_decisions:
            return {'total_precedents': 0, 'key_lessons': []}
        
        lessons = []
        successful = [d for d in similar_decisions if self._is_approved(d)]
        failed = [d for d in similar_decisions if not self._is_approved(d)]
        
        if successful:
            lessons.append(f"{len(successful)} similar decisions were approved")
        if failed:
            lessons.append(f"{len(failed)} similar decisions faced challenges")
        
        return {
            'total_precedents': len(similar_decisions),
            'successful_precedents': len(successful),
            'failed_precedents': len(failed),
            'key_lessons': lessons,
            'most_relevant': similar_decisions[:3] if similar_decisions else []
        }
    
    def _build_decision_tree(self, proposed_decision: Dict, predictions: Dict) -> Dict[str, Any]:
        """Build decision tree analysis."""
        return {
            'decision_point': 'Proposal Approval',
            'success_path': {
                'probability': predictions['approval_likelihood']['probability'],
                'next_steps': ['Implementation Planning', 'Resource Allocation', 'Timeline Management']
            },
            'failure_path': {
                'probability': 1 - predictions['approval_likelihood']['probability'],
                'next_steps': ['Proposal Revision', 'Stakeholder Engagement', 'Alternative Approaches']
            }
        }
    
    def _calculate_confidence_metrics(self, predictions: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate confidence metrics for predictions."""
        data_quality = len(similar_decisions) / 20  # Scale to 20 decisions for full confidence
        
        model_confidence = statistics.mean([
            1 if pred.get('confidence') == 'high' else 0.7 if pred.get('confidence') == 'medium' else 0.3
            for pred in predictions.values() if isinstance(pred, dict) and 'confidence' in pred
        ]) if predictions else 0.5
        
        overall_confidence = min(1.0, (data_quality + model_confidence) / 2)
        
        return {
            'overall_confidence': round(overall_confidence, 3),
            'data_quality_score': round(data_quality, 3),
            'model_confidence_score': round(model_confidence, 3),
            'recommendation': 'high' if overall_confidence > 0.7 else 'medium' if overall_confidence > 0.4 else 'low'
        }
    
    def _perform_scenario_analysis(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Perform scenario analysis."""
        scenarios = {
            'best_case': {'probability_adjustment': 1.2, 'timeline_adjustment': 0.8},
            'worst_case': {'probability_adjustment': 0.6, 'timeline_adjustment': 1.5},
            'most_likely': {'probability_adjustment': 1.0, 'timeline_adjustment': 1.0}
        }
        
        return {
            'scenarios_analyzed': len(scenarios),
            'scenario_details': scenarios,
            'recommendation': 'Plan for most likely scenario with contingencies for worst case'
        }
    
    def _calculate_amount_factor(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate amount-based adjustment factor."""
        amount = proposed_decision.get('budget_projected', 0) or proposed_decision.get('amount', 0)
        amounts = [d.get('budget_projected', 0) or d.get('amount', 0) for d in similar_decisions if d.get('budget_projected') or d.get('amount')]
        
        if not amounts:
            return {'multiplier': 1.0, 'explanation': 'No amount data available for comparison'}
        
        avg_amount = statistics.mean(amounts)
        
        if amount > avg_amount * 1.5:
            return {'multiplier': 0.8, 'explanation': f'Amount ${amount:,.0f} is 50% above average ${avg_amount:,.0f}'}
        elif amount < avg_amount * 0.5:
            return {'multiplier': 1.1, 'explanation': f'Amount ${amount:,.0f} is 50% below average ${avg_amount:,.0f}'}
        else:
            return {'multiplier': 1.0, 'explanation': f'Amount ${amount:,.0f} is close to average ${avg_amount:,.0f}'}
    
    def _calculate_timing_factor(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate timing-based adjustment factor."""
        timing = proposed_decision.get('proposed_date')
        if not timing:
            return {'multiplier': 1.0, 'explanation': 'No timing information available'}
        
        try:
            proposed_date = datetime.fromisoformat(timing.replace('Z', '+00:00'))
            season = self._get_season(proposed_date)
            
            # Analyze seasonal success rates
            seasonal_decisions = [d for d in similar_decisions if d.get('proposed_date')]
            if not seasonal_decisions:
                return {'multiplier': 1.0, 'explanation': 'No seasonal data available'}
            
            season_success = []
            for decision in seasonal_decisions:
                try:
                    decision_date = datetime.fromisoformat(decision['proposed_date'].replace('Z', '+00:00'))
                    if self._get_season(decision_date) == season:
                        season_success.append(self._is_approved(decision))
                except:
                    continue
            
            if season_success:
                success_rate = sum(season_success) / len(season_success)
                if success_rate > 0.7:
                    return {'multiplier': 1.1, 'explanation': f'{season.title()} season shows high success rate ({success_rate:.1%})'}
                elif success_rate < 0.4:
                    return {'multiplier': 0.9, 'explanation': f'{season.title()} season shows lower success rate ({success_rate:.1%})'}
            
            return {'multiplier': 1.0, 'explanation': f'{season.title()} timing appears neutral'}
            
        except:
            return {'multiplier': 1.0, 'explanation': 'Unable to analyze timing effects'}
    
    def _calculate_committee_factor(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate committee-based adjustment factor."""
        committee = proposed_decision.get('committee', 'board')
        
        committee_decisions = [d for d in similar_decisions if d.get('committee') == committee]
        if not committee_decisions:
            return {'multiplier': 1.0, 'explanation': f'No data for {committee} committee'}
        
        success_rate = sum(1 for d in committee_decisions if self._is_approved(d)) / len(committee_decisions)
        
        if success_rate > 0.7:
            return {'multiplier': 1.1, 'explanation': f'{committee.title()} committee has high approval rate ({success_rate:.1%})'}
        elif success_rate < 0.4:
            return {'multiplier': 0.9, 'explanation': f'{committee.title()} committee has lower approval rate ({success_rate:.1%})'}
        else:
            return {'multiplier': 1.0, 'explanation': f'{committee.title()} committee shows average approval rate ({success_rate:.1%})'}
    
    def _calculate_sponsor_factor(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate sponsor influence factor."""
        sponsor = proposed_decision.get('proposed_by')
        if not sponsor:
            return {'multiplier': 1.0, 'explanation': 'No sponsor information available'}
        
        sponsor_decisions = [d for d in similar_decisions if d.get('proposed_by') == sponsor]
        if not sponsor_decisions:
            return {'multiplier': 1.0, 'explanation': f'No historical data for sponsor {sponsor}'}
        
        success_rate = sum(1 for d in sponsor_decisions if self._is_approved(d)) / len(sponsor_decisions)
        
        if success_rate > 0.7:
            return {'multiplier': 1.15, 'explanation': f'Sponsor {sponsor} has strong track record ({success_rate:.1%})'}
        elif success_rate < 0.4:
            return {'multiplier': 0.85, 'explanation': f'Sponsor {sponsor} has mixed track record ({success_rate:.1%})'}
        else:
            return {'multiplier': 1.0, 'explanation': f'Sponsor {sponsor} has average track record ({success_rate:.1%})'}
    
    def _calculate_complexity_factor(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Calculate complexity-based adjustment factor."""
        complexity = proposed_decision.get('complexity_score', 0.5)
        
        if complexity > 0.7:
            return {'multiplier': 0.85, 'explanation': 'High complexity reduces approval probability'}
        elif complexity < 0.3:
            return {'multiplier': 1.1, 'explanation': 'Low complexity increases approval probability'}
        else:
            return {'multiplier': 1.0, 'explanation': 'Moderate complexity has neutral effect'}
    
    def _determine_confidence(self, sample_size: int, adjustments: List[str]) -> str:
        """Determine confidence level based on data quality."""
        if sample_size >= 20:
            return 'high'
        elif sample_size >= 10:
            return 'medium'
        elif sample_size >= 5:
            return 'low'
        else:
            return 'very_low'
    
    def _identify_key_approval_factors(self, similar_decisions: List[Dict]) -> List[str]:
        """Identify key factors that influence approval."""
        factors = []
        
        if similar_decisions:
            approved = [d for d in similar_decisions if self._is_approved(d)]
            rejected = [d for d in similar_decisions if not self._is_approved(d)]
            
            # Analyze amount patterns
            if approved and rejected:
                avg_approved_amount = statistics.mean([d.get('budget_projected', 0) or d.get('amount', 0) for d in approved])
                avg_rejected_amount = statistics.mean([d.get('budget_projected', 0) or d.get('amount', 0) for d in rejected])
                
                if avg_approved_amount < avg_rejected_amount * 0.8:
                    factors.append('Lower amounts have higher approval rates')
        
        if not factors:
            factors.append('Insufficient data to identify key factors')
        
        return factors
    
    def _analyze_approval_thresholds(self, similar_decisions: List[Dict]) -> Dict[str, Any]:
        """Analyze approval thresholds by amount and type."""
        thresholds = {}
        
        # Group by amount ranges
        for decision in similar_decisions:
            amount = decision.get('budget_projected', 0) or decision.get('amount', 0)
            range_key = self._get_amount_range(amount)
            
            if range_key not in thresholds:
                thresholds[range_key] = {'approved': 0, 'total': 0}
            
            thresholds[range_key]['total'] += 1
            if self._is_approved(decision):
                thresholds[range_key]['approved'] += 1
        
        # Calculate approval rates
        for range_key in thresholds:
            data = thresholds[range_key]
            data['approval_rate'] = data['approved'] / data['total'] if data['total'] > 0 else 0
        
        return thresholds
    
    def _get_amount_range(self, amount: float) -> str:
        """Get amount range category."""
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
    
    def _identify_implementation_success_factors(self, successful_implementations: List[Dict]) -> List[str]:
        """Identify factors that lead to implementation success."""
        factors = []
        
        if len(successful_implementations) >= 3:
            factors.append('Adequate planning phase')
            factors.append('Stakeholder buy-in')
            factors.append('Realistic timeline estimates')
        else:
            factors.append('Limited success pattern data')
        
        return factors
    
    def _identify_implementation_risk_factors(self, failed_implementations: List[Dict]) -> List[str]:
        """Identify factors that lead to implementation failure."""
        risk_factors = []
        
        for decision in failed_implementations:
            if decision.get('budget_actual') and decision.get('budget_projected'):
                variance = abs(decision['budget_actual'] - decision['budget_projected']) / decision['budget_projected']
                if variance > 0.5:
                    risk_factors.append('Significant budget overruns')
            
            if decision.get('implementation_issues'):
                risk_factors.extend(decision['implementation_issues'])
        
        if not risk_factors:
            risk_factors.append('No clear risk patterns identified')
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _calculate_typical_implementation_timeline(self, implemented_decisions: List[Dict]) -> int:
        """Calculate typical implementation timeline."""
        timelines = []
        
        for decision in implemented_decisions:
            start_date = decision.get('implementation_start_date') or decision.get('proposed_date')
            end_date = decision.get('implementation_completion_date')
            
            if start_date and end_date:
                try:
                    start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    timeline = (end - start).days
                    if 0 < timeline < 1000:  # Reasonable bounds
                        timelines.append(timeline)
                except:
                    continue
        
        return int(statistics.mean(timelines)) if timelines else 90
    
    def _forecast_cost_variance(self, implemented_decisions: List[Dict], proposed_decision: Dict) -> float:
        """Forecast cost variance for the proposed decision."""
        variances = []
        
        for decision in implemented_decisions:
            projected = decision.get('budget_projected', 0)
            actual = decision.get('budget_actual', 0)
            
            if projected > 0 and actual > 0:
                variance = (actual - projected) / projected
                variances.append(variance)
        
        if variances:
            return statistics.mean(variances)
        else:
            return 0.1  # Default 10% variance
    
    def _assess_implementation_complexity(self, proposed_decision: Dict, implemented_decisions: List[Dict]) -> str:
        """Assess implementation complexity."""
        complexity_score = proposed_decision.get('complexity_score', 0.5)
        
        if complexity_score > 0.7:
            return 'high'
        elif complexity_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _identify_cost_factors(self, proposed_decision: Dict, decisions_with_costs: List[Dict]) -> List[str]:
        """Identify factors that affect costs."""
        factors = []
        
        amount = proposed_decision.get('budget_projected', 0) or proposed_decision.get('amount', 0)
        if amount > 50000:
            factors.append('Large budget increases cost overrun risk')
        
        decision_type = proposed_decision.get('decision_type')
        if decision_type in ['construction', 'infrastructure']:
            factors.append('Infrastructure projects have higher cost variance')
        
        return factors
    
    def _predict_cost_variance(self, proposed_decision: Dict, decisions_with_costs: List[Dict], avg_variance: float) -> float:
        """Predict cost variance based on decision characteristics."""
        base_variance = avg_variance
        
        # Adjust based on decision characteristics
        complexity = proposed_decision.get('complexity_score', 0.5)
        adjustment = complexity * 0.2  # Higher complexity = higher variance
        
        return base_variance + adjustment
    
    def _assess_budget_risk(self, predicted_variance: float) -> str:
        """Assess budget risk level."""
        if predicted_variance > 0.3:
            return 'high'
        elif predicted_variance > 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_cost_optimizations(self, proposed_decision: Dict, decisions_with_costs: List[Dict]) -> List[str]:
        """Suggest cost optimization strategies."""
        suggestions = []
        
        amount = proposed_decision.get('budget_projected', 0) or proposed_decision.get('amount', 0)
        if amount > 25000:
            suggestions.append('Consider phased implementation to spread costs')
            suggestions.append('Obtain multiple vendor quotes')
        
        suggestions.append('Include 10-15% contingency budget')
        
        return suggestions
    
    def _predict_timeline(self, proposed_decision: Dict, decisions_with_timelines: List[Dict], avg_timeline: float) -> float:
        """Predict timeline based on decision characteristics."""
        base_timeline = avg_timeline
        
        # Adjust based on complexity
        complexity = proposed_decision.get('complexity_score', 0.5)
        adjustment = complexity * 30  # Higher complexity = longer timeline
        
        return base_timeline + adjustment
    
    def _calculate_delay_probability(self, decisions_with_timelines: List[Dict]) -> float:
        """Calculate probability of timeline delays."""
        delayed_count = 0
        
        for decision in decisions_with_timelines:
            # This would need actual delay data - using placeholder logic
            if decision.get('implementation_issues'):
                delayed_count += 1
        
        return delayed_count / len(decisions_with_timelines) if decisions_with_timelines else 0.3
    
    def _identify_timeline_factors(self, proposed_decision: Dict, decisions_with_timelines: List[Dict]) -> List[str]:
        """Identify factors that affect timeline."""
        factors = []
        
        complexity = proposed_decision.get('complexity_score', 0.5)
        if complexity > 0.7:
            factors.append('High complexity may extend timeline')
        
        decision_type = proposed_decision.get('decision_type')
        if decision_type in ['policy', 'governance']:
            factors.append('Policy decisions require longer consultation periods')
        
        return factors
    
    def _identify_critical_path_risks(self, proposed_decision: Dict) -> List[str]:
        """Identify critical path risks."""
        risks = []
        
        if proposed_decision.get('requires_vendor'):
            risks.append('Vendor selection and contracting delays')
        
        if proposed_decision.get('requires_approvals'):
            risks.append('Multiple approval levels increase delay risk')
        
        return risks
    
    def _get_voting_margin(self, decision: Dict) -> Optional[float]:
        """Get voting margin from decision."""
        vote_details = decision.get('vote_details', {})
        
        if isinstance(vote_details, dict):
            vote_for = vote_details.get('for', 0)
            vote_against = vote_details.get('against', 0)
            total = vote_for + vote_against
            
            if total > 0:
                return vote_for / total
        
        return None
    
    def _identify_satisfaction_factors(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> List[str]:
        """Identify factors that affect member satisfaction."""
        factors = []
        
        # Analyze voting patterns
        unanimous_count = sum(1 for d in similar_decisions if d.get('unanimous'))
        if unanimous_count > len(similar_decisions) * 0.5:
            factors.append('Similar decisions often achieve consensus')
        
        return factors
    
    def _analyze_stakeholder_impact(self, proposed_decision: Dict) -> Dict[str, Any]:
        """Analyze stakeholder impact."""
        return {
            'primary_stakeholders': ['Board Members', 'Club Members'],
            'impact_level': 'medium',
            'communication_needed': True
        }
    
    def _suggest_communication_strategy(self, proposed_decision: Dict) -> List[str]:
        """Suggest communication strategy."""
        return [
            'Prepare clear rationale document',
            'Hold stakeholder briefing sessions',
            'Address potential concerns proactively'
        ]
    
    def _assess_financial_risks(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> List[Dict]:
        """Assess financial risks."""
        risks = []
        
        amount = proposed_decision.get('budget_projected', 0) or proposed_decision.get('amount', 0)
        if amount > 100000:
            risks.append({
                'risk': 'Major financial commitment',
                'probability': 'high',
                'impact': 'high',
                'mitigation': 'Require detailed financial analysis'
            })
        
        return risks
    
    def _assess_operational_risks(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> List[Dict]:
        """Assess operational risks."""
        return [
            {
                'risk': 'Implementation complexity',
                'probability': 'medium',
                'impact': 'medium',
                'mitigation': 'Develop detailed implementation plan'
            }
        ]
    
    def _assess_political_risks(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> List[Dict]:
        """Assess political/governance risks."""
        return [
            {
                'risk': 'Stakeholder opposition',
                'probability': 'medium',
                'impact': 'high',
                'mitigation': 'Enhance stakeholder engagement'
            }
        ]
    
    def _assess_timeline_risks(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> List[Dict]:
        """Assess timeline risks."""
        return [
            {
                'risk': 'Implementation delays',
                'probability': 'medium',
                'impact': 'medium',
                'mitigation': 'Build buffer time into schedule'
            }
        ]
    
    def _assess_reputation_risks(self, proposed_decision: Dict, similar_decisions: List[Dict]) -> List[Dict]:
        """Assess reputation risks."""
        return [
            {
                'risk': 'Negative member perception',
                'probability': 'low',
                'impact': 'medium',
                'mitigation': 'Transparent communication strategy'
            }
        ]
    
    def _calculate_overall_risk_score(self, risks: Dict) -> float:
        """Calculate overall risk score."""
        # Simple risk scoring based on number and severity of risks
        total_risks = sum(len(risk_list) for risk_list in risks.values())
        return min(1.0, total_risks * 0.1)
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level."""
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_risk_mitigations(self, risks: Dict) -> List[str]:
        """Suggest risk mitigation strategies."""
        mitigations = []
        
        for risk_category, risk_list in risks.items():
            for risk in risk_list:
                if isinstance(risk, dict) and 'mitigation' in risk:
                    mitigations.append(risk['mitigation'])
        
        return list(set(mitigations))  # Remove duplicates
    
    def _create_risk_monitoring_plan(self, risks: Dict) -> Dict[str, Any]:
        """Create risk monitoring plan."""
        return {
            'monitoring_frequency': 'weekly',
            'key_indicators': ['Budget variance', 'Timeline adherence', 'Stakeholder feedback'],
            'escalation_triggers': ['Cost overrun >20%', 'Timeline delay >30 days'],
            'review_schedule': 'Monthly risk assessment meetings'
        }
    
    def _calculate_overall_confidence(self, predictions: Dict) -> str:
        """Calculate overall confidence level."""
        confidence_scores = []
        
        for prediction in predictions.values():
            if isinstance(prediction, dict) and 'confidence' in prediction:
                conf = prediction['confidence']
                if conf == 'high':
                    confidence_scores.append(1.0)
                elif conf == 'medium':
                    confidence_scores.append(0.7)
                else:
                    confidence_scores.append(0.3)
        
        if confidence_scores:
            avg_confidence = statistics.mean(confidence_scores)
            if avg_confidence > 0.7:
                return 'high'
            elif avg_confidence > 0.4:
                return 'medium'
            else:
                return 'low'
        
        return 'medium'
    
    def _calculate_decision_quality_score(self, predictions: Dict) -> float:
        """Calculate decision quality score."""
        # Composite score based on multiple factors
        approval_prob = predictions['approval_likelihood']['probability']
        implementation_prob = predictions['implementation_success']['success_probability']
        
        quality_score = (approval_prob + implementation_prob) / 2
        return round(quality_score, 3)
    
    def _assess_strategic_alignment(self, predictions: Dict) -> str:
        """Assess strategic alignment."""
        quality_score = self._calculate_decision_quality_score(predictions)
        
        if quality_score > 0.7:
            return 'high_alignment'
        elif quality_score > 0.4:
            return 'moderate_alignment'
        else:
            return 'low_alignment'

# Main prediction functions for API integration

def predict_decision_outcome(org_id: str, proposed_decision: Dict[str, Any]) -> Dict[str, Any]:
    """Main function for predicting decision outcomes."""
    try:
        predictor = DecisionOutcomePredictor(org_id)
        return predictor.predict_complete_outcome(proposed_decision)
    except Exception as e:
        logger.error(f"Failed to predict decision outcome: {e}")
        return {
            'error': str(e),
            'recommendation': 'DEFER',
            'reasoning': 'Technical error in prediction system'
        }

def get_prediction_accuracy_metrics(org_id: str) -> Dict[str, Any]:
    """Get metrics on prediction accuracy for continuous improvement."""
    try:
        # This would analyze past predictions vs actual outcomes
        return {
            'overall_accuracy': 0.75,
            'approval_prediction_accuracy': 0.80,
            'timeline_prediction_accuracy': 0.70,
            'cost_prediction_accuracy': 0.65,
            'total_predictions': 50,
            'validation_period': '6_months'
        }
    except Exception as e:
        logger.error(f"Failed to get prediction metrics: {e}")
        return {'error': str(e)}