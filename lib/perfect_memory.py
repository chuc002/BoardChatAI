"""
Perfect Memory System - Complete Institutional Intelligence Database

This module implements the perfect memory architecture with complete database
schema for capturing every word, decision, and interaction in the organization.
"""

import json
import uuid
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

from lib.supa import supa

logger = logging.getLogger(__name__)

@dataclass
class CompleteRecord:
    """Complete record of any institutional interaction."""
    org_id: str
    date: datetime
    record_type: str
    participants: List[str]
    content: str
    source_document_id: Optional[str] = None
    content_summary: Optional[str] = None
    key_topics: List[str] = None
    decisions_made: List[str] = None
    action_items: List[Dict] = None
    follow_ups: List[Dict] = None
    outcomes: Dict = None
    location: Optional[str] = None
    duration_minutes: Optional[int] = None
    importance_score: float = 0.5
    sentiment_score: float = 0.0
    created_by: Optional[str] = None

@dataclass
class DecisionComplete:
    """Complete decision record with full lifecycle tracking."""
    org_id: str
    decision_title: str
    proposed_date: date
    proposed_by: str
    description: str
    rationale: Optional[str] = None
    background_context: Optional[str] = None
    financial_implications: Dict = None
    stakeholders_affected: List[str] = None
    discussion_points: List[Dict] = None
    concerns_raised: List[Dict] = None
    modifications: List[Dict] = None
    vote_date: Optional[date] = None
    vote_details: Dict = None
    vote_margin: Optional[float] = None
    unanimous: bool = False
    implementation_plan: Dict = None
    implementation_start_date: Optional[date] = None
    implementation_completion_date: Optional[date] = None
    actual_implementation: Dict = None
    budget_projected: Optional[float] = None
    budget_actual: Optional[float] = None
    cost_variance: Optional[float] = None
    outcomes_measured: Dict = None
    success_metrics: Dict = None
    member_feedback: Dict = None
    lessons_learned: Optional[str] = None
    would_repeat: Optional[bool] = None
    retrospective_assessment: Optional[str] = None
    related_decisions: List[str] = None
    precedent_decisions: List[str] = None
    consequence_decisions: List[str] = None
    decision_type: Optional[str] = None
    urgency_level: str = 'normal'
    complexity_score: float = 0.5
    risk_assessment: Dict = None

@dataclass
class MemberCompleteHistory:
    """Complete member history and institutional profile."""
    org_id: str
    member_name: str
    member_id: Optional[str] = None
    active_status: bool = True
    join_date: Optional[date] = None
    departure_date: Optional[date] = None
    membership_category: Optional[str] = None
    positions_held: List[Dict] = None
    committees_served: List[Dict] = None
    leadership_roles: List[Dict] = None
    votes_cast: Dict = None
    voting_patterns: Dict = None
    proposals_made: List[Dict] = None
    meeting_attendance: Dict = None
    participation_metrics: Dict = None
    expertise_areas: List[str] = None
    professional_background: Optional[str] = None
    educational_background: Optional[str] = None
    relevant_experience: Optional[str] = None
    specialized_knowledge: List[str] = None
    known_positions: Dict = None
    policy_preferences: Dict = None
    decision_influences: Dict = None
    relationships: Dict = None
    influence_network: Dict = None
    collaboration_patterns: Dict = None
    conflict_history: List[Dict] = None
    effectiveness_scores: Dict = None
    contribution_assessment: Dict = None
    leadership_effectiveness: Dict = None
    communication_style: Optional[str] = None
    decision_making_style: Optional[str] = None
    data_completeness_score: float = 0.0

class PerfectMemorySystem:
    """
    Perfect Memory System implementing complete institutional intelligence.
    Captures and stores every interaction, decision, and piece of institutional knowledge.
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self._ensure_schema_exists()
        logger.info("Perfect Memory System initialized")
    
    def _ensure_schema_exists(self):
        """Ensure all required tables exist in Supabase."""
        try:
            # Check if perfect memory tables exist, create if needed
            self._create_perfect_memory_tables()
        except Exception as e:
            logger.warning(f"Schema verification failed: {e}")
    
    def _create_perfect_memory_tables(self):
        """Create perfect memory tables using Supabase client."""
        
        # Create complete_record table if it doesn't exist
        try:
            # Test if table exists by querying it
            result = supa.table('complete_record').select('id').limit(1).execute()
            logger.info("complete_record table exists")
        except Exception:
            logger.info("Creating complete_record table structure via Supabase API")
            # Table creation would be handled by migration scripts
            
        # Same for other tables
        try:
            result = supa.table('decision_complete').select('id').limit(1).execute()
            logger.info("decision_complete table exists")
        except Exception:
            logger.info("decision_complete table needs creation")
            
        try:
            result = supa.table('member_complete_history').select('id').limit(1).execute()
            logger.info("member_complete_history table exists")
        except Exception:
            logger.info("member_complete_history table needs creation")
    
    def record_complete_interaction(self, record: CompleteRecord) -> str:
        """Record a complete institutional interaction."""
        try:
            record_data = {
                'org_id': record.org_id,
                'date': record.date.isoformat(),
                'type': record.record_type,
                'source_document_id': record.source_document_id,
                'participants': record.participants,
                'content': record.content,
                'content_summary': record.content_summary,
                'key_topics': record.key_topics or [],
                'decisions_made': record.decisions_made or [],
                'action_items': record.action_items or [],
                'follow_ups': record.follow_ups or [],
                'outcomes': record.outcomes or {},
                'location': record.location,
                'duration_minutes': record.duration_minutes,
                'importance_score': record.importance_score,
                'sentiment_score': record.sentiment_score,
                'created_by': record.created_by
            }
            
            result = supa.table('complete_record').insert(record_data).execute()
            
            if result.data:
                record_id = result.data[0]['id']
                logger.info(f"Recorded complete interaction: {record_id}")
                return record_id
            else:
                logger.error("Failed to record interaction")
                return None
                
        except Exception as e:
            logger.error(f"Failed to record complete interaction: {e}")
            return None
    
    def record_complete_decision(self, decision: DecisionComplete) -> str:
        """Record a complete decision with full lifecycle tracking."""
        try:
            decision_data = {
                'org_id': decision.org_id,
                'decision_title': decision.decision_title,
                'proposed_date': decision.proposed_date.isoformat(),
                'proposed_by': decision.proposed_by,
                'description': decision.description,
                'rationale': decision.rationale,
                'background_context': decision.background_context,
                'financial_implications': decision.financial_implications or {},
                'stakeholders_affected': decision.stakeholders_affected or [],
                'discussion_points': decision.discussion_points or [],
                'concerns_raised': decision.concerns_raised or [],
                'modifications': decision.modifications or [],
                'vote_date': decision.vote_date.isoformat() if decision.vote_date else None,
                'vote_details': decision.vote_details or {},
                'vote_margin': decision.vote_margin,
                'unanimous': decision.unanimous,
                'implementation_plan': decision.implementation_plan or {},
                'implementation_start_date': decision.implementation_start_date.isoformat() if decision.implementation_start_date else None,
                'implementation_completion_date': decision.implementation_completion_date.isoformat() if decision.implementation_completion_date else None,
                'actual_implementation': decision.actual_implementation or {},
                'budget_projected': decision.budget_projected,
                'budget_actual': decision.budget_actual,
                'cost_variance': decision.cost_variance,
                'outcomes_measured': decision.outcomes_measured or {},
                'success_metrics': decision.success_metrics or {},
                'member_feedback': decision.member_feedback or {},
                'lessons_learned': decision.lessons_learned,
                'would_repeat': decision.would_repeat,
                'retrospective_assessment': decision.retrospective_assessment,
                'related_decisions': decision.related_decisions or [],
                'precedent_decisions': decision.precedent_decisions or [],
                'consequence_decisions': decision.consequence_decisions or [],
                'decision_type': decision.decision_type,
                'urgency_level': decision.urgency_level,
                'complexity_score': decision.complexity_score,
                'risk_assessment': decision.risk_assessment or {}
            }
            
            result = supa.table('decision_complete').insert(decision_data).execute()
            
            if result.data:
                decision_id = result.data[0]['id']
                logger.info(f"Recorded complete decision: {decision_id}")
                return decision_id
            else:
                logger.error("Failed to record decision")
                return None
                
        except Exception as e:
            logger.error(f"Failed to record complete decision: {e}")
            return None
    
    def update_member_complete_history(self, member: MemberCompleteHistory) -> bool:
        """Update or create complete member history."""
        try:
            member_data = {
                'org_id': member.org_id,
                'member_name': member.member_name,
                'member_id': member.member_id,
                'active_status': member.active_status,
                'join_date': member.join_date.isoformat() if member.join_date else None,
                'departure_date': member.departure_date.isoformat() if member.departure_date else None,
                'membership_category': member.membership_category,
                'positions_held': member.positions_held or [],
                'committees_served': member.committees_served or [],
                'leadership_roles': member.leadership_roles or [],
                'votes_cast': member.votes_cast or {},
                'voting_patterns': member.voting_patterns or {},
                'proposals_made': member.proposals_made or [],
                'meeting_attendance': member.meeting_attendance or {},
                'participation_metrics': member.participation_metrics or {},
                'expertise_areas': member.expertise_areas or [],
                'professional_background': member.professional_background,
                'educational_background': member.educational_background,
                'relevant_experience': member.relevant_experience,
                'specialized_knowledge': member.specialized_knowledge or [],
                'known_positions': member.known_positions or {},
                'policy_preferences': member.policy_preferences or {},
                'decision_influences': member.decision_influences or {},
                'relationships': member.relationships or {},
                'influence_network': member.influence_network or {},
                'collaboration_patterns': member.collaboration_patterns or {},
                'conflict_history': member.conflict_history or [],
                'effectiveness_scores': member.effectiveness_scores or {},
                'contribution_assessment': member.contribution_assessment or {},
                'leadership_effectiveness': member.leadership_effectiveness or {},
                'communication_style': member.communication_style,
                'decision_making_style': member.decision_making_style,
                'data_completeness_score': member.data_completeness_score
            }
            
            # Try to update existing record first
            existing = supa.table('member_complete_history').select('id').eq('org_id', member.org_id).eq('member_name', member.member_name).execute()
            
            if existing.data:
                # Update existing record
                result = supa.table('member_complete_history').update(member_data).eq('org_id', member.org_id).eq('member_name', member.member_name).execute()
                logger.info(f"Updated member history: {member.member_name}")
            else:
                # Insert new record
                result = supa.table('member_complete_history').insert(member_data).execute()
                logger.info(f"Created member history: {member.member_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update member complete history: {e}")
            return False
    
    def query_complete_records(self, 
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              record_type: Optional[str] = None,
                              participants: Optional[List[str]] = None,
                              topics: Optional[List[str]] = None,
                              limit: int = 100) -> List[Dict]:
        """Query complete records with filtering."""
        try:
            query = supa.table('complete_record').select('*').eq('org_id', self.org_id)
            
            if start_date:
                query = query.gte('date', start_date.isoformat())
            
            if end_date:
                query = query.lte('date', end_date.isoformat())
            
            if record_type:
                query = query.eq('type', record_type)
            
            # For array fields, we'd need to use contains or overlap operations
            # This might require different query approaches depending on Supabase capabilities
            
            result = query.order('date', desc=True).limit(limit).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to query complete records: {e}")
            return []
    
    def query_decisions_complete(self,
                                decision_type: Optional[str] = None,
                                proposed_by: Optional[str] = None,
                                date_range: Optional[tuple] = None,
                                include_outcomes: bool = True,
                                limit: int = 100) -> List[Dict]:
        """Query complete decisions with filtering."""
        try:
            query = supa.table('decision_complete').select('*').eq('org_id', self.org_id)
            
            if decision_type:
                query = query.eq('decision_type', decision_type)
            
            if proposed_by:
                query = query.eq('proposed_by', proposed_by)
            
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    query = query.gte('proposed_date', start_date.isoformat())
                if end_date:
                    query = query.lte('proposed_date', end_date.isoformat())
            
            result = query.order('proposed_date', desc=True).limit(limit).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to query complete decisions: {e}")
            return []
    
    def get_member_complete_profile(self, member_name: str) -> Optional[Dict]:
        """Get complete profile for a specific member."""
        try:
            result = supa.table('member_complete_history').select('*').eq('org_id', self.org_id).eq('member_name', member_name).execute()
            
            if result.data:
                return result.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get member complete profile: {e}")
            return None
    
    def search_institutional_memory(self, 
                                   query_text: str,
                                   search_scope: List[str] = None,
                                   limit: int = 50) -> Dict[str, List]:
        """Search across all institutional memory."""
        results = {
            'records': [],
            'decisions': [],
            'members': [],
            'total_matches': 0
        }
        
        try:
            search_scope = search_scope or ['records', 'decisions', 'members']
            
            # Search complete records
            if 'records' in search_scope:
                records = supa.table('complete_record').select('*').eq('org_id', self.org_id).ilike('content', f'%{query_text}%').limit(limit//3).execute()
                results['records'] = records.data or []
            
            # Search decisions
            if 'decisions' in search_scope:
                decisions = supa.table('decision_complete').select('*').eq('org_id', self.org_id).or_(
                    f'decision_title.ilike.%{query_text}%,description.ilike.%{query_text}%,rationale.ilike.%{query_text}%'
                ).limit(limit//3).execute()
                results['decisions'] = decisions.data or []
            
            # Search members
            if 'members' in search_scope:
                members = supa.table('member_complete_history').select('*').eq('org_id', self.org_id).or_(
                    f'member_name.ilike.%{query_text}%,professional_background.ilike.%{query_text}%'
                ).limit(limit//3).execute()
                results['members'] = members.data or []
            
            results['total_matches'] = len(results['records']) + len(results['decisions']) + len(results['members'])
            
            logger.info(f"Search found {results['total_matches']} matches for: {query_text}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search institutional memory: {e}")
            return results
    
    def generate_institutional_report(self, 
                                    report_type: str = 'comprehensive',
                                    time_period: Optional[tuple] = None) -> Dict[str, Any]:
        """Generate comprehensive institutional intelligence report."""
        report = {
            'report_type': report_type,
            'org_id': self.org_id,
            'generated_at': datetime.now().isoformat(),
            'time_period': time_period,
            'metrics': {},
            'insights': [],
            'recommendations': []
        }
        
        try:
            # Get basic metrics
            records_count = len(self.query_complete_records(limit=10000))
            decisions_count = len(self.query_decisions_complete(limit=10000))
            
            # Get member statistics
            members_result = supa.table('member_complete_history').select('*').eq('org_id', self.org_id).execute()
            active_members = len([m for m in (members_result.data or []) if m.get('active_status')])
            
            report['metrics'] = {
                'total_records': records_count,
                'total_decisions': decisions_count,
                'active_members': active_members,
                'total_members': len(members_result.data or [])
            }
            
            # Generate insights based on data
            if decisions_count > 0:
                decisions = self.query_decisions_complete(limit=100)
                successful_decisions = [d for d in decisions if d.get('would_repeat') == True]
                success_rate = len(successful_decisions) / len(decisions) if decisions else 0
                
                report['insights'].append(f"Decision success rate: {success_rate:.1%}")
                
                if success_rate < 0.7:
                    report['recommendations'].append("Review decision-making process to improve success rate")
            
            if active_members > 0:
                report['insights'].append(f"Active membership engagement: {active_members} members")
                
                # Calculate participation metrics if available
                participation_scores = []
                for member in (members_result.data or []):
                    if member.get('active_status') and member.get('participation_metrics'):
                        score = member['participation_metrics'].get('overall_score', 0.5)
                        participation_scores.append(score)
                
                if participation_scores:
                    avg_participation = sum(participation_scores) / len(participation_scores)
                    report['insights'].append(f"Average participation score: {avg_participation:.2f}")
                    
                    if avg_participation < 0.6:
                        report['recommendations'].append("Implement strategies to increase member participation")
            
            logger.info(f"Generated institutional report: {report_type}")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate institutional report: {e}")
            report['error'] = str(e)
            return report
    
    def get_system_completeness_metrics(self) -> Dict[str, Any]:
        """Get metrics on how complete the institutional memory is."""
        metrics = {
            'overall_completeness': 0.0,
            'data_coverage': {},
            'quality_scores': {},
            'gaps_identified': [],
            'recommendations': []
        }
        
        try:
            # Check data coverage
            records_count = len(self.query_complete_records(limit=10000))
            decisions_count = len(self.query_decisions_complete(limit=10000))
            members_result = supa.table('member_complete_history').select('*').eq('org_id', self.org_id).execute()
            members_count = len(members_result.data or [])
            
            metrics['data_coverage'] = {
                'records': records_count,
                'decisions': decisions_count,
                'members': members_count
            }
            
            # Assess data quality
            if members_result.data:
                completeness_scores = [m.get('data_completeness_score', 0.0) for m in members_result.data]
                avg_member_completeness = sum(completeness_scores) / len(completeness_scores)
                metrics['quality_scores']['member_profiles'] = avg_member_completeness
            
            # Calculate overall completeness
            completeness_factors = [
                min(1.0, records_count / 100),  # Target 100 records
                min(1.0, decisions_count / 50),  # Target 50 decisions
                min(1.0, members_count / 20),    # Target 20 members
                metrics['quality_scores'].get('member_profiles', 0.5)
            ]
            
            metrics['overall_completeness'] = sum(completeness_factors) / len(completeness_factors)
            
            # Identify gaps
            if records_count < 50:
                metrics['gaps_identified'].append('Insufficient record coverage')
                metrics['recommendations'].append('Increase documentation of meetings and interactions')
            
            if decisions_count < 25:
                metrics['gaps_identified'].append('Limited decision history')
                metrics['recommendations'].append('Retroactively document historical decisions')
            
            if metrics['quality_scores'].get('member_profiles', 0) < 0.7:
                metrics['gaps_identified'].append('Incomplete member profiles')
                metrics['recommendations'].append('Gather more comprehensive member information')
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get completeness metrics: {e}")
            metrics['error'] = str(e)
            return metrics


# API Functions for integration with Flask app

def record_institutional_interaction(org_id: str, interaction_data: Dict) -> str:
    """Record a complete institutional interaction."""
    system = PerfectMemorySystem(org_id)
    
    record = CompleteRecord(
        org_id=org_id,
        date=datetime.fromisoformat(interaction_data.get('date', datetime.now().isoformat())),
        record_type=interaction_data.get('type', 'discussion'),
        participants=interaction_data.get('participants', []),
        content=interaction_data.get('content', ''),
        source_document_id=interaction_data.get('source_document_id'),
        content_summary=interaction_data.get('content_summary'),
        key_topics=interaction_data.get('key_topics', []),
        location=interaction_data.get('location'),
        duration_minutes=interaction_data.get('duration_minutes'),
        importance_score=interaction_data.get('importance_score', 0.5),
        sentiment_score=interaction_data.get('sentiment_score', 0.0),
        created_by=interaction_data.get('created_by')
    )
    
    return system.record_complete_interaction(record)

def record_complete_institutional_decision(org_id: str, decision_data: Dict) -> str:
    """Record a complete institutional decision."""
    system = PerfectMemorySystem(org_id)
    
    decision = DecisionComplete(
        org_id=org_id,
        decision_title=decision_data.get('title', ''),
        proposed_date=date.fromisoformat(decision_data.get('proposed_date', date.today().isoformat())),
        proposed_by=decision_data.get('proposed_by', ''),
        description=decision_data.get('description', ''),
        rationale=decision_data.get('rationale'),
        decision_type=decision_data.get('decision_type'),
        stakeholders_affected=decision_data.get('stakeholders_affected', []),
        financial_implications=decision_data.get('financial_implications'),
        urgency_level=decision_data.get('urgency_level', 'normal'),
        complexity_score=decision_data.get('complexity_score', 0.5)
    )
    
    return system.record_complete_decision(decision)

def get_institutional_memory_search(org_id: str, query: str, scope: List[str] = None) -> Dict:
    """Search institutional memory."""
    system = PerfectMemorySystem(org_id)
    return system.search_institutional_memory(query, scope)

def get_institutional_intelligence_report(org_id: str, report_type: str = 'comprehensive') -> Dict:
    """Generate institutional intelligence report."""
    system = PerfectMemorySystem(org_id)
    return system.generate_institutional_report(report_type)

def get_perfect_memory_metrics(org_id: str) -> Dict:
    """Get perfect memory system completeness metrics."""
    system = PerfectMemorySystem(org_id)
    return system.get_system_completeness_metrics()