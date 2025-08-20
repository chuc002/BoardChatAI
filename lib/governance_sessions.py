import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from lib.supa import supa

logger = logging.getLogger(__name__)

@dataclass
class GovernanceSession:
    """Persistent institutional memory session like Claude Projects"""
    
    session_id: str
    org_id: str
    title: str
    created_at: datetime
    last_updated: datetime
    conversation_summary: str
    decisions_analyzed: List[Dict[str, Any]]
    precedents_identified: List[Dict[str, Any]]
    patterns_discovered: List[Dict[str, Any]]
    next_actions: List[str]
    key_insights: List[str]
    participants: List[str]
    session_metrics: Dict[str, Any]
    status: str = "active"

class GovernanceSessionManager:
    """Manage persistent governance sessions for institutional memory"""
    
    def __init__(self):
        self.current_sessions = {}
        self._ensure_session_table()
    
    def _ensure_session_table(self):
        """Ensure governance sessions table exists"""
        try:
            # Test if table exists
            supa.table("governance_sessions").select("id").limit(1).execute()
            logger.info("Governance sessions table verified")
        except Exception as e:
            logger.warning(f"Governance sessions table may not exist: {e}")
            # Table creation would happen through migrations in production
    
    def create_session_note(self, org_id: str, conversation: Dict[str, Any], 
                           title: Optional[str] = None) -> GovernanceSession:
        """Create comprehensive session note from conversation"""
        
        session_id = f"gs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{org_id[:8]}"
        current_time = datetime.now()
        
        # Extract governance intelligence from conversation
        decisions = self.extract_decisions(conversation)
        precedents = self.find_precedents(conversation) 
        patterns = self.identify_patterns(conversation)
        actions = self.determine_follow_ups(conversation)
        insights = self.generate_key_insights(conversation)
        participants = self.extract_participants(conversation)
        
        # Generate session title if not provided
        if not title:
            title = self._generate_session_title(decisions, patterns)
        
        # Create session object
        session = GovernanceSession(
            session_id=session_id,
            org_id=org_id,
            title=title,
            created_at=current_time,
            last_updated=current_time,
            conversation_summary=self._create_conversation_summary(conversation),
            decisions_analyzed=decisions,
            precedents_identified=precedents,
            patterns_discovered=patterns,
            next_actions=actions,
            key_insights=insights,
            participants=participants,
            session_metrics=self._calculate_session_metrics(conversation),
            status="active"
        )
        
        # Store in database
        self._store_session(session)
        
        # Cache locally
        self.current_sessions[session_id] = session
        
        logger.info(f"Created governance session: {title} ({session_id})")
        return session
    
    def extract_decisions(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract decisions from conversation content"""
        
        decisions = []
        content = self._get_conversation_text(conversation)
        
        # Decision patterns
        decision_indicators = [
            r'(?i)(?:decided|resolved|approved|adopted)\s+(?:that\s+)?([^.]{20,150})',
            r'(?i)(?:motion\s+)?(?:carried|passed)\s*[:\-]\s*([^.]{20,150})',
            r'(?i)(?:board|committee)\s+(?:agrees|approves)\s+([^.]{20,150})',
            r'(?i)(?:unanimous|majority)\s+(?:decision|vote)\s+(?:on\s+)?([^.]{20,150})'
        ]
        
        import re
        for pattern in decision_indicators:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match.strip()) > 10:
                    decisions.append({
                        'decision_text': match.strip(),
                        'confidence': 0.8,
                        'type': 'extracted_decision',
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Add query-based decisions
        queries = conversation.get('queries', [])
        for query in queries:
            if any(word in query.lower() for word in ['decide', 'approve', 'vote', 'motion']):
                decisions.append({
                    'decision_context': query,
                    'confidence': 0.6,
                    'type': 'decision_query',
                    'timestamp': datetime.now().isoformat()
                })
        
        return decisions[:10]  # Limit to top 10
    
    def find_precedents(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify precedents and historical patterns"""
        
        precedents = []
        content = self._get_conversation_text(conversation)
        
        # Precedent indicators
        precedent_patterns = [
            r'(?i)(?:historically|previously|in the past)\s+([^.]{20,100})',
            r'(?i)(?:precedent|tradition|custom)\s+(?:of\s+)?([^.]{20,100})',
            r'(?i)(?:similar\s+)?(?:case|situation|instance)\s+([^.]{20,100})',
            r'(?i)(?:based on|following|according to)\s+([^.]{20,100})'
        ]
        
        import re
        for pattern in precedent_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                precedents.append({
                    'precedent_text': match.strip(),
                    'type': 'historical_reference',
                    'confidence': 0.7,
                    'timestamp': datetime.now().isoformat()
                })
        
        return precedents[:8]  # Limit to top 8
    
    def identify_patterns(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover governance patterns from conversation"""
        
        patterns = []
        content = self._get_conversation_text(conversation)
        
        # Pattern analysis
        pattern_indicators = {
            'financial_pattern': ['budget', 'cost', 'fee', 'expense', '$'],
            'membership_pattern': ['member', 'join', 'application', 'dues'],
            'committee_pattern': ['committee', 'board', 'group', 'team'],
            'policy_pattern': ['rule', 'policy', 'regulation', 'guideline'],
            'meeting_pattern': ['meeting', 'agenda', 'minutes', 'vote'],
            'facility_pattern': ['facility', 'building', 'maintenance', 'improvement']
        }
        
        content_lower = content.lower()
        for pattern_type, keywords in pattern_indicators.items():
            keyword_count = sum(content_lower.count(keyword) for keyword in keywords)
            
            if keyword_count >= 3:  # Threshold for pattern detection
                patterns.append({
                    'pattern_type': pattern_type.replace('_pattern', ''),
                    'keyword_count': keyword_count,
                    'confidence': min(keyword_count / 10.0, 1.0),
                    'keywords_found': [kw for kw in keywords if kw in content_lower],
                    'timestamp': datetime.now().isoformat()
                })
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:6]
    
    def determine_follow_ups(self, conversation: Dict[str, Any]) -> List[str]:
        """Determine next actions from conversation"""
        
        actions = []
        content = self._get_conversation_text(conversation)
        
        # Action patterns
        action_indicators = [
            r'(?i)(?:need to|should|must|will)\s+([^.]{10,80})',
            r'(?i)(?:action item|follow.?up|next step)\s*[:\-]\s*([^.]{10,80})',
            r'(?i)(?:schedule|plan|prepare)\s+([^.]{10,80})',
            r'(?i)(?:review|investigate|research)\s+([^.]{10,80})'
        ]
        
        import re
        for pattern in action_indicators:
            matches = re.findall(pattern, content)
            for match in matches:
                clean_action = match.strip()
                if len(clean_action) > 8 and clean_action not in actions:
                    actions.append(clean_action)
        
        # Add standard governance follow-ups
        queries = conversation.get('queries', [])
        if any('budget' in q.lower() for q in queries):
            actions.append("Review budget implications and financial impact")
        if any('member' in q.lower() for q in queries):
            actions.append("Consider membership committee consultation")
        if any('policy' in q.lower() for q in queries):
            actions.append("Verify policy compliance and precedent alignment")
        
        return actions[:8]  # Limit to top 8
    
    def generate_key_insights(self, conversation: Dict[str, Any]) -> List[str]:
        """Generate key insights from governance conversation"""
        
        insights = []
        content = self._get_conversation_text(conversation)
        queries = conversation.get('queries', [])
        
        # Query-based insights
        if queries:
            unique_topics = set()
            for query in queries:
                # Extract key topics
                topic_words = [word.lower() for word in query.split() 
                             if len(word) > 4 and word.lower() not in 
                             ['about', 'what', 'how', 'when', 'where', 'why', 'which']]
                unique_topics.update(topic_words[:3])
            
            if unique_topics:
                insights.append(f"Primary governance focus: {', '.join(list(unique_topics)[:4])}")
        
        # Content-based insights
        if 'precedent' in content.lower():
            insights.append("Historical precedent analysis requested")
        if 'committee' in content.lower():
            insights.append("Multi-committee coordination involved")
        if '$' in content or 'budget' in content.lower():
            insights.append("Financial implications require consideration")
        if 'member' in content.lower():
            insights.append("Membership impact analysis needed")
        
        # Complexity insights
        if len(queries) > 3:
            insights.append("Complex multi-faceted governance inquiry")
        if len(content) > 2000:
            insights.append("Comprehensive governance analysis session")
        
        return insights[:6]  # Limit to top 6
    
    def extract_participants(self, conversation: Dict[str, Any]) -> List[str]:
        """Extract participants from conversation"""
        
        participants = ["BoardContinuity AI"]  # Always include AI
        content = self._get_conversation_text(conversation)
        
        # Look for names and roles
        import re
        name_patterns = [
            r'(?i)(?:president|chair|director|member|secretary|treasurer)\s+([A-Z][a-zA-Z\s]{2,25})',
            r'(?i)([A-Z][a-zA-Z\s]{2,25})\s+(?:said|asked|mentioned|reported|presented)'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                clean_name = match.strip()
                if len(clean_name) > 2 and clean_name not in participants:
                    participants.append(clean_name)
        
        return participants[:10]  # Limit to top 10
    
    def _get_conversation_text(self, conversation: Dict[str, Any]) -> str:
        """Extract text content from conversation object"""
        
        text_parts = []
        
        # Add queries
        if 'queries' in conversation:
            text_parts.extend(conversation['queries'])
        
        # Add responses
        if 'responses' in conversation:
            for response in conversation['responses']:
                if isinstance(response, dict):
                    text_parts.append(response.get('answer', response.get('response', '')))
                else:
                    text_parts.append(str(response))
        
        # Add any additional content
        if 'content' in conversation:
            text_parts.append(str(conversation['content']))
        
        return ' '.join(text_parts)
    
    def _generate_session_title(self, decisions: List[Dict], patterns: List[Dict]) -> str:
        """Generate descriptive session title"""
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Use primary pattern for title
        if patterns:
            primary_pattern = patterns[0]['pattern_type'].replace('_', ' ').title()
            return f"Governance Session: {primary_pattern} - {current_date}"
        
        # Use decision count
        if decisions:
            return f"Governance Session: {len(decisions)} Decisions - {current_date}"
        
        # Default title
        return f"Governance Session - {current_date}"
    
    def _create_conversation_summary(self, conversation: Dict[str, Any]) -> str:
        """Create comprehensive conversation summary"""
        
        queries = conversation.get('queries', [])
        query_count = len(queries)
        
        # Sample queries for summary
        sample_queries = queries[:3] if queries else []
        
        summary = f"""
GOVERNANCE SESSION SUMMARY - {datetime.now().strftime('%B %d, %Y')}

QUERIES PROCESSED: {query_count}
SAMPLE TOPICS: {'; '.join(sample_queries) if sample_queries else 'General governance inquiry'}

SESSION FOCUS: Institutional memory and governance intelligence
PROCESSING MODE: BoardContinuity AI enterprise analysis
INSTITUTIONAL CONTEXT: Private club governance and decision support
        """.strip()
        
        return summary
    
    def _calculate_session_metrics(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate session performance metrics"""
        
        queries = conversation.get('queries', [])
        content = self._get_conversation_text(conversation)
        
        return {
            'total_queries': len(queries),
            'content_length': len(content),
            'governance_density': content.lower().count('governance') + content.lower().count('board') + content.lower().count('committee'),
            'complexity_score': min(len(queries) * 0.2 + len(content) / 1000.0, 10.0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _store_session(self, session: GovernanceSession):
        """Store session in database"""
        
        try:
            session_data = {
                'id': session.session_id,
                'org_id': session.org_id,
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'last_updated': session.last_updated.isoformat(),
                'conversation_summary': session.conversation_summary,
                'decisions_analyzed': json.dumps(session.decisions_analyzed),
                'precedents_identified': json.dumps(session.precedents_identified),
                'patterns_discovered': json.dumps(session.patterns_discovered),
                'next_actions': json.dumps(session.next_actions),
                'key_insights': json.dumps(session.key_insights),
                'participants': json.dumps(session.participants),
                'session_metrics': json.dumps(session.session_metrics),
                'status': session.status
            }
            
            result = supa.table("governance_sessions").insert(session_data).execute()
            
            if result.data:
                logger.info(f"Stored governance session: {session.session_id}")
            else:
                logger.warning(f"Failed to store session: {session.session_id}")
                
        except Exception as e:
            logger.error(f"Error storing governance session: {e}")
    
    def get_session_history(self, org_id: str, limit: int = 10) -> List[GovernanceSession]:
        """Get recent session history for organization"""
        
        try:
            result = supa.table("governance_sessions").select("*").eq("org_id", org_id).order("created_at", desc=True).limit(limit).execute()
            
            sessions = []
            if result.data:
                for session_data in result.data:
                    # Parse JSON fields
                    session = GovernanceSession(
                        session_id=session_data['id'],
                        org_id=session_data['org_id'],
                        title=session_data['title'],
                        created_at=datetime.fromisoformat(session_data['created_at']),
                        last_updated=datetime.fromisoformat(session_data['last_updated']),
                        conversation_summary=session_data['conversation_summary'],
                        decisions_analyzed=json.loads(session_data['decisions_analyzed']),
                        precedents_identified=json.loads(session_data['precedents_identified']),
                        patterns_discovered=json.loads(session_data['patterns_discovered']),
                        next_actions=json.loads(session_data['next_actions']),
                        key_insights=json.loads(session_data['key_insights']),
                        participants=json.loads(session_data['participants']),
                        session_metrics=json.loads(session_data['session_metrics']),
                        status=session_data['status']
                    )
                    sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving session history: {e}")
            return []
    
    def update_session_with_new_interaction(self, session_id: str, new_conversation: Dict[str, Any]) -> bool:
        """Update existing session with new interaction"""
        
        try:
            if session_id not in self.current_sessions:
                # Try to load from database
                result = supa.table("governance_sessions").select("*").eq("id", session_id).execute()
                if not result.data:
                    return False
                # Would reconstruct session object here
            
            # Update session with new data
            session = self.current_sessions[session_id]
            
            # Extract new intelligence
            new_decisions = self.extract_decisions(new_conversation)
            new_precedents = self.find_precedents(new_conversation)
            new_patterns = self.identify_patterns(new_conversation)
            new_actions = self.determine_follow_ups(new_conversation)
            
            # Merge with existing data
            session.decisions_analyzed.extend(new_decisions)
            session.precedents_identified.extend(new_precedents)
            session.patterns_discovered.extend(new_patterns)
            session.next_actions.extend(new_actions)
            
            # Update timestamps
            session.last_updated = datetime.now()
            
            # Re-store in database
            self._store_session(session)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return False

def create_governance_session_manager() -> GovernanceSessionManager:
    """Factory function to create governance session manager"""
    return GovernanceSessionManager()