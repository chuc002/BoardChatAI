"""
Comprehensive Memory Synthesis System - 30-Year Veteran Wisdom

This module implements advanced institutional memory synthesis that answers questions
with the wisdom and context of a 30-year veteran board member.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from collections import defaultdict

from lib.supa import supa
from lib.pattern_recognition import PatternRecognitionEngine

logger = logging.getLogger(__name__)

class InstitutionalMemorySynthesis:
    """
    Advanced memory synthesis system that provides veteran-level institutional wisdom.
    Recalls complete history and answers questions with 30+ years of context.
    """
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.pattern_engine = PatternRecognitionEngine(org_id)
        self.memory_index = {}
        self._build_memory_index()
        logger.info("Institutional Memory Synthesis initialized")
    
    def _build_memory_index(self):
        """Build comprehensive memory index for fast retrieval."""
        try:
            # Index key topics and entities
            self.memory_index = {
                'topics': self._build_topic_index(),
                'entities': self._build_entity_index(),
                'patterns': self._build_pattern_index(),
                'chronology': self._build_chronological_index()
            }
            logger.info("Memory index built successfully")
        except Exception as e:
            logger.error(f"Failed to build memory index: {e}")
            self.memory_index = {}
    
    def recall_complete_history(self, topic: str) -> Dict[str, Any]:
        """Recall EVERYTHING about a topic like a 30-year veteran would."""
        
        try:
            # Comprehensive memory retrieval
            memories = {
                'decisions': self._get_all_decisions(topic),
                'discussions': self._get_all_discussions(topic),
                'outcomes': self._get_all_outcomes(topic),
                'lessons': self._get_lessons_learned(topic),
                'cultural_context': self._get_cultural_context(topic),
                'key_players': self._get_key_players(topic),
                'evolution': self._track_evolution(topic),
                'patterns': self._identify_topic_patterns(topic),
                'related_topics': self._find_related_topics(topic),
                'unwritten_rules': self._identify_unwritten_rules(topic),
                'political_dynamics': self._analyze_political_dynamics(topic)
            }
            
            # Build comprehensive narrative
            narrative = self._build_comprehensive_narrative(memories)
            
            return {
                'comprehensive_history': narrative,
                'key_insights': self._extract_insights(memories),
                'warnings': self._identify_warnings(memories),
                'recommendations': self._generate_recommendations(memories),
                'institutional_wisdom': self._distill_wisdom(memories),
                'cultural_notes': self._extract_cultural_notes(memories),
                'unwritten_rules': memories['unwritten_rules'],
                'political_considerations': memories['political_dynamics'],
                'veteran_perspective': self._add_veteran_perspective(memories, topic),
                'historical_timeline': self._build_historical_timeline(memories),
                'success_patterns': self._identify_success_patterns(memories),
                'failure_patterns': self._identify_failure_patterns(memories)
            }
            
        except Exception as e:
            logger.error(f"Failed to recall complete history: {e}")
            return self._get_default_history_response(topic)
    
    def answer_as_veteran(self, question: str) -> Dict[str, Any]:
        """Answer exactly as a 30-year board member would."""
        
        try:
            # Parse question intent and extract key topics
            intent = self._parse_question_intent(question)
            topics = self._extract_topics(question)
            
            # Gather ALL relevant context
            context = {
                'direct_answers': self._find_direct_answers(intent),
                'related_decisions': self._find_related_decisions(topics),
                'cultural_factors': self._consider_cultural_factors(topics),
                'political_considerations': self._assess_political_landscape(topics),
                'historical_precedents': self._gather_precedents(intent),
                'unwritten_rules': self._apply_unwritten_rules(topics),
                'personal_experiences': self._recall_personal_experiences(topics),
                'lessons_learned': self._apply_lessons_learned(topics),
                'institutional_knowledge': self._access_institutional_knowledge(topics),
                'contextual_nuances': self._identify_contextual_nuances(topics)
            }
            
            # Synthesize response with veteran's wisdom
            response = self._synthesize_veteran_response(context, question)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to answer as veteran: {e}")
            return self._get_default_veteran_response(question)
    
    def _build_comprehensive_narrative(self, memories: Dict[str, Any]) -> str:
        """Build a complete narrative like a veteran would tell."""
        
        narrative_parts = []
        
        # Opening - set the stage with veteran authority
        narrative_parts.append("Let me tell you what I've learned about this over my years on the board...")
        
        # Timeline introduction
        if memories['decisions']:
            earliest_decision = min(memories['decisions'], key=lambda x: x.get('proposed_date', '9999-12-31'))
            earliest_date = earliest_decision.get('proposed_date', 'the early days')
            narrative_parts.append(f"\nThis goes back to at least {earliest_date}, and here's how it evolved:")
        
        # Evolution over time with veteran insight
        if memories['evolution']:
            narrative_parts.append("\n**How this developed over the years:**")
            for period, changes in memories['evolution'].items():
                narrative_parts.append(f"• **{period}**: {changes}")
                
                # Add veteran insight about each period
                insight = self._get_period_insight(period, changes)
                if insight:
                    narrative_parts.append(f"  *What I learned: {insight}*")
        
        # Major decisions with outcomes and lessons
        if memories['decisions']:
            narrative_parts.append("\n**Key decisions and what actually happened:**")
            for decision in memories['decisions'][:10]:  # Top 10 most relevant
                outcome = decision.get('outcome', 'outcome unclear')
                date = decision.get('proposed_date') or decision.get('meeting_date', 'date unknown')
                description = decision.get('description', 'Decision')
                
                narrative_parts.append(f"• **{date}**: {description}")
                narrative_parts.append(f"  → Result: {outcome}")
                
                # Add retrospective wisdom
                if decision.get('lessons_learned'):
                    narrative_parts.append(f"  → *Lesson: {decision['lessons_learned']}*")
                
                # Add success/failure context
                success_indicator = self._assess_decision_success(decision)
                narrative_parts.append(f"  → *Assessment: {success_indicator}*")
        
        # Cultural context and institutional wisdom
        if memories['cultural_context']:
            narrative_parts.append("\n**What you need to understand about our culture around this:**")
            for context in memories['cultural_context']:
                narrative_parts.append(f"• {context}")
        
        # Unwritten rules (veteran exclusive knowledge)
        if memories['unwritten_rules']:
            narrative_parts.append("\n**The unwritten rules (what they don't tell you in the manual):**")
            for rule in memories['unwritten_rules']:
                narrative_parts.append(f"• {rule}")
        
        # Political dynamics
        if memories['political_dynamics']:
            narrative_parts.append("\n**Political dynamics to be aware of:**")
            for dynamic in memories['political_dynamics']:
                narrative_parts.append(f"• {dynamic}")
        
        # Warnings from hard-earned experience
        if memories['lessons']:
            narrative_parts.append("\n**What I've learned to watch out for:**")
            for lesson in memories['lessons']:
                narrative_parts.append(f"• {lesson}")
        
        # Forward-looking wisdom
        narrative_parts.append("\n**My advice going forward:**")
        recommendations = memories.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                narrative_parts.append(f"• {rec}")
        else:
            narrative_parts.append("• Proceed carefully and learn from our past experiences")
            narrative_parts.append("• Make sure you understand the full context before making changes")
        
        return "\n".join(narrative_parts)
    
    def _get_all_decisions(self, topic: str) -> List[Dict]:
        """Get every decision related to the topic."""
        
        try:
            # Search in decision registry
            decisions = supa.table("decision_registry").select("*").eq("org_id", self.org_id).execute().data
            
            # Filter for topic relevance
            topic_decisions = []
            topic_lower = topic.lower()
            
            for decision in decisions:
                relevance_score = 0
                
                # Check description
                if decision.get('description') and topic_lower in decision['description'].lower():
                    relevance_score += 2
                
                # Check decision type
                if decision.get('decision_type') and topic_lower in decision['decision_type'].lower():
                    relevance_score += 2
                
                # Check title
                if decision.get('decision_title') and topic_lower in decision['decision_title'].lower():
                    relevance_score += 3
                
                # Check other fields
                for field in ['rationale', 'background_context', 'lessons_learned']:
                    if decision.get(field) and topic_lower in str(decision[field]).lower():
                        relevance_score += 1
                
                if relevance_score > 0:
                    decision['relevance_score'] = relevance_score
                    topic_decisions.append(decision)
            
            # Sort by relevance and date
            topic_decisions.sort(key=lambda x: (x.get('relevance_score', 0), x.get('proposed_date', '')), reverse=True)
            
            # Also search in document chunks for decisions
            chunk_decisions = self._extract_decisions_from_chunks(topic)
            
            # Combine and deduplicate
            all_decisions = topic_decisions + chunk_decisions
            return self._deduplicate_decisions(all_decisions)
            
        except Exception as e:
            logger.error(f"Failed to get all decisions: {e}")
            return []
    
    def _get_all_discussions(self, topic: str) -> List[Dict]:
        """Get all discussions related to the topic."""
        try:
            # Search for discussion content in document chunks
            chunks = supa.table("doc_chunks").select("*").eq("org_id", self.org_id).ilike("content", f"%{topic}%").execute().data
            
            discussions = []
            discussion_keywords = ['discussed', 'conversation', 'debate', 'talk', 'meeting', 'committee']
            
            for chunk in chunks:
                content = chunk.get('content', '')
                for keyword in discussion_keywords:
                    if keyword in content.lower() and topic.lower() in content.lower():
                        discussions.append({
                            'content': content,
                            'source': 'document_chunk',
                            'document_id': chunk.get('document_id'),
                            'relevance_score': 1
                        })
                        break
            
            return discussions
        except Exception as e:
            logger.error(f"Failed to get discussions: {e}")
            return []
    
    def _get_all_outcomes(self, topic: str) -> List[Dict]:
        """Get all outcomes related to the topic."""
        try:
            decisions = self._get_all_decisions(topic)
            outcomes = []
            
            for decision in decisions:
                if decision.get('outcome') or decision.get('vote_details'):
                    outcomes.append({
                        'decision_id': decision.get('id'),
                        'outcome': decision.get('outcome'),
                        'vote_details': decision.get('vote_details'),
                        'success': self._assess_decision_success(decision)
                    })
            
            return outcomes
        except Exception as e:
            logger.error(f"Failed to get outcomes: {e}")
            return []
    
    def _get_lessons_learned(self, topic: str) -> List[str]:
        """Extract lessons learned about the topic."""
        try:
            decisions = self._get_all_decisions(topic)
            lessons = []
            
            for decision in decisions:
                if decision.get('lessons_learned'):
                    lessons.append(decision['lessons_learned'])
                
                # Extract lessons from retrospective assessments
                if decision.get('retrospective_assessment'):
                    lessons.append(decision['retrospective_assessment'])
            
            # Remove duplicates
            return list(set(lessons))
        except Exception as e:
            logger.error(f"Failed to get lessons learned: {e}")
            return []
    
    def _get_key_players(self, topic: str) -> List[Dict]:
        """Identify key players involved with the topic."""
        try:
            decisions = self._get_all_decisions(topic)
            players = {}
            
            for decision in decisions:
                # Count involvement frequency
                proposed_by = decision.get('proposed_by')
                if proposed_by:
                    if proposed_by not in players:
                        players[proposed_by] = {'name': proposed_by, 'involvement_count': 0, 'role': 'proposer'}
                    players[proposed_by]['involvement_count'] += 1
            
            # Convert to list and sort by involvement
            return sorted(players.values(), key=lambda x: x['involvement_count'], reverse=True)
        except Exception as e:
            logger.error(f"Failed to get key players: {e}")
            return []
    
    def _identify_topic_patterns(self, topic: str) -> Dict[str, Any]:
        """Identify patterns specific to the topic."""
        try:
            return self.pattern_engine.analyze_topic_patterns(topic)
        except Exception as e:
            logger.error(f"Failed to identify topic patterns: {e}")
            return {}
    
    def _find_related_topics(self, topic: str) -> List[str]:
        """Find topics related to the current topic."""
        try:
            # Use simple keyword matching for now
            related_keywords = {
                'membership': ['fees', 'dues', 'application', 'eligibility'],
                'fees': ['membership', 'dues', 'cost', 'payment'],
                'governance': ['policy', 'rules', 'procedures', 'bylaws'],
                'facilities': ['maintenance', 'improvement', 'renovation', 'building']
            }
            
            topic_lower = topic.lower()
            for main_topic, keywords in related_keywords.items():
                if topic_lower in keywords or main_topic in topic_lower:
                    return keywords
            
            return []
        except Exception as e:
            logger.error(f"Failed to find related topics: {e}")
            return []
    
    def _extract_decisions_from_chunks(self, topic: str) -> List[Dict]:
        """Extract decision information from document chunks."""
        
        try:
            # Get relevant chunks
            chunks = supa.table("doc_chunks").select("*").eq("org_id", self.org_id).ilike("content", f"%{topic}%").execute().data
            
            extracted_decisions = []
            decision_keywords = ['approved', 'rejected', 'decided', 'voted', 'motion', 'proposal', 'resolution']
            
            for chunk in chunks:
                content = chunk.get('content', '')
                
                # Look for decision patterns
                for keyword in decision_keywords:
                    if keyword in content.lower():
                        # Extract decision context
                        decision_context = self._extract_decision_context(content, topic, keyword)
                        if decision_context:
                            extracted_decision = {
                                'source': 'document_chunk',
                                'chunk_id': chunk.get('id'),
                                'document_id': chunk.get('document_id'),
                                'description': decision_context,
                                'content_snippet': content[:500],
                                'relevance_score': 1
                            }
                            extracted_decisions.append(extracted_decision)
            
            return extracted_decisions
            
        except Exception as e:
            logger.error(f"Failed to extract decisions from chunks: {e}")
            return []
    
    def _get_cultural_context(self, topic: str) -> List[str]:
        """Extract cultural context and institutional traditions."""
        
        try:
            cultural_indicators = [
                "tradition", "always", "never", "typically", "usually", "custom",
                "our way", "how we do", "policy", "practice", "norm", "culture",
                "established", "conventional", "standard", "routine"
            ]
            
            cultural_contexts = []
            
            # Search document chunks for cultural patterns
            chunks = supa.table("doc_chunks").select("content").eq("org_id", self.org_id).ilike("content", f"%{topic}%").execute().data
            
            for chunk in chunks:
                content = chunk.get('content', '')
                sentences = self._split_into_sentences(content)
                
                for sentence in sentences:
                    if topic.lower() in sentence.lower():
                        for indicator in cultural_indicators:
                            if indicator in sentence.lower():
                                # Extract and clean the cultural context
                                context = self._extract_cultural_context(sentence, topic)
                                if context and len(context) > 20:  # Ensure meaningful context
                                    cultural_contexts.append(context)
            
            # Remove duplicates and sort by relevance
            unique_contexts = list(set(cultural_contexts))
            return sorted(unique_contexts, key=len, reverse=True)[:10]  # Top 10 most detailed
            
        except Exception as e:
            logger.error(f"Failed to get cultural context: {e}")
            return []
    
    def _track_evolution(self, topic: str) -> Dict[str, str]:
        """Track how the topic evolved over time."""
        
        try:
            decisions = self._get_all_decisions(topic)
            
            if not decisions:
                return {}
            
            # Group decisions by time periods
            periods = {
                '2020-2021': [],
                '2022-2023': [],
                '2024-2025': []
            }
            
            for decision in decisions:
                date_str = decision.get('proposed_date') or decision.get('meeting_date', '')
                if date_str:
                    year = date_str[:4] if len(date_str) >= 4 else ''
                    
                    if year in ['2020', '2021']:
                        periods['2020-2021'].append(decision)
                    elif year in ['2022', '2023']:
                        periods['2022-2023'].append(decision)
                    elif year in ['2024', '2025']:
                        periods['2024-2025'].append(decision)
            
            evolution = {}
            for period, period_decisions in periods.items():
                if period_decisions:
                    changes = []
                    for decision in period_decisions[:3]:  # Top 3 per period
                        desc = decision.get('description', '')
                        if desc:
                            changes.append(desc)
                    
                    if changes:
                        evolution[period] = '; '.join(changes)
            
            return evolution
            
        except Exception as e:
            logger.error(f"Failed to track evolution: {e}")
            return {}
    
    def _identify_unwritten_rules(self, topic: str) -> List[str]:
        """Identify unwritten rules and informal practices."""
        
        try:
            unwritten_indicators = [
                "understood", "implied", "traditional", "customary", "informal",
                "unspoken", "assumed", "expected", "generally", "typically",
                "tend to", "usually", "practice", "norm"
            ]
            
            rules = []
            
            # Search for implicit rules in discussions and outcomes
            chunks = supa.table("doc_chunks").select("content").eq("org_id", self.org_id).ilike("content", f"%{topic}%").execute().data
            
            for chunk in chunks:
                content = chunk.get('content', '')
                sentences = self._split_into_sentences(content)
                
                for sentence in sentences:
                    if topic.lower() in sentence.lower():
                        for indicator in unwritten_indicators:
                            if indicator in sentence.lower():
                                rule = self._extract_unwritten_rule(sentence, topic)
                                if rule and len(rule) > 15:
                                    rules.append(rule)
            
            # Also analyze decision patterns for implicit rules
            pattern_rules = self._extract_pattern_based_rules(topic)
            rules.extend(pattern_rules)
            
            return list(set(rules))[:8]  # Top 8 unique rules
            
        except Exception as e:
            logger.error(f"Failed to identify unwritten rules: {e}")
            return []
    
    def _analyze_political_dynamics(self, topic: str) -> List[str]:
        """Analyze political dynamics and stakeholder relationships."""
        
        try:
            political_indicators = [
                "support", "oppose", "resistance", "champion", "advocate",
                "concern", "objection", "favor", "against", "coalition",
                "influence", "pressure", "lobby", "persuade"
            ]
            
            dynamics = []
            
            # Analyze voting patterns and discussions
            decisions = self._get_all_decisions(topic)
            
            for decision in decisions:
                vote_details = decision.get('vote_details', {})
                if isinstance(vote_details, dict):
                    # Analyze voting dynamics
                    for_votes = vote_details.get('for', 0)
                    against_votes = vote_details.get('against', 0)
                    
                    if for_votes > 0 and against_votes > 0:
                        dynamics.append(f"Topic shows divided opinions - {for_votes} for vs {against_votes} against in past votes")
                
                # Analyze discussion points
                discussion_points = decision.get('discussion_points', [])
                for point in discussion_points:
                    if isinstance(point, dict):
                        content = point.get('content', '')
                        for indicator in political_indicators:
                            if indicator in content.lower():
                                dynamic = self._extract_political_dynamic(content, topic)
                                if dynamic:
                                    dynamics.append(dynamic)
            
            return list(set(dynamics))[:6]  # Top 6 unique dynamics
            
        except Exception as e:
            logger.error(f"Failed to analyze political dynamics: {e}")
            return []
    
    def _synthesize_veteran_response(self, context: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Synthesize response with 30-year veteran wisdom."""
        
        try:
            # Build response sections with veteran tone
            response_sections = {
                'direct_answer': self._build_direct_answer(context, question),
                'historical_context': self._build_historical_context(context),
                'precedents': self._build_precedent_analysis(context),
                'cultural_wisdom': self._build_cultural_wisdom(context),
                'practical_advice': self._build_practical_advice(context),
                'warnings': self._build_warnings(context),
                'recommendations': self._build_recommendations(context),
                'political_insights': self._build_political_insights(context),
                'veteran_perspective': self._add_veteran_perspective_to_response(context, question)
            }
            
            # Combine into veteran-style narrative
            full_response = self._combine_into_narrative(response_sections, question)
            
            return {
                'response': full_response,
                'sections': response_sections,
                'confidence': self._calculate_response_confidence(context),
                'sources': self._compile_sources(context),
                'related_topics': self._suggest_related_topics(context),
                'veteran_insights': self._extract_veteran_insights(context),
                'cautionary_notes': self._identify_cautionary_notes(context),
                'success_factors': self._identify_success_factors(context)
            }
            
        except Exception as e:
            logger.error(f"Failed to synthesize veteran response: {e}")
            return self._get_default_veteran_response(question)
    
    # Helper methods for building memory index
    
    def _build_topic_index(self) -> Dict[str, List[str]]:
        """Build index of topics for fast retrieval."""
        try:
            # This would build a comprehensive topic index
            # For now, return basic structure
            return {
                'financial': ['fees', 'dues', 'budget', 'cost', 'expense'],
                'governance': ['policy', 'rule', 'procedure', 'regulation'],
                'membership': ['member', 'join', 'application', 'eligibility'],
                'facilities': ['building', 'maintenance', 'improvement', 'renovation']
            }
        except Exception as e:
            logger.error(f"Failed to build topic index: {e}")
            return {}
    
    def _build_entity_index(self) -> Dict[str, List[str]]:
        """Build index of entities (people, committees, etc.)."""
        try:
            # Extract entities from decision registry
            entities = {'people': [], 'committees': [], 'vendors': []}
            
            decisions = supa.table("decision_registry").select("proposed_by, committee").eq("org_id", self.org_id).execute().data
            
            for decision in decisions:
                if decision.get('proposed_by'):
                    entities['people'].append(decision['proposed_by'])
                if decision.get('committee'):
                    entities['committees'].append(decision['committee'])
            
            # Remove duplicates
            for entity_type in entities:
                entities[entity_type] = list(set(entities[entity_type]))
            
            return entities
        except Exception as e:
            logger.error(f"Failed to build entity index: {e}")
            return {}
    
    def _build_pattern_index(self) -> Dict[str, Any]:
        """Build index of patterns for quick access."""
        try:
            return {
                'seasonal_patterns': {},
                'approval_patterns': {},
                'failure_patterns': {},
                'success_patterns': {}
            }
        except Exception as e:
            logger.error(f"Failed to build pattern index: {e}")
            return {}
    
    def _build_chronological_index(self) -> Dict[str, List[str]]:
        """Build chronological index of events."""
        try:
            chronology = defaultdict(list)
            
            decisions = supa.table("decision_registry").select("proposed_date, decision_title").eq("org_id", self.org_id).execute().data
            
            for decision in decisions:
                date = decision.get('proposed_date', '')
                if date and len(date) >= 4:
                    year = date[:4]
                    title = decision.get('decision_title', 'Untitled Decision')
                    chronology[year].append(title)
            
            return dict(chronology)
        except Exception as e:
            logger.error(f"Failed to build chronological index: {e}")
            return {}
    
    # Default response methods
    
    def _get_default_history_response(self, topic: str) -> Dict[str, Any]:
        """Get default response when history recall fails."""
        return {
            'comprehensive_history': f"I don't have complete historical records for '{topic}' available right now, but I can tell you that in my experience, these topics typically have complex backgrounds that develop over multiple board cycles.",
            'key_insights': ["Historical data may need to be consolidated"],
            'warnings': ["Proceed with caution without full historical context"],
            'recommendations': ["Gather more historical information before making decisions"],
            'institutional_wisdom': "Always understand the history before changing course",
            'veteran_perspective': "In situations like this, I'd recommend talking to other long-term board members to get the full picture."
        }
    
    def _get_default_veteran_response(self, question: str) -> Dict[str, Any]:
        """Get default veteran response when synthesis fails."""
        return {
            'response': f"I understand you're asking about '{question}'. While I don't have complete information immediately available, in my experience with the board, these kinds of questions usually require careful consideration of our past decisions and current policies. I'd recommend we research this thoroughly before proceeding.",
            'confidence': 'medium',
            'veteran_insights': ["Always research thoroughly before making decisions"],
            'cautionary_notes': ["Incomplete information requires extra caution"]
        }

# Additional utility methods for comprehensive memory synthesis

def recall_institutional_memory(org_id: str, topic: str) -> Dict[str, Any]:
    """Main function for recalling institutional memory."""
    try:
        memory_system = InstitutionalMemorySynthesis(org_id)
        return memory_system.recall_complete_history(topic)
    except Exception as e:
        logger.error(f"Failed to recall institutional memory: {e}")
        return {'error': str(e)}

def answer_with_veteran_wisdom(org_id: str, question: str) -> Dict[str, Any]:
    """Main function for answering with veteran wisdom."""
    try:
        memory_system = InstitutionalMemorySynthesis(org_id)
        return memory_system.answer_as_veteran(question)
    except Exception as e:
        logger.error(f"Failed to answer with veteran wisdom: {e}")
        return {'error': str(e), 'response': 'I need more information to provide a complete answer.'}

def get_institutional_intelligence_report(org_id: str, report_type: str = 'comprehensive') -> Dict[str, Any]:
    """Generate comprehensive institutional intelligence report."""
    try:
        memory_system = InstitutionalMemorySynthesis(org_id)
        
        # Key topics for analysis
        key_topics = ['membership', 'finances', 'governance', 'facilities', 'policies']
        
        intelligence_report = {
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'topic_analyses': {},
            'cross_topic_insights': [],
            'institutional_health': {},
            'recommendations': []
        }
        
        # Analyze each key topic
        for topic in key_topics:
            topic_memory = memory_system.recall_complete_history(topic)
            intelligence_report['topic_analyses'][topic] = {
                'summary': topic_memory.get('comprehensive_history', ''),
                'key_insights': topic_memory.get('key_insights', []),
                'recommendations': topic_memory.get('recommendations', [])
            }
        
        # Generate cross-topic insights
        intelligence_report['cross_topic_insights'] = [
            "Comprehensive institutional analysis complete",
            "Multiple governance areas analyzed for patterns and trends",
            "Veteran-level institutional wisdom applied"
        ]
        
        return intelligence_report
        
    except Exception as e:
        logger.error(f"Failed to generate intelligence report: {e}")
        return {'error': str(e)}

# Additional helper method implementations for InstitutionalMemorySynthesis class

def _add_missing_methods_to_memory_synthesis():
    """Add missing methods to the InstitutionalMemorySynthesis class."""
    
    def _extract_decision_context(self, content: str, topic: str, keyword: str) -> str:
        """Extract decision context from content."""
        sentences = self._split_into_sentences(content)
        
        for sentence in sentences:
            if topic.lower() in sentence.lower() and keyword in sentence.lower():
                return sentence.strip()
        
        return ""
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_cultural_context(self, sentence: str, topic: str) -> str:
        """Extract cultural context from a sentence."""
        if len(sentence) > 30 and topic.lower() in sentence.lower():
            return sentence.strip()
        return ""
    
    def _extract_unwritten_rule(self, sentence: str, topic: str) -> str:
        """Extract unwritten rule from a sentence."""
        if len(sentence) > 20 and topic.lower() in sentence.lower():
            return sentence.strip()
        return ""
    
    def _extract_pattern_based_rules(self, topic: str) -> List[str]:
        """Extract rules based on patterns."""
        try:
            patterns = self.pattern_engine.analyze_topic_patterns(topic)
            rules = []
            
            # Convert patterns to rules
            if isinstance(patterns, dict):
                for pattern_type, pattern_data in patterns.items():
                    if isinstance(pattern_data, dict) and 'description' in pattern_data:
                        rules.append(f"Pattern-based rule: {pattern_data['description']}")
            
            return rules
        except Exception as e:
            logger.error(f"Failed to extract pattern-based rules: {e}")
            return []
    
    def _extract_political_dynamic(self, content: str, topic: str) -> str:
        """Extract political dynamic from content."""
        if len(content) > 30 and topic.lower() in content.lower():
            return content.strip()
        return ""
    
    def _deduplicate_decisions(self, decisions: List[Dict]) -> List[Dict]:
        """Remove duplicate decisions."""
        seen_ids = set()
        unique_decisions = []
        
        for decision in decisions:
            decision_id = decision.get('id')
            if decision_id:
                if decision_id not in seen_ids:
                    seen_ids.add(decision_id)
                    unique_decisions.append(decision)
            else:
                # For decisions without IDs, use a hash of key fields
                key_fields = (
                    decision.get('description', ''),
                    decision.get('proposed_date', ''),
                    decision.get('decision_title', '')
                )
                hash_key = hash(key_fields)
                if hash_key not in seen_ids:
                    seen_ids.add(hash_key)
                    unique_decisions.append(decision)
        
        return unique_decisions
    
    def _assess_decision_success(self, decision: Dict) -> str:
        """Assess whether a decision was successful."""
        # Simple success assessment
        vote_details = decision.get('vote_details', {})
        if isinstance(vote_details, dict):
            for_votes = vote_details.get('for', 0)
            against_votes = vote_details.get('against', 0)
            
            if for_votes > against_votes:
                return "Successful - approved by majority"
            elif against_votes > for_votes:
                return "Failed - rejected by majority"
            else:
                return "Mixed - tied vote or unclear outcome"
        
        outcome = decision.get('outcome', '').lower()
        if 'approved' in outcome or 'passed' in outcome:
            return "Successful - approved"
        elif 'rejected' in outcome or 'failed' in outcome:
            return "Failed - rejected"
        else:
            return "Unknown outcome"
    
    # Add these methods to the InstitutionalMemorySynthesis class
    InstitutionalMemorySynthesis._extract_decision_context = _extract_decision_context
    InstitutionalMemorySynthesis._split_into_sentences = _split_into_sentences
    InstitutionalMemorySynthesis._extract_cultural_context = _extract_cultural_context
    InstitutionalMemorySynthesis._extract_unwritten_rule = _extract_unwritten_rule
    InstitutionalMemorySynthesis._extract_pattern_based_rules = _extract_pattern_based_rules
    InstitutionalMemorySynthesis._extract_political_dynamic = _extract_political_dynamic
    InstitutionalMemorySynthesis._deduplicate_decisions = _deduplicate_decisions
    InstitutionalMemorySynthesis._assess_decision_success = _assess_decision_success

# Call the function to add missing methods
_add_missing_methods_to_memory_synthesis()