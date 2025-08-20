"""
Committee Agents System for BoardContinuity AI
Provides specialized committee expertise with targeted insights for specific governance areas.
"""

from typing import Dict, List, Any, Optional
import os
from openai import OpenAI
import logging

# Initialize logging
logger = logging.getLogger(__name__)

class CommitteeAgent:
    def __init__(self, committee_name: str, specialization: str, client: OpenAI):
        self.committee_name = committee_name
        self.specialization = specialization
        self.client = client
        self.instructions = self._build_instructions()
    
    def _build_instructions(self) -> str:
        return f"""
        You are the {self.committee_name} Committee specialist within BoardContinuity AI - a veteran board member with deep expertise in {self.committee_name.lower()} governance.
        
        SPECIALIZATION: {self.specialization}
        
        YOUR COMMITTEE EXPERTISE:
        - Provide deep institutional knowledge on {self.committee_name.lower()} matters with historical context
        - Reference specific {self.committee_name.lower()} decisions, vote patterns, and outcomes from past years
        - Understand {self.committee_name.lower()} budget cycles, approval processes, and seasonal patterns
        - Know {self.committee_name.lower()} vendor relationships, performance history, and contract patterns
        - Recognize {self.committee_name.lower()}-specific risks and success factors from institutional experience
        
        VETERAN COMMITTEE MEMBER RESPONSE STYLE:
        - Begin with "As your {self.committee_name} Committee veteran..." or "In my years chairing {self.committee_name}..."
        - Focus specifically on {self.committee_name.lower()}-related aspects with exact details
        - Reference committee meeting patterns, decision timelines, and historical outcomes
        - Provide {self.committee_name.lower()}-specific implementation guidance with precedent warnings
        - Cite {self.committee_name.lower()} precedents with dates, amounts, and success/failure rates
        - Warn about {self.committee_name.lower()}-specific pitfalls based on historical experience
        
        RESPONSE FORMAT:
        ### Committee Historical Context
        [Specific {self.committee_name.lower()} decisions with years, amounts, outcomes]
        
        ### {self.committee_name} Committee Wisdom
        [Committee-specific lessons learned and precedent warnings]
        
        ### Implementation Guidance
        [Step-by-step {self.committee_name.lower()} committee recommendations]
        
        Always ground advice in specific {self.committee_name} Committee institutional experience.
        """
    
    def process_query(self, query: str, context: str) -> Dict[str, Any]:
        """Process committee-specific query with veteran expertise"""
        
        full_prompt = f"""
        {self.instructions}
        
        INSTITUTIONAL CONTEXT: {context}
        
        COMMITTEE QUERY: {query}
        
        Provide your {self.committee_name} Committee veteran perspective on this matter, including specific historical examples and institutional wisdom.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            
            return {
                'committee': self.committee_name,
                'response': response.choices[0].message.content.strip(),
                'specialization': self.specialization,
                'expertise_level': 'veteran'
            }
            
        except Exception as e:
            logger.error(f"Committee {self.committee_name} query processing failed: {e}")
            return {
                'committee': self.committee_name,
                'response': f"I apologize, but I cannot access my {self.committee_name} Committee expertise at this moment. Please try again.",
                'specialization': self.specialization,
                'error': str(e)
            }

class CommitteeManager:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.committees = {
            'golf': CommitteeAgent(
                'Golf',
                'Course maintenance, pro shop operations, golf tournaments, member golf services, equipment management, course improvements',
                self.client
            ),
            'finance': CommitteeAgent(
                'Finance',
                'Budget planning, financial oversight, dues structure, investment decisions, audit oversight, cost control measures',
                self.client
            ),
            'food_beverage': CommitteeAgent(
                'Food & Beverage',
                'Dining operations, catering services, bar management, food quality standards, vendor relationships, event planning',
                self.client
            ),
            'house': CommitteeAgent(
                'House',
                'Facilities maintenance, building renovations, member amenities, safety protocols, infrastructure upgrades, space management',
                self.client
            ),
            'membership': CommitteeAgent(
                'Membership',
                'Member recruitment, retention strategies, orientation programs, membership categories, social events, member satisfaction',
                self.client
            ),
            'grounds': CommitteeAgent(
                'Grounds',
                'Landscaping, irrigation systems, outdoor facility maintenance, seasonal preparations, environmental sustainability',
                self.client
            ),
            'strategic': CommitteeAgent(
                'Strategic Planning',
                'Long-term planning, capital projects, competitive analysis, member surveys, governance improvements, vision setting',
                self.client
            )
        }
    
    def route_query(self, query: str, context: str) -> Dict[str, Any]:
        """Intelligently route queries to appropriate committee specialists"""
        
        routing_prompt = f"""
        Analyze this board governance query and determine which committee(s) should provide specialized input.
        
        AVAILABLE COMMITTEE SPECIALISTS:
        - golf: Course operations, tournaments, golf services, equipment
        - finance: Budget, dues, investments, financial oversight, audits
        - food_beverage: Dining, catering, bar operations, events, vendor management
        - house: Facilities, renovations, maintenance, amenities, safety
        - membership: Recruitment, retention, orientation, social events, satisfaction
        - grounds: Landscaping, irrigation, outdoor facilities, environmental
        - strategic: Long-term planning, capital projects, governance, vision
        
        QUERY CONTEXT: {context[:500]}...
        
        GOVERNANCE QUERY: {query}
        
        Respond with:
        1. Committee names separated by commas if specialized input needed
        2. "general" if it requires broad board perspective without specific committee expertise
        
        Examples: "finance", "golf,grounds", "membership,food_beverage", "house,strategic", "general"
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": routing_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            routing_result = response.choices[0].message.content.strip().lower()
            
            if routing_result == "general":
                return {
                    'committees': [], 
                    'route': 'general',
                    'reasoning': 'Requires broad board perspective rather than specific committee expertise'
                }
            
            # Parse and validate committee names
            committee_names = [name.strip() for name in routing_result.split(',')]
            valid_committees = [name for name in committee_names if name in self.committees]
            invalid_committees = [name for name in committee_names if name not in self.committees]
            
            if invalid_committees:
                logger.warning(f"Invalid committee routing: {invalid_committees}")
            
            return {
                'committees': valid_committees,
                'route': 'committee_specific' if valid_committees else 'general',
                'routing_decision': routing_result,
                'valid_committees': valid_committees,
                'invalid_committees': invalid_committees
            }
            
        except Exception as e:
            logger.error(f"Committee routing failed: {e}")
            return {
                'committees': [], 
                'route': 'general', 
                'error': str(e),
                'fallback': True
            }
    
    def get_committee_perspectives(self, query: str, context: str, committees: List[str]) -> List[Dict[str, Any]]:
        """Get veteran perspectives from specific committee specialists"""
        
        perspectives = []
        
        for committee_name in committees:
            if committee_name in self.committees:
                logger.info(f"Getting {committee_name} committee perspective")
                perspective = self.committees[committee_name].process_query(query, context)
                perspectives.append(perspective)
            else:
                logger.warning(f"Unknown committee requested: {committee_name}")
                perspectives.append({
                    'committee': committee_name,
                    'response': f"Committee specialist '{committee_name}' is not available.",
                    'specialization': 'unknown',
                    'error': 'Committee not found'
                })
        
        return perspectives
    
    def synthesize_committee_insights(self, perspectives: List[Dict[str, Any]], original_query: str) -> str:
        """Synthesize multiple committee perspectives into unified response"""
        
        if not perspectives:
            return ""
        
        if len(perspectives) == 1:
            return perspectives[0]['response']
        
        synthesis_prompt = f"""
        You are BoardContinuity AI synthesizing insights from multiple committee specialists.
        
        ORIGINAL GOVERNANCE QUERY: {original_query}
        
        COMMITTEE SPECIALIST PERSPECTIVES:
        """
        
        for i, perspective in enumerate(perspectives, 1):
            committee = perspective.get('committee', f'Committee {i}')
            response = perspective.get('response', 'No response available')
            synthesis_prompt += f"""
        
        {i}. {committee.upper()} COMMITTEE PERSPECTIVE:
        {response}
        """
        
        synthesis_prompt += f"""
        
        SYNTHESIS TASK:
        Combine these committee perspectives into a unified BoardContinuity AI response that:
        - Integrates all relevant committee insights
        - Maintains the veteran board member voice
        - Identifies areas where committees agree or disagree
        - Provides comprehensive implementation guidance
        - Highlights cross-committee coordination needs
        - Uses the standard BoardContinuity format with Historical Context, Practical Wisdom, etc.
        
        Present a single, authoritative response that reflects the collective institutional wisdom of all relevant committees.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2,
                max_tokens=1800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Committee synthesis failed: {e}")
            # Fallback: concatenate committee responses
            combined = f"Based on input from {len(perspectives)} committee specialists:\n\n"
            for perspective in perspectives:
                committee = perspective.get('committee', 'Committee')
                response = perspective.get('response', '')
                combined += f"**{committee} Committee Perspective:**\n{response}\n\n"
            return combined
    
    def get_committee_enhanced_response(self, query: str, context: str) -> Dict[str, Any]:
        """Get committee-enhanced response for governance queries"""
        
        # Route the query to appropriate committees
        routing = self.route_query(query, context)
        
        if routing['route'] == 'general':
            return {
                'enhanced': False,
                'route': 'general',
                'reasoning': routing.get('reasoning', 'General board perspective sufficient')
            }
        
        # Get committee perspectives
        committees = routing.get('committees', [])
        if not committees:
            return {
                'enhanced': False,
                'route': 'general',
                'reason': 'No valid committees identified'
            }
        
        perspectives = self.get_committee_perspectives(query, context, committees)
        
        # Synthesize if multiple committees
        if len(perspectives) > 1:
            synthesized_response = self.synthesize_committee_insights(perspectives, query)
        else:
            synthesized_response = perspectives[0]['response'] if perspectives else ""
        
        return {
            'enhanced': True,
            'route': 'committee_enhanced',
            'committees_consulted': committees,
            'committee_count': len(perspectives),
            'response': synthesized_response,
            'perspectives': perspectives,
            'routing_info': routing
        }
    
    def get_available_committees(self) -> Dict[str, str]:
        """Get list of available committee specialists"""
        return {
            name: agent.specialization 
            for name, agent in self.committees.items()
        }