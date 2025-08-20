"""
Enterprise RAG Agent for BoardContinuity AI
Advanced multi-agent system integrating guardrails, committee expertise, and veteran intelligence
"""

from typing import Dict, List, Any, Optional
import os
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# BoardContinuity Agent Instructions
BOARDCONTINUITY_AGENT_INSTRUCTIONS = """
You are BoardContinuity AI - the digital embodiment of a 30-year veteran board member with perfect institutional memory.

CORE IDENTITY:
- You have witnessed every decision, vote, and discussion in this organization's history
- You understand the cultural context, unwritten rules, and governance patterns
- You provide wisdom that prevents expensive mistakes and accelerates decision-making

SPECIFIC ROUTINES:

1. PRECEDENT ANALYSIS ROUTINE:
   When asked about decisions or proposals:
   a) Search for similar historical decisions with exact details
   b) Reference specific dates, amounts, vote counts, and outcomes
   c) Explain the reasoning behind past decisions
   d) Warn if current approach deviates from successful patterns

2. OUTCOME PREDICTION ROUTINE:
   When evaluating proposals:
   a) Cite historical success/failure rates for similar decisions
   b) Provide timeline predictions based on past experience
   c) Identify risk factors that led to problems historically
   d) Suggest optimizations based on what worked before

3. NEW MEMBER ONBOARDING ROUTINE:
   When orienting new board members:
   a) Explain governance culture and decision-making patterns
   b) Share institutional wisdom about "how we do things here"
   c) Provide context about key relationships and dynamics
   d) Outline unwritten rules and expectations

4. BUDGET/FINANCIAL ROUTINE:
   When discussing financial matters:
   a) Reference historical spending patterns and outcomes
   b) Warn about timing factors (seasonal impacts, etc.)
   c) Cite committee approval patterns and requirements
   d) Predict cost variance based on similar projects

RESPONSE STRUCTURE:
Always organize responses with these sections:
- Historical Context (specific examples with dates/amounts)
- Practical Wisdom (lessons learned from experience)
- Outcome Predictions (success rates and timelines)
- Implementation Guidance (step-by-step recommendations)

EDGE CASE HANDLING:
- If insufficient historical data: State "In my experience, we haven't faced this exact situation before, but based on similar decisions..."
- If conflicting precedents: Explain both approaches and provide context for when each worked
- If outside governance scope: "This falls outside board governance. You might want to consult [specific expertise]"
- If confidential information requested: "I maintain confidentiality of sensitive board discussions"

LANGUAGE PATTERNS:
- Use phrases like "In my experience...", "We tried this before in [year]...", "Based on [X] similar decisions..."
- Reference specific board members when appropriate: "When Sarah Thompson chaired Finance in 2015..."
- Provide exact details: "The 2019 renovation went 23% over budget, taking 4 months instead of 2"

QUALITY STANDARDS:
- Never provide generic advice - always ground in specific institutional experience
- Include exact financial figures, dates, and vote counts when available
- Warn about deviations from successful patterns
- Predict outcomes with historical confidence levels

Use simple numbered citations [1], [2], [3] for readability.
"""

class EnterpriseRAGAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize guardrails
        try:
            from lib.enterprise_guardrails import BoardContinuityGuardrails
            self.guardrails = BoardContinuityGuardrails()
            self.guardrails_enabled = True
            logger.info("Enterprise guardrails initialized for RAG agent")
        except Exception as e:
            logger.warning(f"Guardrails initialization failed: {e}")
            self.guardrails = None
            self.guardrails_enabled = False
        
        # Initialize committee agents
        try:
            from lib.committee_agents import CommitteeManager
            self.committee_manager = CommitteeManager()
            self.committee_agents_enabled = True
            logger.info("Committee agents initialized for RAG agent")
        except Exception as e:
            logger.warning(f"Committee agents initialization failed: {e}")
            self.committee_manager = None
            self.committee_agents_enabled = False
        
        # Initialize human intervention system
        try:
            from lib.human_intervention import create_human_intervention_manager
            self.intervention_manager = create_human_intervention_manager()
            self.intervention_enabled = True
            logger.info("Human intervention system initialized for RAG agent")
        except Exception as e:
            logger.warning(f"Human intervention initialization failed: {e}")
            self.intervention_manager = None
            self.intervention_enabled = False
        
        # Initialize performance monitoring
        try:
            from lib.agent_monitoring import create_agent_performance_monitor
            self.performance_monitor = create_agent_performance_monitor()
            self.monitoring_enabled = True
            logger.info("Performance monitoring initialized for RAG agent")
        except Exception as e:
            logger.warning(f"Performance monitoring initialization failed: {e}")
            self.performance_monitor = None
            self.monitoring_enabled = False
        
        self.instructions = BOARDCONTINUITY_AGENT_INSTRUCTIONS
        
        # Tools available to the agent
        self.tools = {
            'document_search': self._search_documents,
            'pattern_analysis': self._analyze_patterns,
            'precedent_lookup': self._lookup_precedents,
            'committee_consultation': self._consult_committees,
            'outcome_prediction': self._predict_outcomes
        }
    
    def run(self, org_id: str, query: str, context: str = "", sources: List[Dict] = None) -> Dict[str, Any]:
        """Main enterprise agent execution loop with enhanced capabilities"""
        
        start_time = time.time()
        
        # Input guardrails validation
        if self.guardrails_enabled and self.guardrails:
            try:
                input_checks = self.guardrails.check_input(query)
                if not self._passes_guardrails(input_checks):
                    return self._create_guardrail_response(input_checks)
            except Exception as e:
                logger.warning(f"Input guardrails check failed: {e}")
        
        # Intelligent committee routing
        routing = {'route': 'general', 'committees': []}
        if self.committee_agents_enabled and self.committee_manager:
            try:
                routing = self.committee_manager.route_query(query, context)
                logger.info(f"Query routed to: {routing.get('committees', ['general'])}")
            except Exception as e:
                logger.warning(f"Committee routing failed: {e}")
        
        # Execute reasoning based on routing decision
        if routing['route'] == 'committee_specific' and routing.get('committees'):
            response = self._execute_committee_consultation(org_id, query, context, routing['committees'], sources)
        else:
            response = self._execute_single_agent(org_id, query, context, sources)
        
        # Output guardrails validation
        if self.guardrails_enabled and self.guardrails:
            try:
                output_checks = self.guardrails.check_output(response.get('response', ''))
                if not self._passes_guardrails(output_checks):
                    response['response'] = "I need to refine my response to maintain institutional confidentiality standards. Please rephrase your question or contact board administration directly."
                    response['guardrail_flags'] = output_checks
                    response['guardrails_passed'] = False
                else:
                    response['guardrails_passed'] = True
            except Exception as e:
                logger.warning(f"Output guardrails check failed: {e}")
                response['guardrails_passed'] = False
        
        # Human intervention check
        if self.intervention_enabled and self.intervention_manager:
            try:
                intervention_trigger = self.intervention_manager.should_intervene(query, response)
                if intervention_trigger:
                    logger.info(f"Human intervention triggered: {intervention_trigger.value}")
                    intervention_response = self.intervention_manager.create_intervention_response(
                        intervention_trigger, query, response
                    )
                    # Return intervention response instead of AI response
                    intervention_response['original_ai_response'] = response
                    intervention_response['performance'] = {
                        'response_time_ms': int((time.time() - start_time) * 1000),
                        'intervention_triggered': True,
                        'trigger_type': intervention_trigger.value
                    }
                    return intervention_response
            except Exception as e:
                logger.warning(f"Human intervention check failed: {e}")
        
        # Add performance metadata
        response['performance'] = {
            'response_time_ms': int((time.time() - start_time) * 1000),
            'guardrails_enabled': self.guardrails_enabled,
            'committee_agents_enabled': self.committee_agents_enabled,
            'intervention_enabled': self.intervention_enabled,
            'monitoring_enabled': self.monitoring_enabled,
            'routing_decision': routing,
            'agent_type': 'committee_consultation' if routing.get('route') == 'committee_specific' else 'single_agent'
        }
        
        # Log interaction for performance monitoring
        if self.monitoring_enabled and self.performance_monitor:
            try:
                self.performance_monitor.log_interaction({
                    'response_time_ms': response['performance']['response_time_ms'],
                    'confidence': response.get('confidence', 0),
                    'guardrails_passed': response.get('guardrails_passed', True),
                    'agent_type': response['performance']['agent_type'],
                    'intervention_triggered': response.get('intervention_triggered', False),
                    'trigger_type': response.get('trigger_type', None)
                })
            except Exception as e:
                logger.warning(f"Performance monitoring logging failed: {e}")
        
        return response
    
    def _passes_guardrails(self, checks: Dict[str, Any]) -> bool:
        """Check if all guardrails pass"""
        if not checks:
            return True
        return all(getattr(check, 'passed', True) for check in checks.values())
    
    def _create_guardrail_response(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Create response when guardrails fail"""
        failed_checks = [name for name, check in checks.items() if not getattr(check, 'passed', True)]
        
        return {
            'response': "I apologize, but I cannot process this request as it doesn't align with board governance topics. Please ask questions related to institutional decisions, policies, or governance matters.",
            'guardrail_flags': checks,
            'failed_checks': failed_checks,
            'confidence': 0.0,
            'strategy': 'guardrail_blocked',
            'sources': []
        }
    
    def _execute_committee_consultation(self, org_id: str, query: str, context: str, committees: List[str], sources: List[Dict] = None) -> Dict[str, Any]:
        """Execute multi-agent committee consultation with document intelligence"""
        
        try:
            # Get committee perspectives
            perspectives = self.committee_manager.get_committee_perspectives(query, context, committees)
            
            # Build enhanced context with document sources
            enhanced_context = context
            if sources:
                enhanced_context += "\n\nDOCUMENT SOURCES:\n"
                for i, source in enumerate(sources[:5], 1):
                    enhanced_context += f"[{i}] {source.get('title', 'Document')} (Page {source.get('page', '?')})\n"
            
            # Synthesize committee input with general governance knowledge
            synthesis_prompt = f"""
            {self.instructions}
            
            GOVERNANCE CONTEXT: {enhanced_context}
            
            COMMITTEE PERSPECTIVES:
            {self._format_committee_perspectives(perspectives)}
            
            USER QUERY: {query}
            
            Synthesize the committee perspectives with your institutional memory and document sources to provide a comprehensive response.
            Include specific committee insights while maintaining your 30-year veteran perspective.
            Use the standard BoardContinuity format with Historical Context, Practical Wisdom, Outcome Predictions, and Implementation Guidance.
            Reference documents using simple numbered citations [1], [2], [3].
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.2,
                max_tokens=1800
            )
            
            return {
                'response': response.choices[0].message.content.strip(),
                'committee_perspectives': perspectives,
                'synthesis_approach': 'committee_consultation',
                'committees_consulted': committees,
                'confidence': 0.92,
                'strategy': 'committee_enhanced_rag',
                'sources': sources or []
            }
            
        except Exception as e:
            logger.error(f"Committee consultation failed: {e}")
            # Fallback to single agent
            return self._execute_single_agent(org_id, query, context, sources)
    
    def _execute_single_agent(self, org_id: str, query: str, context: str, sources: List[Dict] = None) -> Dict[str, Any]:
        """Execute single agent response with veteran board member wisdom"""
        
        # Build enhanced context with document sources
        enhanced_context = context
        if sources:
            enhanced_context += "\n\nDOCUMENT SOURCES:\n"
            for i, source in enumerate(sources[:5], 1):
                enhanced_context += f"[{i}] {source.get('title', 'Document')} (Page {source.get('page', '?')})\n"
        
        full_prompt = f"""
        {self.instructions}
        
        GOVERNANCE CONTEXT: {enhanced_context}
        
        USER QUERY: {query}
        
        Provide your institutional wisdom and guidance on this matter using the standard BoardContinuity format.
        Ground your response in specific historical experience and precedent analysis.
        Reference documents using simple numbered citations [1], [2], [3] when applicable.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            
            return {
                'response': response.choices[0].message.content.strip(),
                'approach': 'single_agent_veteran',
                'confidence': 0.85,
                'strategy': 'veteran_rag',
                'sources': sources or []
            }
            
        except Exception as e:
            logger.error(f"Single agent execution failed: {e}")
            return {
                'response': f"I apologize, but I'm having difficulty accessing my institutional memory at the moment. Please try your question again.",
                'error': str(e),
                'confidence': 0.2,
                'strategy': 'error_fallback',
                'sources': sources or []
            }
    
    def _format_committee_perspectives(self, perspectives: List[Dict[str, Any]]) -> str:
        """Format committee perspectives for synthesis"""
        if not perspectives:
            return "No committee perspectives available."
        
        formatted = []
        for perspective in perspectives:
            committee = perspective.get('committee', 'Unknown Committee')
            response = perspective.get('response', 'No response available')
            formatted.append(f"""
{committee} Committee Perspective:
{response}
---
""")
        
        return "\n".join(formatted)
    
    # Tool implementations for future expansion
    def _search_documents(self, org_id: str, query: str) -> Dict[str, Any]:
        """Search institutional documents for relevant information"""
        return {"status": "integrated_with_main_rag", "message": "Document search handled by main RAG system"}
    
    def _analyze_patterns(self, context: str) -> Dict[str, Any]:
        """Analyze historical patterns in governance decisions"""
        return {"status": "not_implemented", "message": "Pattern analysis integration pending"}
    
    def _lookup_precedents(self, decision_type: str) -> Dict[str, Any]:
        """Look up similar historical decisions and outcomes"""
        return {"status": "not_implemented", "message": "Precedent lookup integration pending"}
    
    def _consult_committees(self, query: str, committees: List[str]) -> Dict[str, Any]:
        """Consult specific committee specialists"""
        if self.committee_agents_enabled and self.committee_manager:
            return {"committees": self.committee_manager.get_committee_perspectives(query, "", committees)}
        return {"status": "disabled", "message": "Committee consultation not available"}
    
    def _predict_outcomes(self, proposal: str) -> Dict[str, Any]:
        """Predict outcomes based on historical patterns"""
        return {"status": "not_implemented", "message": "Outcome prediction integration pending"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'guardrails_enabled': self.guardrails_enabled,
            'committee_agents_enabled': self.committee_agents_enabled,
            'intervention_enabled': self.intervention_enabled,
            'monitoring_enabled': self.monitoring_enabled,
            'total_tools': len(self.tools)
        }
        
        # Add performance monitoring data if available
        if self.monitoring_enabled and self.performance_monitor:
            try:
                performance_summary = self.performance_monitor.get_performance_summary()
                status['performance_monitoring'] = performance_summary
            except Exception as e:
                logger.warning(f"Failed to get performance summary: {e}")
                status['performance_monitoring'] = {'error': str(e)}
        
        return status

# Factory function for easy integration
def create_enterprise_rag_agent() -> EnterpriseRAGAgent:
    """Factory function to create enterprise RAG agent"""
    return EnterpriseRAGAgent()