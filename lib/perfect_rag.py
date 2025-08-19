"""
Enhanced RAG system with perfect context assembly and veteran board member intelligence.
Provides sophisticated context building with specific historical details and predictive insights.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from lib.detail_extractor import detail_extractor
from lib.precedent_analyzer import precedent_analyzer

logger = logging.getLogger(__name__)

class PerfectRAGEngine:
    """Enhanced RAG engine for building comprehensive context with veteran intelligence"""
    
    def __init__(self):
        self.logger = logger
        
    def _build_enhanced_context_prompt(self, query: str, context_package: Dict[str, Any]) -> str:
        """Build context prompt with emphasis on specific details and predictions"""
        
        prompt_parts = [f"Question: {query}\n"]
        
        # Add historical context with emphasis on specifics
        if context_package.get('primary_contexts'):
            prompt_parts.append("HISTORICAL CONTEXT (extract ALL specific details):")
            for i, context in enumerate(context_package['primary_contexts'][:5]):
                prompt_parts.append(f"\n{i+1}. From {context.get('source', 'Document')} (page {context.get('page', '?')}):")
                prompt_parts.append(f"   {context.get('content', '')}")
                
                # Extract and highlight specific data
                content = context.get('content', '')
                amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', content)
                years = re.findall(r'\b(?:19|20)\d{2}\b', content)
                percentages = re.findall(r'\d+(?:\.\d+)?%', content)
                
                if amounts or years or percentages:
                    prompt_parts.append(f"   KEY DETAILS: Amounts: {amounts}, Years: {years}, Percentages: {percentages}")
        
        # Add pattern-based predictions
        if context_package.get('historical_patterns'):
            prompt_parts.append("\nHISTORICAL PATTERNS FOR PREDICTIONS:")
            for pattern in context_package['historical_patterns'][:3]:
                prompt_parts.append(f"- {pattern}")
        
        # Add specific instruction for veteran response
        prompt_parts.append("""
CRITICAL INSTRUCTIONS:
1. Reference specific amounts, years, and outcomes from the context
2. Warn about deviations from successful patterns
3. Predict outcomes based on historical success/failure rates
4. Include exact timelines from past similar decisions
5. Use veteran language: "In my experience...", "We tried this before...", "Based on [X] similar decisions..."
""")
        
        return "\n".join(prompt_parts)
    
    def extract_specific_details(self, content: str) -> Dict[str, List[str]]:
        """Extract specific numerical and temporal details from content using enhanced extractor"""
        return detail_extractor.extract_all_details(content)
    
    def identify_historical_patterns(self, contexts: List[Dict[str, Any]]) -> List[str]:
        """Identify recurring patterns across historical contexts"""
        
        patterns = []
        
        # Pattern: Recurring fee structures
        fee_mentions = []
        for ctx in contexts:
            content = ctx.get('content', '')
            if any(term in content.lower() for term in ['fee', 'cost', 'charge', 'payment']):
                amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', content)
                if amounts:
                    fee_mentions.extend(amounts)
        
        if len(set(fee_mentions)) > 1:
            patterns.append(f"Fee structure pattern: Multiple fee amounts found ({', '.join(set(fee_mentions)[:5])})")
        
        # Pattern: Approval timelines
        timeline_mentions = []
        for ctx in contexts:
            content = ctx.get('content', '')
            timelines = re.findall(r'\b\d+\s+(?:days?|weeks?|months?)\b', content)
            timeline_mentions.extend(timelines)
        
        if timeline_mentions:
            patterns.append(f"Approval timeline pattern: Typically {', '.join(set(timeline_mentions)[:3])}")
        
        # Pattern: Committee involvement
        committee_mentions = []
        for ctx in contexts:
            content = ctx.get('content', '')
            committees = re.findall(r'\b(?:finance|governance|membership|executive|board)\s+committee\b', content, re.IGNORECASE)
            committee_mentions.extend(committees)
        
        if committee_mentions:
            patterns.append(f"Committee involvement pattern: {', '.join(set(committee_mentions)[:3])} typically involved")
        
        return patterns
    
    def build_veteran_context_package(self, query: str, raw_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build comprehensive context package for veteran board member responses"""
        
        # Extract specific details from all contexts
        all_details = {}
        for ctx in raw_contexts:
            content = ctx.get('content', '')
            details = self.extract_specific_details(content)
            for key, values in details.items():
                if key not in all_details:
                    all_details[key] = []
                all_details[key].extend(values)
        
        # Remove duplicates while preserving order
        for key in all_details:
            all_details[key] = list(dict.fromkeys(all_details[key]))
        
        # Identify historical patterns
        patterns = self.identify_historical_patterns(raw_contexts)
        
        # Generate veteran insights for each context
        veteran_insights = []
        for ctx in raw_contexts:
            content = ctx.get('content', '')
            if content:
                insight = detail_extractor.create_veteran_insight_summary(content)
                if insight and insight != "Limited specific details available for veteran analysis.":
                    veteran_insights.append(insight)
        
        # Build enhanced context package
        context_package = {
            'primary_contexts': raw_contexts,
            'extracted_details': all_details,
            'historical_patterns': patterns,
            'veteran_insights': veteran_insights,
            'query': query,
            'context_summary': {
                'total_sources': len(raw_contexts),
                'years_mentioned': len(all_details.get('years', [])),
                'amounts_mentioned': len(all_details.get('amounts', [])),
                'patterns_identified': len(patterns),
                'veteran_insights_generated': len(veteran_insights)
            }
        }
        
        return context_package
    
    def generate_precedent_warnings(self, query: str, context_package: Dict[str, Any]) -> List[str]:
        """Generate specific precedent warnings based on historical context"""
        
        warnings = []
        details = context_package.get('extracted_details', {})
        
        # Warning for fee deviations
        if 'fee' in query.lower() and details.get('amounts'):
            amounts = details['amounts']
            if len(set(amounts)) > 2:
                warnings.append(f"Fee structure has varied historically ({', '.join(amounts[:3])}). Ensure consistency with established precedents.")
        
        # Warning for timeline deviations
        if any(term in query.lower() for term in ['approve', 'decision', 'process']) and details.get('timeframes'):
            timeframes = details['timeframes']
            warnings.append(f"Historical processing times: {', '.join(timeframes[:3])}. Rushing decisions has led to complications in the past.")
        
        # Warning for missing committee approval
        if 'approval' in query.lower() and details.get('meeting_references'):
            warnings.append("Historical precedent shows committee review is essential. Bypassing this step has caused problems in previous decisions.")
        
        return warnings
    
    def predict_outcomes(self, query: str, context_package: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes based on historical patterns and context"""
        
        predictions = {
            'success_probability': 'Unknown',
            'estimated_timeline': 'Unknown',
            'risk_factors': [],
            'success_factors': []
        }
        
        details = context_package.get('extracted_details', {})
        patterns = context_package.get('historical_patterns', [])
        
        # Timeline prediction
        if details.get('timeframes'):
            timelines = details['timeframes']
            # Extract numbers and find common timeline
            timeline_nums = []
            for t in timelines:
                nums = re.findall(r'\d+', t)
                if nums:
                    timeline_nums.append(int(nums[0]))
            
            if timeline_nums:
                avg_timeline = sum(timeline_nums) // len(timeline_nums)
                unit = 'days' if 'day' in timelines[0] else ('weeks' if 'week' in timelines[0] else 'months')
                predictions['estimated_timeline'] = f"{avg_timeline} {unit} (based on historical average)"
        
        # Success factors based on patterns
        if any('committee' in p.lower() for p in patterns):
            predictions['success_factors'].append("Committee involvement historically increases approval success")
        
        if details.get('amounts') and len(set(details['amounts'])) <= 2:
            predictions['success_factors'].append("Consistent fee structure aligns with historical precedents")
        
        # Risk factors
        if 'fee' in query.lower() and details.get('amounts') and len(set(details['amounts'])) > 3:
            predictions['risk_factors'].append("Multiple fee structures create precedent confusion")
        
        return predictions
    
    def generate_perfect_response(self, org_id: str, query: str) -> Dict[str, Any]:
        """Generate response with enhanced veteran wisdom and comprehensive analysis"""
        try:
            # Import OpenAI for direct response generation
            from openai import OpenAI
            client = OpenAI()
            
            # Get comprehensive context using existing RAG system
            from lib.rag import smart_retrieve
            contexts = smart_retrieve(org_id, query, k=5)
            
            # Create mock basic response structure for compatibility
            basic_response = ("", {"sources": contexts})
            
            if isinstance(basic_response, tuple) and len(basic_response) >= 2:
                response_text, metadata = basic_response
                contexts = metadata.get('sources', []) if isinstance(metadata, dict) else []
            else:
                # Fallback if RAG response format is unexpected
                contexts = []
                response_text = str(basic_response)
            
            # Build veteran context package
            context_package = self.build_veteran_context_package(query, contexts)
            
            # Generate comprehensive precedent analysis
            precedent_analysis = precedent_analyzer.analyze_precedents(query, contexts)
            
            # Extract all specific details
            all_details = context_package.get('extracted_details', {})
            
            # Build enhanced system prompt with veteran requirements
            system_prompt = """You are the digital embodiment of a 30-year veteran board member with perfect institutional memory. 

ENHANCED RESPONSE REQUIREMENTS:
1. Begin with "In my 30 years of board experience..." or similar veteran opening
2. Include specific amounts, years, and vote counts from the provided details
3. Reference historical precedents with exact examples and outcomes
4. Provide specific warnings based on past failures with percentages
5. Predict timeline and success probability based on similar decisions
6. Use veteran language patterns throughout
7. Structure response with clear sections for maximum impact

FORMAT YOUR RESPONSE AS:
### Historical Context
[Specific years, amounts, decisions with exact details from extracted data]

### Practical Wisdom  
[Precedent warnings and lessons learned with specific examples and percentages]

### Outcome Predictions
[Success rates, timelines, risk factors based on historical data]

### Implementation Guidance
[Step-by-step advice based on what has worked historically]

Use simple numbered citations [1], [2], [3] for readability."""
            
            # Build enhanced user prompt with specific details and precedents
            detail_summary = detail_extractor.create_veteran_insight_summary("\n".join([ctx.get('content', '') for ctx in contexts]))
            precedent_summary = precedent_analyzer.generate_veteran_precedent_summary(query, precedent_analysis)
            
            user_prompt_parts = [
                f"QUESTION: {query}\n",
                f"EXTRACTED INSTITUTIONAL DETAILS:\n{detail_summary}\n" if detail_summary else "",
                f"PRECEDENT ANALYSIS:\n{precedent_summary}\n" if precedent_summary else "",
                "RELEVANT HISTORICAL CONTEXT:"
            ]
            
            # Add top context with citations
            for i, context in enumerate(contexts[:3], 1):
                content = context.get('content', '')[:400]
                source = context.get('source', context.get('title', 'Document'))
                page = context.get('page', context.get('page_number', '?'))
                user_prompt_parts.append(f"[{i}] From {source} (page {page}):\n{content}...\n")
            
            user_prompt_parts.append("""
CRITICAL VETERAN RESPONSE REQUIREMENTS:
• Start with veteran perspective opening
• Include specific extracted amounts, years, and percentages in Historical Context
• Reference exact precedents with success/failure rates in Practical Wisdom
• Provide timeline predictions and risk assessments in Outcome Predictions
• Give step-by-step guidance in Implementation section
• Use authentic 30-year veteran language throughout
• Cite sources with simple numbered references [1], [2], [3]
""")
            
            user_prompt = "\n".join(filter(None, user_prompt_parts))
            
            # Generate enhanced veteran response
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Use the working model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
                
                enhanced_response_text = response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                # Fallback to basic response with veteran enhancement
                enhanced_response_text = f"In my 30 years of board experience, {response_text}"
            
            # Calculate enhanced confidence score
            confidence = self._calculate_enhanced_confidence(all_details, precedent_analysis)
            
            return {
                'response': enhanced_response_text,
                'context_used': context_package,
                'extracted_details': all_details,
                'precedent_analysis': precedent_analysis,
                'confidence': confidence,
                'sources': self._compile_enhanced_sources(contexts),
                'veteran_wisdom_applied': True,
                'precedent_score': precedent_analysis.get('precedent_score', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in generate_perfect_response: {e}")
            return {
                'response': f"I encountered an issue accessing my institutional memory. Please try again.",
                'error': str(e),
                'veteran_wisdom_applied': False
            }
    
    def _calculate_enhanced_confidence(self, details: Dict[str, List[str]], precedent_analysis: Dict[str, Any]) -> int:
        """Calculate confidence score based on details and precedent analysis"""
        confidence = 50  # Base confidence
        
        # Add confidence for specific details
        if details.get('amounts'):
            confidence += len(details['amounts']) * 5
        if details.get('years'):
            confidence += len(details['years']) * 3
        if details.get('vote_counts'):
            confidence += len(details['vote_counts']) * 10
        if details.get('committee_names'):
            confidence += len(details['committee_names']) * 5
        
        # Add confidence for precedent analysis
        precedent_score = precedent_analysis.get('precedent_score', 0)
        confidence += precedent_score * 0.3  # Scale precedent score
        
        # Add confidence for similar decisions
        similar_decisions = len(precedent_analysis.get('similar_decisions', []))
        confidence += similar_decisions * 5
        
        # Reduce confidence for high risk factors
        risk_factors = len(precedent_analysis.get('risk_factors', []))
        confidence -= risk_factors * 8
        
        return max(0, min(100, int(confidence)))
    
    def _compile_enhanced_sources(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compile enhanced source information with precedent context"""
        sources = []
        for i, context in enumerate(contexts[:5], 1):
            source_info = {
                'citation': f"[{i}]",
                'title': context.get('source', context.get('title', 'Document')),
                'page': context.get('page', context.get('page_number', '?')),
                'content_preview': context.get('content', '')[:200] + '...',
                'relevance_score': context.get('relevance_score', 0.5)
            }
            
            # Add extracted details for this source
            if context.get('content'):
                source_details = detail_extractor.extract_all_details(context['content'])
                if source_details:
                    source_info['extracted_details'] = source_details
            
            sources.append(source_info)
        
        return sources

# Global instance for use across the application
perfect_rag = PerfectRAGEngine()

# Legacy functions for compatibility with existing imports
def retrieve_perfect_context(org_id: str, query: str) -> Dict[str, Any]:
    """Legacy function for retrieving context (compatibility wrapper)"""
    try:
        # Import the original RAG functions to maintain compatibility
        from lib.rag import smart_retrieve
        contexts = smart_retrieve(org_id, query, k=5)
        return {"sources": contexts}
    except Exception as e:
        logger.error(f"Error in retrieve_perfect_context: {e}")
        return {"error": str(e)}

def generate_perfect_rag_response(org_id: str, query: str) -> Dict[str, Any]:
    """Enhanced RAG response with veteran board member intelligence using perfect response generation"""
    try:
        # Use the enhanced perfect response generation
        return perfect_rag.generate_perfect_response(org_id, query)
            
    except Exception as e:
        logger.error(f"Error in generate_perfect_rag_response: {e}")
        return {
            'response': f"I encountered an issue accessing my institutional memory. Please try again.",
            'metadata': {'error': str(e)},
            'veteran_enhanced': False
        }