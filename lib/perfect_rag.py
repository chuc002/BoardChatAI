"""
Enhanced RAG system with perfect context assembly and veteran board member intelligence.
Provides sophisticated context building with specific historical details and predictive insights.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

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
        """Extract specific numerical and temporal details from content"""
        
        details = {
            'amounts': re.findall(r'\$[\d,]+(?:\.\d{2})?', content),
            'years': re.findall(r'\b(?:19|20)\d{2}\b', content),
            'percentages': re.findall(r'\d+(?:\.\d+)?%', content),
            'dates': re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', content),
            'timeframes': re.findall(r'\b\d+\s+(?:days?|weeks?|months?|years?)\b', content),
            'vote_counts': re.findall(r'\b\d+\s*-\s*\d+\s*(?:vote|approval)\b', content, re.IGNORECASE),
            'meeting_references': re.findall(r'\b(?:board|committee|meeting)\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2}\/\d{1,2}\/\d{4})\b', content, re.IGNORECASE)
        }
        
        return {k: v for k, v in details.items() if v}  # Only return non-empty lists
    
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
        
        # Build enhanced context package
        context_package = {
            'primary_contexts': raw_contexts,
            'extracted_details': all_details,
            'historical_patterns': patterns,
            'query': query,
            'context_summary': {
                'total_sources': len(raw_contexts),
                'years_mentioned': len(all_details.get('years', [])),
                'amounts_mentioned': len(all_details.get('amounts', [])),
                'patterns_identified': len(patterns)
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

# Global instance for use across the application
perfect_rag = PerfectRAGEngine()

# Legacy functions for compatibility with existing imports
def retrieve_perfect_context(org_id: str, query: str) -> Dict[str, Any]:
    """Legacy function for retrieving context (compatibility wrapper)"""
    try:
        # Import the original RAG functions to maintain compatibility
        from lib.rag import perform_rag_advanced
        return perform_rag_advanced(org_id, query)
    except Exception as e:
        logger.error(f"Error in retrieve_perfect_context: {e}")
        return {"error": str(e)}

def generate_perfect_rag_response(org_id: str, query: str) -> Dict[str, Any]:
    """Enhanced RAG response with veteran board member intelligence"""
    try:
        # Import the original RAG functions
        from lib.rag import perform_rag_advanced
        
        # Get basic RAG response
        basic_response = perform_rag_advanced(org_id, query)
        
        if isinstance(basic_response, tuple) and len(basic_response) >= 2:
            response_text, metadata = basic_response
            
            # Extract contexts from metadata if available
            contexts = []
            if isinstance(metadata, dict) and metadata.get('sources'):
                contexts = metadata.get('sources', [])
            
            # Build enhanced context package
            context_package = perfect_rag.build_veteran_context_package(query, contexts)
            
            # Generate precedent warnings and predictions
            warnings = perfect_rag.generate_precedent_warnings(query, context_package)
            predictions = perfect_rag.predict_outcomes(query, context_package)
            
            # Enhanced metadata with veteran insights
            enhanced_metadata = {
                **(metadata if isinstance(metadata, dict) else {}),
                'veteran_insights': {
                    'extracted_details': context_package.get('extracted_details', {}),
                    'historical_patterns': context_package.get('historical_patterns', []),
                    'precedent_warnings': warnings,
                    'outcome_predictions': predictions,
                    'context_summary': context_package.get('context_summary', {})
                }
            }
            
            return {
                'response': response_text,
                'metadata': enhanced_metadata,
                'veteran_enhanced': True
            }
        else:
            # Fallback for unexpected response format
            return {
                'response': str(basic_response),
                'metadata': {},
                'veteran_enhanced': False
            }
            
    except Exception as e:
        logger.error(f"Error in generate_perfect_rag_response: {e}")
        return {
            'response': f"I encountered an issue accessing my institutional memory. Please try again.",
            'metadata': {'error': str(e)},
            'veteran_enhanced': False
        }