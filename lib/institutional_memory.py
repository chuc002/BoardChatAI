"""
Institutional Memory System - Advanced decision tracking and pattern analysis.

This module provides functions to extract decisions from documents, track patterns,
and maintain institutional knowledge for perfect organizational recall.
"""

import re
import json
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple, Any
from lib.supa import supa
import os

# Decision extraction patterns
DECISION_PATTERNS = [
    r'(?:voted|approved|rejected|deferred|tabled)\s+(?:to\s+)?(.+?)(?:by\s+(?:a\s+)?vote\s+of\s+)?(\d+)[-–—](\d+)[-–—](\d+)',
    r'(?:motion|proposal)\s+(?:to\s+)?(.+?)\s+(?:was\s+)?(?:approved|passed|rejected|failed)',
    r'(?:board\s+)?(?:decided|resolved|determined)\s+(?:to\s+)?(.+)',
    r'(?:fee|dues|assessment)\s+(?:of\s+)?\$?([\d,]+(?:\.\d{2})?)',
    r'(?:membership|transfer)\s+(?:fee|cost|charge)\s+(?:of\s+)?\$?([\d,]+(?:\.\d{2})?)',
]

# Voting patterns
VOTING_PATTERNS = [
    r'(?:vote|voting):\s*(\d+)[-–—](\d+)[-–—](\d+)',
    r'(?:in\s+favor|for):\s*(\d+)[,;\s]*(?:against|opposed):\s*(\d+)[,;\s]*(?:abstain|abstaining):\s*(\d+)',
    r'(\d+)\s+(?:in\s+favor|for)[,;\s]*(\d+)\s+(?:against|opposed)[,;\s]*(\d+)\s+(?:abstain|abstaining)',
]

# Financial amount patterns
AMOUNT_PATTERNS = [
    r'\$?([\d,]+(?:\.\d{2})?)',
    r'(?:dollar|dollar amount|sum|cost|fee|charge|payment)(?:\s+of)?\s+\$?([\d,]+(?:\.\d{2})?)',
]

class DecisionExtractor:
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.decision_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in DECISION_PATTERNS]
        self.voting_patterns = [re.compile(p, re.IGNORECASE) for p in VOTING_PATTERNS]
        self.amount_patterns = [re.compile(p, re.IGNORECASE) for p in AMOUNT_PATTERNS]
    
    def extract_decisions_from_chunk(self, chunk_content: str, document_id: str, chunk_index: int) -> List[Dict[str, Any]]:
        """Extract decision information from a document chunk."""
        decisions = []
        
        # Look for decision indicators
        decision_matches = []
        for pattern in self.decision_patterns:
            matches = pattern.findall(chunk_content)
            decision_matches.extend(matches)
        
        if not decision_matches:
            return decisions
        
        # Extract voting information
        voting_info = self._extract_voting_info(chunk_content)
        
        # Extract financial amounts
        amounts = self._extract_amounts(chunk_content)
        
        # Extract dates
        dates = self._extract_dates(chunk_content)
        
        # Create decision record
        for i, match in enumerate(decision_matches):
            decision = {
                'decision_type': self._classify_decision_type(chunk_content),
                'title': self._extract_title(match, chunk_content),
                'description': chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content,
                'source_document_id': document_id,
                'chunk_index': chunk_index,
                'voting_info': voting_info,
                'financial_amounts': amounts,
                'extracted_dates': dates,
                'content_context': chunk_content
            }
            decisions.append(decision)
        
        return decisions
    
    def _extract_voting_info(self, text: str) -> Dict[str, Any]:
        """Extract voting information from text."""
        for pattern in self.voting_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    return {
                        'votes_for': int(groups[0]),
                        'votes_against': int(groups[1]),
                        'votes_abstain': int(groups[2]) if groups[2] else 0
                    }
        return {}
    
    def _extract_amounts(self, text: str) -> List[float]:
        """Extract financial amounts from text."""
        amounts = []
        for pattern in self.amount_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    if amount > 0:
                        amounts.append(amount)
                except ValueError:
                    continue
        return list(set(amounts))  # Remove duplicates
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    
    def _classify_decision_type(self, text: str) -> str:
        """Classify the type of decision based on content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['membership', 'member', 'join', 'resign', 'transfer']):
            return 'membership'
        elif any(word in text_lower for word in ['fee', 'dues', 'cost', 'payment', 'budget', 'financial']):
            return 'financial'
        elif any(word in text_lower for word in ['policy', 'rule', 'bylaw', 'governance']):
            return 'governance'
        elif any(word in text_lower for word in ['emergency', 'urgent', 'immediate']):
            return 'emergency'
        else:
            return 'general'
    
    def _extract_title(self, match: Any, context: str) -> str:
        """Extract a title for the decision."""
        if isinstance(match, tuple) and match:
            title = str(match[0]).strip()
        else:
            title = str(match).strip()
        
        # Clean up the title
        title = re.sub(r'\s+', ' ', title)
        title = title[:100] + "..." if len(title) > 100 else title
        
        return title or "Decision extracted from document"

def store_decision(org_id: str, user_id: str, decision_data: Dict[str, Any]) -> Optional[str]:
    """Store a decision in the decision registry."""
    try:
        # Generate decision ID
        today = datetime.now().strftime("%Y-%m-%d")
        decision_type = decision_data.get('decision_type', 'general').upper()
        
        # Get count for today to create unique ID
        existing_today = supa.table('decision_registry').select('decision_id').eq('org_id', org_id).gte('date', today).execute()
        count = len(existing_today.data) if existing_today.data else 0
        
        decision_id = f"{today}-{decision_type}-{count+1:03d}"
        
        # Prepare record
        record = {
            'decision_id': decision_id,
            'org_id': org_id,
            'created_by': user_id,
            'date': decision_data.get('extracted_dates', [today])[0] if decision_data.get('extracted_dates') else today,
            'decision_type': decision_data.get('decision_type', 'general'),
            'title': decision_data.get('title', 'Extracted Decision'),
            'description': decision_data.get('description', ''),
            'source_document_id': decision_data.get('source_document_id'),
            'outcome': 'extracted',  # Mark as extracted vs manually entered
            'tags': [decision_data.get('decision_type', 'general'), 'extracted']
        }
        
        # Add voting information if available
        voting_info = decision_data.get('voting_info', {})
        if voting_info:
            record.update({
                'vote_count_for': voting_info.get('votes_for', 0),
                'vote_count_against': voting_info.get('votes_against', 0),
                'vote_count_abstain': voting_info.get('votes_abstain', 0)
            })
        
        # Add financial information if available
        amounts = decision_data.get('financial_amounts', [])
        if amounts:
            record['amount_involved'] = max(amounts)  # Use largest amount
        
        # Insert decision
        result = supa.table('decision_registry').insert(record).execute()
        
        if result.data:
            return result.data[0]['id']
        else:
            print(f"[INSTITUTIONAL_MEMORY] Failed to store decision: {decision_id}")
            return None
            
    except Exception as e:
        print(f"[INSTITUTIONAL_MEMORY] Error storing decision: {str(e)}")
        return None

def analyze_patterns(org_id: str) -> Dict[str, Any]:
    """Analyze decision patterns for the organization."""
    try:
        # Get all decisions
        decisions = supa.table('decision_registry').select('*').eq('org_id', org_id).execute()
        
        if not decisions.data:
            return {"message": "No decisions found for pattern analysis"}
        
        decisions_data = decisions.data
        
        # Analyze by type
        type_analysis = {}
        for decision in decisions_data:
            decision_type = decision.get('decision_type', 'general')
            if decision_type not in type_analysis:
                type_analysis[decision_type] = {
                    'count': 0,
                    'approved': 0,
                    'rejected': 0,
                    'total_amount': 0,
                    'decisions': []
                }
            
            type_analysis[decision_type]['count'] += 1
            type_analysis[decision_type]['decisions'].append(decision['id'])
            
            outcome = decision.get('outcome', '')
            if outcome in ['approved', 'passed']:
                type_analysis[decision_type]['approved'] += 1
            elif outcome in ['rejected', 'failed']:
                type_analysis[decision_type]['rejected'] += 1
            
            amount = decision.get('amount_involved', 0)
            if amount:
                type_analysis[decision_type]['total_amount'] += float(amount)
        
        # Calculate success rates
        for decision_type, analysis in type_analysis.items():
            total_voted = analysis['approved'] + analysis['rejected']
            analysis['success_rate'] = (analysis['approved'] / total_voted * 100) if total_voted > 0 else 0
        
        # Store patterns
        for decision_type, analysis in type_analysis.items():
            pattern_data = {
                'org_id': org_id,
                'pattern_type': decision_type,
                'pattern_name': f"{decision_type.title()} Decisions",
                'frequency_count': analysis['count'],
                'success_rate': analysis['success_rate'],
                'typical_amount': analysis['total_amount'] / analysis['count'] if analysis['count'] > 0 else 0,
                'decision_instances': analysis['decisions'],
                'last_occurrence': datetime.now().date()
            }
            
            # Upsert pattern
            existing = supa.table('historical_patterns').select('id').eq('org_id', org_id).eq('pattern_type', decision_type).execute()
            
            if existing.data:
                supa.table('historical_patterns').update(pattern_data).eq('id', existing.data[0]['id']).execute()
            else:
                supa.table('historical_patterns').insert(pattern_data).execute()
        
        return {
            "patterns_analyzed": len(type_analysis),
            "total_decisions": len(decisions_data),
            "pattern_summary": type_analysis
        }
        
    except Exception as e:
        print(f"[INSTITUTIONAL_MEMORY] Pattern analysis error: {str(e)}")
        return {"error": str(e)}

def enhance_chunk_with_decisions(chunk_id: str, org_id: str, user_id: str) -> bool:
    """Enhance a document chunk by extracting and linking decisions."""
    try:
        # Get chunk data
        chunk = supa.table('doc_chunks').select('*').eq('id', chunk_id).execute()
        
        if not chunk.data:
            return False
        
        chunk_data = chunk.data[0]
        content = chunk_data.get('content', '')
        document_id = chunk_data.get('document_id')
        chunk_index = chunk_data.get('chunk_index', 0)
        
        # Extract decisions
        extractor = DecisionExtractor(org_id)
        decisions = extractor.extract_decisions_from_chunk(content, document_id, chunk_index)
        
        if not decisions:
            # Update chunk to mark as analyzed
            supa.table('doc_chunks').update({
                'contains_decision': False,
                'decision_count': 0
            }).eq('id', chunk_id).execute()
            return True
        
        # Store extracted decisions
        decision_ids = []
        for decision_data in decisions:
            decision_id = store_decision(org_id, user_id, decision_data)
            if decision_id:
                decision_ids.append(decision_id)
        
        # Extract entities and create cross-references
        entities = extract_entities_from_text(content)
        
        # Update chunk with decision information
        update_data = {
            'contains_decision': True,
            'decision_count': len(decisions),
            'extracted_decisions': decision_ids,
            'entities_mentioned': entities,
            'importance_score': min(1.0, 0.5 + (len(decisions) * 0.2))  # Higher score for chunks with more decisions
        }
        
        supa.table('doc_chunks').update(update_data).eq('id', chunk_id).execute()
        
        return True
        
    except Exception as e:
        print(f"[INSTITUTIONAL_MEMORY] Chunk enhancement error: {str(e)}")
        return False

def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """Extract entities (names, amounts, dates) from text."""
    entities = {
        'people': [],
        'amounts': [],
        'dates': [],
        'organizations': []
    }
    
    # Extract names (basic pattern)
    name_patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
        r'\b(?:Mr|Mrs|Ms|Dr|President|Chairman|Secretary|Treasurer)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, text)
        entities['people'].extend(matches)
    
    # Extract amounts
    amount_matches = re.findall(r'\$?([\d,]+(?:\.\d{2})?)', text)
    entities['amounts'] = [match for match in amount_matches if match]
    
    # Extract dates
    date_matches = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', text, re.IGNORECASE)
    entities['dates'] = date_matches
    
    # Remove duplicates
    for key in entities:
        entities[key] = list(set(entities[key]))
    
    return entities

def process_document_for_institutional_memory(document_id: str, org_id: str, user_id: str) -> Dict[str, Any]:
    """Process an entire document to extract institutional memory."""
    try:
        # Get all chunks for the document
        chunks = supa.table('doc_chunks').select('*').eq('document_id', document_id).execute()
        
        if not chunks.data:
            return {"error": "No chunks found for document"}
        
        processed_chunks = 0
        decisions_found = 0
        
        for chunk in chunks.data:
            if enhance_chunk_with_decisions(chunk['id'], org_id, user_id):
                processed_chunks += 1
                # Check if decisions were found
                updated_chunk = supa.table('doc_chunks').select('decision_count').eq('id', chunk['id']).execute()
                if updated_chunk.data and updated_chunk.data[0].get('decision_count', 0) > 0:
                    decisions_found += updated_chunk.data[0]['decision_count']
        
        # Analyze patterns after processing
        pattern_analysis = analyze_patterns(org_id)
        
        return {
            "processed_chunks": processed_chunks,
            "decisions_extracted": decisions_found,
            "pattern_analysis": pattern_analysis
        }
        
    except Exception as e:
        print(f"[INSTITUTIONAL_MEMORY] Document processing error: {str(e)}")
        return {"error": str(e)}

def get_institutional_insights(org_id: str, query: str = None) -> Dict[str, Any]:
    """Get institutional insights and knowledge for a query."""
    try:
        insights = {
            "decisions": [],
            "patterns": [],
            "knowledge": [],
            "summary": {}
        }
        
        # Get recent decisions
        recent_decisions = supa.table('decision_registry').select('*').eq('org_id', org_id).order('date', desc=True).limit(10).execute()
        if recent_decisions.data:
            insights["decisions"] = recent_decisions.data
        
        # Get patterns
        patterns = supa.table('historical_patterns').select('*').eq('org_id', org_id).order('frequency_count', desc=True).limit(5).execute()
        if patterns.data:
            insights["patterns"] = patterns.data
        
        # Get institutional knowledge
        knowledge = supa.table('institutional_knowledge').select('*').eq('org_id', org_id).eq('is_current', True).order('confidence_score', desc=True).limit(10).execute()
        if knowledge.data:
            insights["knowledge"] = knowledge.data
        
        # Create summary
        insights["summary"] = {
            "total_decisions": len(insights["decisions"]),
            "active_patterns": len(insights["patterns"]),
            "knowledge_items": len(insights["knowledge"]),
            "last_updated": datetime.now().isoformat()
        }
        
        return insights
        
    except Exception as e:
        print(f"[INSTITUTIONAL_MEMORY] Insights error: {str(e)}")
        return {"error": str(e)}