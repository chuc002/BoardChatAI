import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SmartFileProcessor:
    """Advanced document processor for multiple formats with governance entity extraction"""
    
    def __init__(self):
        self.governance_patterns = self._initialize_governance_patterns()
        self.visual_element_patterns = self._initialize_visual_patterns()
        self.voting_patterns = self._initialize_voting_patterns()
        
    def process_document_intelligently(self, text_content: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process document with intelligent extraction of governance elements"""
        
        try:
            processing_results = {
                'document_id': document_metadata.get('id'),
                'document_type': self._classify_document_type(text_content, document_metadata),
                'extracted_entities': {},
                'visual_elements': {},
                'governance_actions': {},
                'processing_confidence': 0.0,
                'extraction_summary': []
            }
            
            # Extract governance entities
            processing_results['extracted_entities'] = self._extract_governance_entities(text_content)
            
            # Process visual elements
            processing_results['visual_elements'] = self._identify_visual_elements(text_content)
            
            # Extract voting and decision data
            processing_results['governance_actions'] = self._extract_governance_actions(text_content)
            
            # Calculate processing confidence
            processing_results['processing_confidence'] = self._calculate_confidence(processing_results)
            
            # Generate extraction summary
            processing_results['extraction_summary'] = self._generate_extraction_summary(processing_results)
            
            logger.info(f"Smart processing completed for {document_metadata.get('filename', 'document')} "
                       f"with {processing_results['processing_confidence']:.0%} confidence")
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Smart processing failed: {e}")
            return self._create_fallback_result(document_metadata, error=str(e))
    
    def _initialize_governance_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for governance entity extraction"""
        return {
            'motions': [
                r'(?i)(?:motion|move|moved)\s+(?:to\s+)?([^.]{10,100})',
                r'(?i)(?:it\s+was\s+)?moved\s+(?:and\s+seconded\s+)?(?:that\s+)?([^.]{10,100})',
                r'(?i)resolved\s+that\s+([^.]{10,100})'
            ],
            'votes': [
                r'(?i)(?:vote|voting)\s*[:]\s*([^.]{5,50})',
                r'(?i)(\d+)\s+(?:in\s+)?favor[,\s]*(\d+)\s+opposed[,\s]*(\d*)\s*abstain',
                r'(?i)approved\s+by\s+(?:a\s+)?vote\s+of\s+(\d+)[-\s](\d+)',
                r'(?i)unanimously\s+(?:approved|passed|carried)'
            ],
            'committees': [
                r'(?i)(finance|membership|house|golf|grounds|food\s*&?\s*beverage|strategic|nominating|audit|board|executive)\s+committee',
                r'(?i)committee\s+(?:on\s+)?([a-zA-Z\s]{3,20})',
                r'(?i)([a-zA-Z\s]{3,20})\s+committee\s+(?:report|meeting|chair)'
            ],
            'members': [
                r'(?i)(?:member|director|governor|trustee|officer)\s+([A-Z][a-zA-Z\s]{2,25})',
                r'(?i)([A-Z][a-zA-Z\s]{2,25})\s+(?:moved|seconded|reported|presented)',
                r'(?i)(?:chair|chairman|chairwoman|president|vice\s*president)\s+([A-Z][a-zA-Z\s]{2,25})'
            ],
            'financial': [
                r'\$[\d,]+\.?\d*',
                r'(?i)budget\s+(?:of\s+)?\$?([\d,]+\.?\d*)',
                r'(?i)(?:dues|fee|assessment|cost)\s*[:]\s*\$?([\d,]+\.?\d*)',
                r'(?i)(?:revenue|income|expense|expenditure)\s+(?:of\s+)?\$?([\d,]+\.?\d*)'
            ],
            'dates': [
                r'(?i)(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}[,\s]+\d{4}',
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'(?i)(?:effective|beginning|ending|expires?|due)\s+([^.]{5,30})'
            ]
        }
    
    def _initialize_visual_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for visual element detection"""
        return {
            'tables': [
                r'(?i)(?:table|chart|schedule)\s+\d+',
                r'(?:\|[^|]*){3,}',  # Table-like structures
                r'(?:\t[^\t]*){3,}',  # Tab-separated columns
                r'(?i)(?:see\s+)?(?:attached\s+)?(?:table|schedule|chart|exhibit)'
            ],
            'charts': [
                r'(?i)(?:chart|graph|figure)\s+\d+',
                r'(?i)(?:pie\s+chart|bar\s+chart|line\s+graph)',
                r'(?i)(?:see\s+)?(?:attached\s+)?(?:chart|graph|figure)'
            ],
            'voting_tallies': [
                r'(?i)(?:tally|count|results?):\s*(\d+)',
                r'(?i)(?:yes|aye|in\s+favor):\s*(\d+)',
                r'(?i)(?:no|nay|opposed):\s*(\d+)',
                r'(?i)abstain(?:ing)?:\s*(\d+)'
            ],
            'signatures': [
                r'(?i)(?:signed|signature|attest)[:]\s*([A-Z][a-zA-Z\s]{2,25})',
                r'(?i)([A-Z][a-zA-Z\s]{2,25})[,\s]*(?:secretary|president|chair)'
            ]
        }
    
    def _initialize_voting_patterns(self) -> Dict[str, str]:
        """Initialize specific voting result patterns"""
        return {
            'unanimous': r'(?i)unanimously\s+(?:approved|passed|carried|adopted)',
            'majority': r'(?i)(?:majority|carried)\s+(?:vote|approval)',
            'failed': r'(?i)(?:motion\s+)?(?:failed|defeated|rejected)',
            'tabled': r'(?i)(?:motion\s+)?(?:tabled|deferred|postponed)',
            'amended': r'(?i)(?:motion\s+)?(?:amended|modified|revised)'
        }
    
    def _classify_document_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """Classify document type based on content and metadata"""
        
        title = metadata.get('title', '').lower()
        filename = metadata.get('filename', '').lower()
        content_lower = content.lower()
        
        # Check title/filename patterns first
        if any(term in title + filename for term in ['bylaw', 'constitution']):
            return 'bylaws'
        elif any(term in title + filename for term in ['rule', 'regulation', 'policy']):
            return 'rules_policies'
        elif any(term in title + filename for term in ['meeting', 'minute']):
            return 'meeting_minutes'
        elif any(term in title + filename for term in ['budget', 'financial', 'audit']):
            return 'financial'
        elif any(term in title + filename for term in ['member', 'roster']):
            return 'membership'
        
        # Check content patterns
        meeting_indicators = content_lower.count('motion') + content_lower.count('vote') + content_lower.count('meeting')
        financial_indicators = content_lower.count('$') + content_lower.count('budget') + content_lower.count('expense')
        rule_indicators = content_lower.count('shall') + content_lower.count('must') + content_lower.count('policy')
        
        if meeting_indicators > 5:
            return 'meeting_minutes'
        elif financial_indicators > 3:
            return 'financial'
        elif rule_indicators > 5:
            return 'rules_policies'
        else:
            return 'governance_general'
    
    def _extract_governance_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract governance-specific entities from content"""
        
        entities = {}
        
        for entity_type, patterns in self.governance_patterns.items():
            entities[entity_type] = []
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Clean and deduplicate matches
                    clean_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            match_text = ' '.join(str(m) for m in match if m)
                        else:
                            match_text = str(match)
                        
                        cleaned = match_text.strip()
                        if cleaned and len(cleaned) > 3 and cleaned not in clean_matches:
                            clean_matches.append(cleaned)
                    
                    entities[entity_type].extend(clean_matches[:5])  # Limit to top 5 per pattern
        
        return entities
    
    def _identify_visual_elements(self, content: str) -> Dict[str, Any]:
        """Identify and extract visual elements from document"""
        
        visual_elements = {
            'tables_detected': 0,
            'charts_referenced': 0,
            'voting_tallies': [],
            'signatures_found': [],
            'visual_content_score': 0.0
        }
        
        # Count tables and charts
        for pattern in self.visual_element_patterns['tables']:
            visual_elements['tables_detected'] += len(re.findall(pattern, content, re.MULTILINE))
        
        for pattern in self.visual_element_patterns['charts']:
            visual_elements['charts_referenced'] += len(re.findall(pattern, content))
        
        # Extract voting tallies
        for pattern in self.visual_element_patterns['voting_tallies']:
            matches = re.findall(pattern, content)
            visual_elements['voting_tallies'].extend(matches)
        
        # Extract signatures
        for pattern in self.visual_element_patterns['signatures']:
            matches = re.findall(pattern, content)
            visual_elements['signatures_found'].extend(matches)
        
        # Calculate visual content score
        total_visual = (visual_elements['tables_detected'] + 
                       visual_elements['charts_referenced'] + 
                       len(visual_elements['voting_tallies']) + 
                       len(visual_elements['signatures_found']))
        
        visual_elements['visual_content_score'] = min(total_visual / 10.0, 1.0)
        
        return visual_elements
    
    def _extract_governance_actions(self, content: str) -> Dict[str, Any]:
        """Extract specific governance actions and decisions"""
        
        actions = {
            'motions_identified': [],
            'voting_results': [],
            'decisions_made': [],
            'action_items': [],
            'governance_score': 0.0
        }
        
        # Extract motions with context
        motion_patterns = self.governance_patterns['motions']
        for pattern in motion_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if len(match) > 10:
                    actions['motions_identified'].append({
                        'text': match.strip(),
                        'type': 'motion',
                        'confidence': 0.8
                    })
        
        # Extract voting results
        for vote_type, pattern in self.voting_patterns.items():
            matches = re.findall(pattern, content)
            for match in matches:
                actions['voting_results'].append({
                    'type': vote_type,
                    'text': match if isinstance(match, str) else ' '.join(match),
                    'confidence': 0.9 if vote_type == 'unanimous' else 0.7
                })
        
        # Extract decisions and resolutions
        decision_patterns = [
            r'(?i)(?:resolved|decided|determined)\s+that\s+([^.]{10,100})',
            r'(?i)(?:board|committee)\s+(?:approved|adopted|accepted)\s+([^.]{10,100})',
            r'(?i)(?:it\s+was\s+)?agreed\s+(?:that\s+)?([^.]{10,100})'
        ]
        
        for pattern in decision_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                actions['decisions_made'].append({
                    'decision': match.strip(),
                    'confidence': 0.8
                })
        
        # Calculate governance score
        total_actions = (len(actions['motions_identified']) + 
                        len(actions['voting_results']) + 
                        len(actions['decisions_made']))
        
        actions['governance_score'] = min(total_actions / 5.0, 1.0)
        
        return actions
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall processing confidence score"""
        
        entity_score = sum(len(entities) for entities in results['extracted_entities'].values()) / 20.0
        visual_score = results['visual_elements'].get('visual_content_score', 0.0)
        governance_score = results['governance_actions'].get('governance_score', 0.0)
        
        # Weight the scores
        overall_confidence = (entity_score * 0.4 + visual_score * 0.3 + governance_score * 0.3)
        
        return min(overall_confidence, 1.0)
    
    def _generate_extraction_summary(self, results: Dict[str, Any]) -> List[str]:
        """Generate human-readable summary of extraction results"""
        
        summary = []
        
        # Document classification
        doc_type = results.get('document_type', 'unknown')
        summary.append(f"Document classified as: {doc_type.replace('_', ' ').title()}")
        
        # Entity extraction summary
        entities = results['extracted_entities']
        total_entities = sum(len(e) for e in entities.values())
        if total_entities > 0:
            summary.append(f"Extracted {total_entities} governance entities")
            
            # Highlight key extractions
            if entities.get('motions'):
                summary.append(f"Found {len(entities['motions'])} motions/resolutions")
            if entities.get('votes'):
                summary.append(f"Identified {len(entities['votes'])} voting records")
            if entities.get('financial'):
                summary.append(f"Extracted {len(entities['financial'])} financial amounts")
        
        # Visual elements summary
        visual = results['visual_elements']
        if visual.get('tables_detected', 0) > 0:
            summary.append(f"Detected {visual['tables_detected']} tables/schedules")
        if visual.get('voting_tallies'):
            summary.append(f"Found {len(visual['voting_tallies'])} voting tallies")
        
        # Governance actions summary
        actions = results['governance_actions']
        total_actions = len(actions.get('motions_identified', [])) + len(actions.get('voting_results', []))
        if total_actions > 0:
            summary.append(f"Processed {total_actions} governance actions")
        
        # Confidence summary
        confidence = results.get('processing_confidence', 0.0)
        if confidence > 0.8:
            summary.append("High confidence extraction")
        elif confidence > 0.6:
            summary.append("Good confidence extraction")
        else:
            summary.append("Basic extraction completed")
        
        return summary
    
    def _create_fallback_result(self, metadata: Dict[str, Any], error: str = None) -> Dict[str, Any]:
        """Create fallback result when processing fails"""
        
        return {
            'document_id': metadata.get('id'),
            'document_type': 'governance_general',
            'extracted_entities': {},
            'visual_elements': {'visual_content_score': 0.0},
            'governance_actions': {'governance_score': 0.0},
            'processing_confidence': 0.3,
            'extraction_summary': ['Basic processing completed', f'Error: {error}' if error else 'Limited extraction'],
            'error': error
        }

def create_smart_processor() -> SmartFileProcessor:
    """Factory function to create smart file processor instance"""
    return SmartFileProcessor()