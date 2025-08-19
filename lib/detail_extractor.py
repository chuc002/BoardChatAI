"""
Specific Detail Extraction System for BoardContinuity
Extracts precise financial, temporal, and governance data for veteran board member analysis
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SpecificDetailExtractor:
    """Advanced extraction system for board governance details"""
    
    def __init__(self):
        self.detail_patterns = {
            'amounts': r'\$[\d,]+(?:\.\d{2})?',
            'years': r'\b(?:19|20)\d{2}\b',
            'percentages': r'\d+(?:\.\d+)?%',
            'vote_counts': r'\d+-\d+\s*(?:vote|voting)',
            'timeframes': r'\d+\s*(?:days?|weeks?|months?|years?)',
            'committee_names': r'(?:Golf|F&B|Food\s*&\s*Beverage|House|Finance|Membership|Board|Executive|Governance|Nominating|Audit)\s*Committee',
            'member_titles': r'(?:President|Treasurer|Secretary|Chair|Chairman|Vice\s*Chair|Member|Director|Governor)',
            'meeting_references': r'(?:meeting|session|vote)\s*(?:on|in|of)\s*[\w\s,]+\d{4}',
            'policy_sections': r'(?:Section|Article|Rule|Policy)\s*\d+(?:\.\d+)?',
            'fee_categories': r'(?:Initiation|Transfer|Reinstatement|Monthly|Annual|Food|Beverage|Minimum)\s*(?:Fee|Charge|Cost|Payment)',
            'approval_statuses': r'(?:Approved|Denied|Tabled|Deferred|Unanimous|Majority)',
            'membership_categories': r'(?:Full|Social|Corporate|Junior|Senior|Foundation|Intermediate|Legacy)\s*(?:Member|Membership)',
            'date_references': r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            'quorum_counts': r'(?:quorum|present):\s*\d+',
            'motion_types': r'(?:Motion|Moved|Seconded)\s+(?:to|by)',
            'deadlines': r'(?:due|deadline|expires?|effective)\s+(?:by|on|in)\s+[\w\s,]+\d{4}'
        }
        
        # Enhanced patterns for specific governance contexts
        self.governance_patterns = {
            'budget_items': r'(?:Budget|Allocation|Expense|Revenue)\s*:\s*\$[\d,]+(?:\.\d{2})?',
            'member_counts': r'(?:Members?|Applicants?|Resignations?):\s*\d+',
            'waiting_list': r'(?:Waiting\s*List|Queue)\s*(?:Position|Number):\s*\d+',
            'assessment_fees': r'(?:Assessment|Special\s*Fee|Capital\s*Assessment)\s*:\s*\$[\d,]+',
            'reciprocal_clubs': r'(?:Reciprocal|Partner)\s+(?:Club|Agreement)',
            'terms_expiring': r'(?:Term|Position)\s+(?:expires?|ending)\s+\d{4}'
        }
    
    def extract_all_details(self, text: str) -> Dict[str, List[str]]:
        """Extract all specific details from text"""
        details = {}
        
        # Extract basic patterns
        for detail_type, pattern in self.detail_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                details[detail_type] = list(set(matches))  # Remove duplicates
        
        # Extract governance-specific patterns
        for detail_type, pattern in self.governance_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                details[detail_type] = list(set(matches))
        
        return details
    
    def create_detail_summary(self, details: Dict[str, List[str]]) -> str:
        """Create summary of extracted details for context"""
        summary_parts = []
        
        if details.get('amounts'):
            summary_parts.append(f"Financial amounts mentioned: {', '.join(details['amounts'][:5])}")
        
        if details.get('years'):
            years = sorted(details['years'])
            summary_parts.append(f"Historical years referenced: {', '.join(years)}")
        
        if details.get('percentages'):
            summary_parts.append(f"Percentages cited: {', '.join(details['percentages'])}")
        
        if details.get('vote_counts'):
            summary_parts.append(f"Vote counts recorded: {', '.join(details['vote_counts'])}")
        
        if details.get('committee_names'):
            summary_parts.append(f"Committees involved: {', '.join(details['committee_names'])}")
        
        if details.get('member_titles'):
            summary_parts.append(f"Leadership positions: {', '.join(details['member_titles'])}")
        
        if details.get('timeframes'):
            summary_parts.append(f"Timeframes mentioned: {', '.join(details['timeframes'])}")
        
        if details.get('membership_categories'):
            summary_parts.append(f"Membership types: {', '.join(details['membership_categories'])}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific details extracted."
    
    def extract_financial_context(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive financial context"""
        financial_data = {
            'amounts': [],
            'fee_types': [],
            'budget_items': [],
            'assessments': [],
            'financial_years': [],
            'payment_terms': []
        }
        
        # Extract amounts with context
        amount_pattern = r'(?P<type>\w+(?:\s+\w+)*)\s*(?:fee|cost|charge|payment|amount)?\s*[:=]\s*(?P<amount>\$[\d,]+(?:\.\d{2})?)'
        for match in re.finditer(amount_pattern, text, re.IGNORECASE):
            financial_data['amounts'].append({
                'amount': match.group('amount'),
                'type': match.group('type').strip(),
                'context': match.group(0)
            })
        
        # Extract fee categories
        fee_matches = re.findall(self.detail_patterns['fee_categories'], text, re.IGNORECASE)
        financial_data['fee_types'] = list(set(fee_matches))
        
        # Extract budget items
        budget_matches = re.findall(self.governance_patterns['budget_items'], text, re.IGNORECASE)
        financial_data['budget_items'] = list(set(budget_matches))
        
        return financial_data
    
    def extract_decision_context(self, text: str) -> Dict[str, Any]:
        """Extract decision-making context and patterns"""
        decision_data = {
            'votes': [],
            'committees': [],
            'positions': [],
            'meetings': [],
            'motions': [],
            'outcomes': []
        }
        
        # Extract vote patterns with context
        vote_pattern = r'(?P<motion>[^.]+?)\s*(?:voted|approved|denied)\s*(?P<count>\d+-\d+|\bunanimous\b|\bmajority\b)'
        for match in re.finditer(vote_pattern, text, re.IGNORECASE):
            decision_data['votes'].append({
                'motion': match.group('motion').strip(),
                'result': match.group('count'),
                'context': match.group(0)
            })
        
        # Extract committee involvement
        committee_matches = re.findall(self.detail_patterns['committee_names'], text, re.IGNORECASE)
        decision_data['committees'] = list(set(committee_matches))
        
        # Extract meeting references
        meeting_matches = re.findall(self.detail_patterns['meeting_references'], text, re.IGNORECASE)
        decision_data['meetings'] = list(set(meeting_matches))
        
        return decision_data
    
    def extract_timeline_context(self, text: str) -> Dict[str, Any]:
        """Extract temporal context and patterns"""
        timeline_data = {
            'dates': [],
            'timeframes': [],
            'deadlines': [],
            'years': [],
            'seasonal_patterns': []
        }
        
        # Extract specific dates
        date_matches = re.findall(self.detail_patterns['date_references'], text, re.IGNORECASE)
        timeline_data['dates'] = list(set(date_matches))
        
        # Extract timeframes with context
        timeframe_pattern = r'(?P<context>[^.]+?)\s*(?:within|in|after|takes?)\s*(?P<timeframe>\d+\s*(?:days?|weeks?|months?|years?))'
        for match in re.finditer(timeframe_pattern, text, re.IGNORECASE):
            timeline_data['timeframes'].append({
                'timeframe': match.group('timeframe'),
                'context': match.group('context').strip(),
                'full_match': match.group(0)
            })
        
        # Extract deadlines
        deadline_matches = re.findall(self.detail_patterns['deadlines'], text, re.IGNORECASE)
        timeline_data['deadlines'] = list(set(deadline_matches))
        
        return timeline_data
    
    def analyze_precedent_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for precedent and pattern indicators"""
        precedent_data = {
            'historical_references': [],
            'pattern_indicators': [],
            'precedent_warnings': [],
            'success_indicators': [],
            'failure_indicators': []
        }
        
        # Historical reference patterns
        historical_patterns = [
            r'(?:previously|historically|in the past|last time|when we)\s+[^.]+',
            r'(?:similar to|like when|as in)\s+\d{4}\s+[^.]+',
            r'(?:this happened before|we\'ve seen this|pattern shows)\s+[^.]+',
            r'(?:successful|failed|worked|didn\'t work)\s+(?:in|when|during)\s+[^.]+'
        ]
        
        for pattern in historical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            precedent_data['historical_references'].extend(matches)
        
        # Success/failure indicators
        success_terms = r'\b(?:successful|approved|effective|smooth|efficient|positive|beneficial)\b'
        failure_terms = r'\b(?:failed|rejected|problematic|delayed|complicated|negative|difficult)\b'
        
        success_matches = re.findall(f'[^.]*{success_terms}[^.]*', text, re.IGNORECASE)
        failure_matches = re.findall(f'[^.]*{failure_terms}[^.]*', text, re.IGNORECASE)
        
        precedent_data['success_indicators'] = success_matches[:3]  # Limit to most relevant
        precedent_data['failure_indicators'] = failure_matches[:3]
        
        return precedent_data
    
    def create_veteran_insight_summary(self, text: str) -> str:
        """Create comprehensive summary for veteran board member context"""
        all_details = self.extract_all_details(text)
        financial_context = self.extract_financial_context(text)
        decision_context = self.extract_decision_context(text)
        timeline_context = self.extract_timeline_context(text)
        precedent_context = self.analyze_precedent_patterns(text)
        
        summary_parts = []
        
        # Financial insights
        if financial_context['amounts']:
            amounts = [item['amount'] for item in financial_context['amounts']]
            summary_parts.append(f"FINANCIAL DATA: {', '.join(amounts[:5])}")
        
        # Decision insights
        if decision_context['votes']:
            vote_results = [item['result'] for item in decision_context['votes']]
            summary_parts.append(f"VOTING PATTERNS: {', '.join(vote_results)}")
        
        # Timeline insights
        if timeline_context['timeframes']:
            timeframes = [item['timeframe'] for item in timeline_context['timeframes']]
            summary_parts.append(f"PROCESSING TIMES: {', '.join(timeframes[:3])}")
        
        # Precedent insights
        if precedent_context['success_indicators']:
            summary_parts.append(f"SUCCESS FACTORS: {len(precedent_context['success_indicators'])} positive precedents found")
        
        if precedent_context['failure_indicators']:
            summary_parts.append(f"RISK FACTORS: {len(precedent_context['failure_indicators'])} negative precedents found")
        
        return "\n".join(summary_parts) if summary_parts else "Limited specific details available for veteran analysis."

# Global instance for use across the application
detail_extractor = SpecificDetailExtractor()