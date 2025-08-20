import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from lib.supa import supa

logger = logging.getLogger(__name__)

class ContextualChunker:
    """Enhanced chunking system that adds governance-specific context to document chunks"""
    
    def __init__(self):
        self.club_context_cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    def create_contextual_chunk(self, chunk: str, document_metadata: Dict[str, Any]) -> str:
        """Create a chunk with rich governance context"""
        
        try:
            # Get club-specific context
            club_context = self.get_club_context(document_metadata.get('org_id'))
            
            # Extract governance metadata
            doc_title = document_metadata.get('title', 'Governance Document')
            doc_date = self._format_date(document_metadata.get('created_at', document_metadata.get('date')))
            committee = self._extract_committee(doc_title)
            meeting_type = self._extract_meeting_type(doc_title)
            document_type = self._classify_document_type(doc_title, chunk)
            
            # Create contextual prefix
            context_prefix = f"""GOVERNANCE CONTEXT:
Document: {doc_title}
Date: {doc_date}
Committee: {committee}
Meeting Type: {meeting_type}
Document Type: {document_type}
Organization: {club_context['name']}
Context: {club_context['type']}

INSTITUTIONAL BACKGROUND:
{club_context['background']}

CONTENT:
"""
            
            return context_prefix + chunk.strip()
            
        except Exception as e:
            logger.error(f"Failed to create contextual chunk: {e}")
            # Fallback to basic context
            return f"GOVERNANCE DOCUMENT:\n{document_metadata.get('title', 'Document')}\n\nCONTENT:\n{chunk}"
    
    def get_club_context(self, org_id: Optional[str] = None) -> Dict[str, str]:
        """Get club-specific contextual information"""
        
        if not org_id:
            return self._get_default_club_context()
            
        # Check cache first
        cache_key = f"club_{org_id}"
        if cache_key in self.club_context_cache:
            cached_data = self.club_context_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < self.cache_ttl:
                return cached_data['context']
        
        try:
            # Try to get organization-specific context from database
            context = self._fetch_org_context(org_id)
            
            # Cache the result
            self.club_context_cache[cache_key] = {
                'context': context,
                'timestamp': datetime.now()
            }
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to get org context for {org_id}: {e}")
            return self._get_default_club_context()
    
    def _fetch_org_context(self, org_id: str) -> Dict[str, str]:
        """Fetch organization-specific context from documents"""
        
        try:
            # Get a sample of documents to infer organization type
            docs_result = supa.table("documents").select("title,filename").eq("org_id", org_id).limit(10).execute()
            
            if docs_result.data:
                titles = [doc.get('title', doc.get('filename', '')) for doc in docs_result.data]
                org_type = self._infer_organization_type(titles)
                
                return {
                    'name': self._extract_organization_name(titles),
                    'type': org_type,
                    'background': self._generate_background_context(org_type, titles)
                }
        except Exception as e:
            logger.error(f"Failed to fetch org context: {e}")
        
        return self._get_default_club_context()
    
    def _get_default_club_context(self) -> Dict[str, str]:
        """Default club context for governance intelligence"""
        return {
            'name': 'Private Club Organization',
            'type': 'Private Membership Club',
            'background': 'This is a private membership organization with governance structures including board of directors, committees, and member oversight. Decisions follow established bylaws, policies, and precedent patterns typical of private club governance.'
        }
    
    def _infer_organization_type(self, titles: List[str]) -> str:
        """Infer organization type from document titles"""
        
        title_text = ' '.join(titles).lower()
        
        if any(term in title_text for term in ['country club', 'golf', 'course']):
            return 'Country Club'
        elif any(term in title_text for term in ['yacht', 'boat', 'marina']):
            return 'Yacht Club'
        elif any(term in title_text for term in ['city club', 'downtown', 'business']):
            return 'City Club'
        elif any(term in title_text for term in ['athletic', 'fitness', 'sports']):
            return 'Athletic Club'
        else:
            return 'Private Membership Club'
    
    def _extract_organization_name(self, titles: List[str]) -> str:
        """Extract organization name from document titles"""
        
        # Look for common patterns
        for title in titles:
            title_upper = title.upper()
            if 'COUNTRY CLUB' in title_upper:
                # Extract name before "COUNTRY CLUB"
                parts = title_upper.split('COUNTRY CLUB')[0].strip()
                if parts:
                    return f"{parts.title()} Country Club"
            elif 'YACHT CLUB' in title_upper:
                parts = title_upper.split('YACHT CLUB')[0].strip()
                if parts:
                    return f"{parts.title()} Yacht Club"
        
        return 'Private Club Organization'
    
    def _extract_committee(self, title: str) -> str:
        """Extract committee name from document title"""
        
        title_lower = title.lower()
        
        committees = {
            'board': 'Board of Directors',
            'finance': 'Finance Committee',
            'membership': 'Membership Committee',
            'house': 'House Committee',
            'golf': 'Golf Committee',
            'food': 'Food & Beverage Committee',
            'grounds': 'Grounds Committee',
            'strategic': 'Strategic Planning Committee',
            'nominating': 'Nominating Committee',
            'audit': 'Audit Committee'
        }
        
        for keyword, committee_name in committees.items():
            if keyword in title_lower:
                return committee_name
        
        return 'General Governance'
    
    def _extract_meeting_type(self, title: str) -> str:
        """Extract meeting type from document title"""
        
        title_lower = title.lower()
        
        if 'annual' in title_lower:
            return 'Annual Meeting'
        elif 'special' in title_lower:
            return 'Special Meeting'
        elif 'board' in title_lower:
            return 'Board Meeting'
        elif 'committee' in title_lower:
            return 'Committee Meeting'
        elif 'member' in title_lower:
            return 'Member Meeting'
        else:
            return 'Regular Meeting'
    
    def _classify_document_type(self, title: str, content: str) -> str:
        """Classify the type of governance document"""
        
        title_lower = title.lower()
        content_lower = content.lower()
        
        if any(term in title_lower for term in ['bylaw', 'constitution']):
            return 'Bylaws/Constitution'
        elif any(term in title_lower for term in ['rule', 'regulation', 'policy']):
            return 'Rules & Policies'
        elif any(term in title_lower for term in ['meeting', 'minute']):
            return 'Meeting Minutes'
        elif any(term in title_lower for term in ['budget', 'financial', 'audit']):
            return 'Financial Document'
        elif any(term in content_lower for term in ['motion', 'vote', 'resolved']):
            return 'Meeting Record'
        elif any(term in content_lower for term in ['membership', 'dues', 'initiation']):
            return 'Membership Document'
        else:
            return 'Governance Document'
    
    def _format_date(self, date_input: Any) -> str:
        """Format date for contextual display"""
        
        if not date_input:
            return 'Date Unknown'
        
        try:
            if isinstance(date_input, str):
                # Try to parse ISO format
                if 'T' in date_input:
                    date_obj = datetime.fromisoformat(date_input.replace('Z', '+00:00'))
                    return date_obj.strftime('%B %Y')
                else:
                    return date_input
            elif isinstance(date_input, datetime):
                return date_input.strftime('%B %Y')
            else:
                return str(date_input)
        except Exception:
            return str(date_input) if date_input else 'Date Unknown'
    
    def _generate_background_context(self, org_type: str, titles: List[str]) -> str:
        """Generate contextual background based on organization type"""
        
        contexts = {
            'Country Club': 'This country club operates with traditional governance including golf, dining, and social committees. Members enjoy golf privileges, dining facilities, and social events. Governance follows private club precedents with emphasis on member experience and facility management.',
            
            'Yacht Club': 'This yacht club focuses on boating, sailing, and marine activities. Governance includes docking, marine operations, and social committees. Members have boat privileges and participate in sailing events and marine activities.',
            
            'City Club': 'This city club provides business networking, dining, and professional services in an urban setting. Governance emphasizes business member services, downtown location advantages, and professional networking opportunities.',
            
            'Athletic Club': 'This athletic club emphasizes fitness, sports, and wellness programs. Governance includes athletics, fitness, and wellness committees focused on member health and recreational activities.',
            
            'Private Membership Club': 'This private membership organization operates with traditional club governance structures including member committees, board oversight, and established policies for member services and facility operations.'
        }
        
        return contexts.get(org_type, contexts['Private Membership Club'])

def create_contextual_chunker() -> ContextualChunker:
    """Factory function to create contextual chunker instance"""
    return ContextualChunker()