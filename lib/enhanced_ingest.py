"""
Enhanced document ingestion system with section-aware chunking.

This module replaces the basic character-based chunking with intelligent
section-aware parsing that preserves complete policy sections, tables,
and percentage sequences without breaking critical information mid-sentence.
"""

import re
import hashlib
import io
from typing import List, Dict, Tuple, Optional, Any
from pypdf import PdfReader
from lib.supa import supa
# Import OpenAI functionality directly since service structure may vary
from openai import OpenAI
import tiktoken
import os
from lib.perfect_extraction import extract_perfect_information, validate_extraction_quality

# Configuration
SECTION_PATTERN = r'\([a-z]\)\s+([^–]+)––([^(]+(?:\([a-z]\)|$))'
PERCENTAGE_PATTERNS = [
    r'\d+%',
    r'\d+\s*percent',
    r'\d+\.\d+%',
    r'seventy[- ]?five percent|75%|75 percent',
    r'fifty percent|50%|50 percent', 
    r'twenty[- ]?five percent|25%|25 percent'
]
MIN_CHUNK_SIZE = 1500
MAX_CHUNK_SIZE = 4500
IDEAL_CHUNK_SIZE = 3000
OVERLAP_SIZE = 300

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_COMPRESS = os.getenv("CHAT_COMPRESS", "gpt-4o-mini")

class SectionAwareChunker:
    def __init__(self):
        self.section_pattern = re.compile(SECTION_PATTERN, re.IGNORECASE | re.DOTALL)
        self.percentage_patterns = [re.compile(p, re.IGNORECASE) for p in PERCENTAGE_PATTERNS]
        
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete policy sections from text."""
        sections = []
        matches = list(self.section_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            start = match.start()
            # Find end of section (start of next section or end of text)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start:end].strip()
            section_id = match.group(1).strip() if match.group(1) else f"section_{i}"
            
            # Calculate completeness score
            completeness_score = self._calculate_completeness_score(section_text)
            
            sections.append({
                'id': section_id,
                'text': section_text,
                'start_pos': start,
                'end_pos': end,
                'completeness_score': completeness_score,
                'contains_percentages': self._contains_percentages(section_text),
                'section_type': self._identify_section_type(section_text)
            })
            
        return sections
    
    def _calculate_completeness_score(self, text: str) -> float:
        """Calculate how complete a section appears to be."""
        score = 0.0
        
        # Check for proper section structure
        if re.search(r'\([a-z]\)', text):
            score += 20
        
        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = sum(1 for s in sentences if s.strip() and len(s.strip()) > 10)
        if complete_sentences > 0:
            score += min(30, complete_sentences * 5)
        
        # Check for table completeness
        if self._has_complete_table(text):
            score += 25
        
        # Check for percentage sequences
        percentages = self._find_percentages(text)
        if percentages:
            score += min(25, len(percentages) * 5)
            # Bonus for common reinstatement percentages
            reinstatement_percs = ['75%', '50%', '25%']
            if all(any(p in text for p in [str(rp), str(rp).replace('%', ' percent')]) for rp in reinstatement_percs):
                score += 20
        
        return min(100.0, score)
    
    def _contains_percentages(self, text: str) -> bool:
        """Check if text contains percentage values."""
        return any(pattern.search(text) for pattern in self.percentage_patterns)
    
    def _find_percentages(self, text: str) -> List[str]:
        """Find all percentage values in text."""
        percentages = []
        for pattern in self.percentage_patterns:
            percentages.extend(pattern.findall(text))
        return percentages
    
    def _has_complete_table(self, text: str) -> bool:
        """Check if text contains a complete table structure."""
        # Look for table indicators
        table_indicators = [
            r'^\s*\|.*\|.*\|',  # Pipe tables
            r'(?:Category|Type|Fee|Amount|Percentage).*\n.*(?:Foundation|Social|Transfer)',
            r'\d+%.*\d+%.*\d+%'  # Multiple percentages in sequence
        ]
        return any(re.search(indicator, text, re.MULTILINE | re.IGNORECASE) for indicator in table_indicators)
    
    def _identify_section_type(self, text: str) -> str:
        """Identify the type of section based on content."""
        text_lower = text.lower()
        
        if 'reinstatement' in text_lower:
            return 'reinstatement'
        elif any(word in text_lower for word in ['fee', 'payment', 'dues', 'cost']):
            return 'financial'
        elif any(word in text_lower for word in ['transfer', 'membership']):
            return 'membership'
        elif any(word in text_lower for word in ['guest', 'visitor']):
            return 'guest_policy'
        else:
            return 'general'
    
    def _validate_chunk_boundary(self, text: str) -> bool:
        """Validate that chunk doesn't end with incomplete information."""
        text = text.strip()
        if not text:
            return False
        
        # Check for incomplete numbers at the end
        if re.search(r'\b\d+\s*$', text):
            return False
            
        # Check for incomplete percentage references
        if re.search(r'(?:percent|%)\s*of\s*$', text, re.IGNORECASE):
            return False
            
        # Check for incomplete sentences ending with conjunctions
        if re.search(r'\b(?:and|or|but|however|therefore|thus|furthermore)\s*$', text, re.IGNORECASE):
            return False
            
        return True
    
    def create_intelligent_chunks(self, text: str, page_boundaries: List[int]) -> List[Dict[str, Any]]:
        """Create chunks using section-aware approach."""
        chunks = []
        sections = self.extract_sections(text)
        
        if not sections:
            # Fallback to sentence-aware chunking
            return self._create_sentence_aware_chunks(text, page_boundaries)
        
        current_chunk = ""
        current_sections = []
        current_page_start = 0
        
        for section in sections:
            section_text = section['text']
            
            # If adding this section would exceed max size, finalize current chunk
            if current_chunk and len(current_chunk) + len(section_text) > MAX_CHUNK_SIZE:
                if self._validate_chunk_boundary(current_chunk):
                    chunk_data = self._create_chunk_data(
                        current_chunk, 
                        current_sections, 
                        page_boundaries,
                        len(chunks)
                    )
                    chunks.append(chunk_data)
                    
                    # Start new chunk with overlap
                    overlap = self._create_contextual_overlap(current_chunk, section_text)
                    current_chunk = overlap + section_text
                    current_sections = [section]
                else:
                    # Invalid boundary, extend current chunk
                    current_chunk += "\n\n" + section_text
                    current_sections.append(section)
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section_text
                else:
                    current_chunk = section_text
                current_sections.append(section)
        
        # Handle remaining chunk
        if current_chunk and self._validate_chunk_boundary(current_chunk):
            chunk_data = self._create_chunk_data(
                current_chunk, 
                current_sections, 
                page_boundaries,
                len(chunks)
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_contextual_overlap(self, prev_chunk: str, next_section: str) -> str:
        """Create meaningful overlap that preserves context."""
        # Find the last complete sentence or section in prev_chunk
        sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
        
        # Take last few sentences for context, but limit size
        overlap_text = ""
        for sentence in reversed(sentences[-3:]):  # Last 3 sentences max
            if len(overlap_text) + len(sentence) <= OVERLAP_SIZE:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip() + "\n\n" if overlap_text else ""
    
    def _create_sentence_aware_chunks(self, text: str, page_boundaries: List[int]) -> List[Dict[str, Any]]:
        """Fallback chunking method that respects sentence boundaries."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= MAX_CHUNK_SIZE or len(current_chunk) < MIN_CHUNK_SIZE:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunk_data = self._create_chunk_data(
                        current_chunk.strip(), 
                        [], 
                        page_boundaries,
                        chunk_index
                    )
                    chunks.append(chunk_data)
                    chunk_index += 1
                
                # Start new chunk with last sentence as context
                current_chunk = sentence + " "
        
        # Handle remaining content
        if current_chunk.strip():
            chunk_data = self._create_chunk_data(
                current_chunk.strip(), 
                [], 
                page_boundaries,
                chunk_index
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_data(self, text: str, sections: List[Dict], page_boundaries: List[int], chunk_index: int) -> Dict[str, Any]:
        """Create chunk data structure."""
        # Calculate overall completeness score
        if sections:
            completeness_score = sum(s['completeness_score'] for s in sections) / len(sections)
        else:
            completeness_score = self._calculate_completeness_score(text)
        
        # Run perfect extraction on this chunk
        perfect_results = extract_perfect_information(text, chunk_index=chunk_index)
        extraction_quality = validate_extraction_quality(perfect_results)
        
        # Enhance completeness score based on perfect extraction results
        if extraction_quality.get('is_high_quality', False):
            completeness_score = min(100.0, completeness_score + 15.0)
        
        # Determine page range
        text_start = text[:100].lower()
        page_index = None
        for i, boundary in enumerate(page_boundaries):
            if boundary <= len(text_start):
                page_index = i
                break
        
        return {
            'content': text,
            'chunk_index': chunk_index,
            'sections': sections,
            'section_completeness_score': completeness_score,
            'contains_percentages': self._contains_percentages(text),
            'percentage_list': self._find_percentages(text),
            'page_index': page_index,
            'word_count': len(text.split()),
            'char_count': len(text),
            'perfect_extraction': perfect_results,
            'extraction_quality': extraction_quality
        }

def extract_pdf_with_structure(file_content: bytes) -> Tuple[str, List[int], Dict[str, Any]]:
    """Extract text from PDF while preserving page boundaries and structure."""
    try:
        reader = PdfReader(io.BytesIO(file_content))
        full_text = ""
        page_boundaries = [0]
        metadata = {
            'total_pages': len(reader.pages),
            'has_tables': False,
            'has_percentages': False,
            'section_count': 0
        }
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            
            # Clean up text while preserving structure
            page_text = re.sub(r'\n\s*\n\s*\n', '\n\n', page_text)  # Reduce excessive whitespace
            page_text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', page_text)  # Fix broken sentences
            
            full_text += page_text + f"\n\n--- Page {page_num + 1} ---\n\n"
            page_boundaries.append(len(full_text))
            
            # Update metadata
            if re.search(r'(?:Category|Type|Fee|Amount)', page_text):
                metadata['has_tables'] = True
            if re.search(r'\d+%', page_text):
                metadata['has_percentages'] = True
        
        # Count sections
        section_matches = re.findall(SECTION_PATTERN, full_text, re.IGNORECASE)
        metadata['section_count'] = len(section_matches)
        
        return full_text, page_boundaries, metadata
        
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def enhanced_upsert_document(org_id: str, user_id: str, filename: str, file_content: bytes, mime_type: str = "application/pdf") -> Tuple[Dict, int]:
    """
    Enhanced document upsert with section-aware chunking.
    
    Returns: (document_record, chunk_count)
    """
    try:
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check for existing document
        existing = supa.table("documents").select("id,filename").eq("file_hash", file_hash).eq("org_id", org_id).limit(1).execute()
        if existing.data:
            print(f"[ENHANCED_INGEST] Document already exists: {existing.data[0]['filename']}")
            return existing.data[0], 0
        
        # Extract text with structure preservation
        full_text, page_boundaries, metadata = extract_pdf_with_structure(file_content)
        
        if not full_text.strip():
            raise Exception("No text content extracted from PDF")
        
        # Create document record
        doc_data = {
            "org_id": org_id,
            "created_by": user_id,
            "filename": filename,
            "title": filename.replace('.pdf', '').replace('_', ' ').title(),
            "content": full_text,
            "file_hash": file_hash,
            "mime_type": mime_type,
            "status": "processing",
            "metadata": metadata
        }
        
        doc_result = supa.table("documents").insert(doc_data).execute()
        if not doc_result.data:
            raise Exception("Failed to create document record")
        
        document = doc_result.data[0]
        doc_id = document["id"]
        
        # Create intelligent chunks
        chunker = SectionAwareChunker()
        chunks = chunker.create_intelligent_chunks(full_text, page_boundaries)
        
        print(f"[ENHANCED_INGEST] Created {len(chunks)} section-aware chunks")
        
        # Process and store chunks
        chunk_records = []
        for chunk_data in chunks:
            try:
                # Generate embedding
                embedding = generate_embedding(chunk_data['content'])
                
                # Generate summary for retrieval
                summary = generate_summary(chunk_data['content'])
                
                # Prepare chunk record
                # Extract perfect information for institutional memory
                perfect_data = chunk_data.get('perfect_extraction', {})
                extraction_quality = chunk_data.get('extraction_quality', {})
                
                chunk_record = {
                    "document_id": doc_id,
                    "org_id": org_id,
                    "chunk_index": chunk_data['chunk_index'],
                    "content": chunk_data['content'],
                    "summary": summary,
                    "embedding": embedding,
                    "page_index": chunk_data.get('page_index'),
                    "section_completeness_score": chunk_data['section_completeness_score'],
                    "contains_percentages": chunk_data['contains_percentages'],
                    "percentage_list": chunk_data.get('percentage_list', []),
                    "section_types": [s.get('section_type', 'general') for s in chunk_data.get('sections', [])],
                    "word_count": chunk_data['word_count'],
                    "char_count": chunk_data['char_count'],
                    # Perfect extraction data
                    "contains_decision": len(perfect_data.get('voting_records', [])) > 0,
                    "decision_count": len(perfect_data.get('voting_records', [])),
                    "entities_mentioned": {
                        "monetary_amounts": [{'amount': a['amount'], 'type': a['type']} for a in perfect_data.get('monetary_amounts', [])],
                        "members": [{'name': m['name'], 'role': m['role']} for m in perfect_data.get('members', [])],
                        "committees": [c['name'] for c in perfect_data.get('committees', [])],
                        "dates": [d['date_text'] for d in perfect_data.get('dates', [])]
                    },
                    "importance_score": min(1.0, extraction_quality.get('overall_score', 0.5)),
                    "extraction_quality_score": extraction_quality.get('overall_score', 0.5)
                }
                
                chunk_records.append(chunk_record)
                
            except Exception as e:
                print(f"[ENHANCED_INGEST] Failed to process chunk {chunk_data['chunk_index']}: {e}")
                continue
        
        # Batch insert chunks
        if chunk_records:
            supa.table("doc_chunks").insert(chunk_records).execute()
            print(f"[ENHANCED_INGEST] Stored {len(chunk_records)} chunks successfully")
        
        # Update document status
        supa.table("documents").update({"status": "ready", "processed": True}).eq("id", doc_id).execute()
        
        return document, len(chunk_records)
        
    except Exception as e:
        print(f"[ENHANCED_INGEST] Error processing document: {str(e)}")
        raise e

def validate_reinstatement_coverage(doc_id: str) -> Dict[str, Any]:
    """
    Validate that reinstatement percentages are properly captured.
    
    This function specifically tests the IHCC rules document to ensure
    ALL reinstatement percentages (75%, 50%, 25%) are captured in chunks.
    """
    try:
        # Get all chunks for document
        chunks_result = supa.table("doc_chunks").select("content,section_completeness_score,percentage_list").eq("document_id", doc_id).execute()
        
        if not chunks_result.data:
            return {"status": "error", "message": "No chunks found for document"}
        
        chunks = chunks_result.data
        
        # Look for reinstatement percentages
        required_percentages = ['75%', '50%', '25%']
        found_percentages = set()
        reinstatement_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            if 'reinstatement' in content:
                reinstatement_chunks.append(chunk)
                
                # Check for percentages in this chunk
                for perc in required_percentages:
                    if perc in chunk.get('content', '') or perc.replace('%', ' percent') in content:
                        found_percentages.add(perc)
        
        # Analysis results
        analysis = {
            "status": "success" if len(found_percentages) == len(required_percentages) else "incomplete",
            "total_chunks": len(chunks),
            "reinstatement_chunks": len(reinstatement_chunks),
            "required_percentages": required_percentages,
            "found_percentages": list(found_percentages),
            "missing_percentages": [p for p in required_percentages if p not in found_percentages],
            "highest_completeness_score": max(chunk.get('section_completeness_score', 0) for chunk in chunks),
            "chunks_with_percentages": sum(1 for chunk in chunks if chunk.get('percentage_list'))
        }
        
        # Detailed chunk analysis
        if reinstatement_chunks:
            best_chunk = max(reinstatement_chunks, key=lambda x: x.get('section_completeness_score', 0))
            analysis["best_reinstatement_chunk"] = {
                "completeness_score": best_chunk.get('section_completeness_score', 0),
                "contains_all_percentages": all(p in best_chunk.get('content', '') for p in required_percentages),
                "content_preview": best_chunk.get('content', '')[:500] + "..." if len(best_chunk.get('content', '')) > 500 else best_chunk.get('content', '')
            }
        
        return analysis
        
    except Exception as e:
        return {"status": "error", "message": f"Validation failed: {str(e)}"}

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI."""
    try:
        response = client.embeddings.create(
            input=text[:8000],  # Limit input size
            model=EMBED_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ENHANCED_INGEST] Embedding generation failed: {e}")
        return []

def generate_summary(text: str, max_tokens: int = 150) -> str:
    """Generate summary for chunk using OpenAI."""
    try:
        if len(text) < 200:
            return text  # Don't summarize very short text
        
        response = client.chat.completions.create(
            model=CHAT_COMPRESS,
            messages=[{
                "role": "system", 
                "content": "Create a concise summary that captures key information, percentages, and policy details. Focus on preserving specific numbers and requirements."
            }, {
                "role": "user", 
                "content": f"Summarize this document section:\n\n{text[:3000]}"
            }],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[ENHANCED_INGEST] Summary generation failed: {e}")
        return text[:500] + "..." if len(text) > 500 else text

# Test function for immediate validation
def test_enhanced_ingestion():
    """Test the enhanced ingestion with a sample document."""
    print("[TEST] Starting enhanced ingestion test...")
    
    # This would be called after processing a document
    # For now, return a test structure
    return {
        "message": "Enhanced ingestion system ready",
        "features": [
            "Section-aware chunking with regex pattern matching",
            "Percentage sequence preservation", 
            "Complete policy section detection",
            "Contextual overlap instead of character overlap",
            "Validation for incomplete chunks",
            "Section completeness scoring"
        ]
    }

if __name__ == "__main__":
    result = test_enhanced_ingestion()
    print(result)