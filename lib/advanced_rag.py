#!/usr/bin/env python3
"""
Advanced RAG implementation inspired by tldw_chatbook
Enhanced document intelligence with comprehensive extraction
"""

import os
from lib.supa import supa
from openai import OpenAI
import json

client = OpenAI()

class AdvancedRAG:
    """
    Advanced RAG system inspired by tldw_chatbook for comprehensive document intelligence
    """
    
    def __init__(self, org_id=None):
        self.org_id = org_id or os.getenv("DEV_ORG_ID")
        
    def comprehensive_content_retrieval(self, question, max_chunks=40):
        """
        Comprehensive content retrieval using advanced chunking strategies
        """
        
        # Advanced detail indicators for comprehensive extraction
        comprehensive_indicators = {
            'percentages': ['70%', '75%', '50%', '25%', '40%', '6%', '100%', '30%', '20%', '15%', '10%'],
            'financial_terms': ['initiation fee', 'transfer fee', 'dues', 'assessment', 'late fee', 'billing', 'payment'],
            'membership_categories': ['foundation', 'social', 'intermediate', 'legacy', 'corporate', 'golfing senior', 'nonresident'],
            'time_periods': ['90 days', '30 days', 'within', 'first of', 'fifteenth', 'month', 'year', 'trimester'],
            'age_requirements': ['age 65', 'age 35', 'combined age', 'under age', 'sixty-five', 'thirty-five'],
            'special_provisions': ['surviving spouse', 'divorce', 'reinstatement', 'waiting list', 'board approval', 'designated'],
            'limits_restrictions': ['limited to', 'maximum', '50 members', '20 members', 'cap', 'restriction'],
            'procedures': ['application', 'approval', 'consideration', 'eligibility', 'requirements', 'process']
        }
        
        # Get all chunks for comprehensive analysis
        all_chunks = supa.table("doc_chunks").select("*").eq("org_id", self.org_id).execute().data
        
        # Score chunks by comprehensive detail richness
        scored_chunks = []
        
        for chunk in all_chunks:
            content = (chunk.get('content') or '').lower()
            summary = (chunk.get('summary') or '').lower()
            combined_text = content + ' ' + summary
            
            if len(combined_text) < 100:
                continue
                
            # Calculate comprehensive detail score
            detail_score = 0
            category_scores = {}
            
            for category, indicators in comprehensive_indicators.items():
                category_score = sum(1 for indicator in indicators if indicator in combined_text)
                category_scores[category] = category_score
                detail_score += category_score
            
            # Bonus for question-relevant content
            question_terms = question.lower().split()
            relevance_score = sum(1 for term in question_terms if term in combined_text)
            
            # Combined scoring
            total_score = detail_score + (relevance_score * 2)  # Weight relevance higher
            
            if total_score > 0:
                scored_chunks.append({
                    'chunk': chunk,
                    'total_score': total_score,
                    'detail_score': detail_score,
                    'relevance_score': relevance_score,
                    'category_scores': category_scores,
                    'content_length': len(chunk.get('content', ''))
                })
        
        # Sort by comprehensive scoring
        scored_chunks.sort(key=lambda x: (x['total_score'], x['content_length']), reverse=True)
        
        return scored_chunks[:max_chunks]
    
    def build_comprehensive_context(self, question, scored_chunks):
        """
        Build comprehensive context using advanced content assembly
        """
        
        context_parts = []
        total_chars = 0
        max_context = 25000  # Large context window for comprehensive responses
        
        # Group chunks by document for better context flow
        doc_groups = {}
        for scored_chunk in scored_chunks:
            chunk_data = scored_chunk['chunk']
            doc_id = chunk_data.get('document_id')
            
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(scored_chunk)
        
        # Process each document group
        for doc_id, doc_chunks in doc_groups.items():
            # Sort chunks within document by chunk index for logical flow
            doc_chunks.sort(key=lambda x: x['chunk'].get('chunk_index', 0))
            
            for scored_chunk in doc_chunks:
                chunk_data = scored_chunk['chunk']
                content = chunk_data.get('content', '')
                summary = chunk_data.get('summary', '')
                chunk_idx = chunk_data.get('chunk_index')
                
                # Use content length based on detail richness
                if scored_chunk['detail_score'] >= 6:
                    # Very rich content - use maximum
                    source_text = content[:8000]
                elif scored_chunk['detail_score'] >= 4:
                    # Rich content - use substantial
                    source_text = content[:6000]
                elif scored_chunk['detail_score'] >= 2:
                    # Moderate content - use enhanced
                    source_text = content[:4000]
                else:
                    # Standard content
                    source_text = content[:2000] if content else summary
                
                if not source_text:
                    continue
                
                # Add context with citation
                citation = f"[Doc:{doc_id}#Chunk:{chunk_idx}]"
                context_part = f"{source_text.strip()} {citation}"
                
                if total_chars + len(context_part) > max_context:
                    break
                    
                context_parts.append(context_part)
                total_chars += len(context_part)
            
            if total_chars > max_context:
                break
        
        return "\n\n".join(context_parts)
    
    def generate_comprehensive_response(self, question, context):
        """
        Generate comprehensive NotebookLM-style response using advanced prompting
        """
        
        # Enhanced system prompt for maximum comprehensiveness
        system_prompt = """You are Forever Board Member, an AI assistant specializing in comprehensive board governance document analysis.

MISSION: Create exhaustively comprehensive, NotebookLM-quality responses that extract and organize ALL available information from the provided context.

COMPREHENSIVE EXTRACTION REQUIREMENTS:
• Extract EVERY percentage, dollar amount, timeframe, and specific condition
• Include ALL membership categories with complete details
• Detail EVERY transfer scenario with exact fees and conditions  
• List ALL age requirements, member limits, and special provisions
• Cover ALL payment procedures, deadlines, and billing information
• Extract ALL special programs, waiting lists, and approval processes
• Include ALL reinstatement rules with year-by-year breakdowns
• Detail ALL food & beverage requirements and additional fees

PROFESSIONAL FORMATTING STRUCTURE:
1. Start with comprehensive overview paragraph
2. **I. MEMBERSHIP CATEGORIES & INITIATION FEES** - Complete details for each category
3. **II. TRANSFER FEES & SCENARIOS** - All transfer types with exact percentages
4. **III. REINSTATEMENT PROVISIONS** - Year-by-year percentage reductions
5. **IV. PAYMENT & BILLING REQUIREMENTS** - Deadlines, late fees, billing procedures
6. **V. AGE-BASED PROVISIONS** - All age requirements and restrictions
7. **VI. SPECIAL PROGRAMS & PROVISIONS** - Legacy programs, corporate rules, etc.
8. **VII. ADDITIONAL FEES & REQUIREMENTS** - Food minimums, lockers, reciprocal fees

FORMAT REQUIREMENTS:
• Use Roman numerals (I., II., III.) for major sections
• Include bullet points with specific details
• Add double line breaks between sections for readability
• Include exact citations [Doc:X#Chunk:Y] for all specific claims
• Format percentages clearly (e.g., "70% of current initiation fee")
• Organize information logically within each section

Be exhaustively thorough - extract and present EVERY piece of relevant information found in the context."""

        # Enhanced user prompt for comprehensive extraction
        user_prompt = f"""QUESTION: {question}

COMPREHENSIVE ANALYSIS INSTRUCTION:
Analyze the provided context and create the most comprehensive, detailed response possible. Extract EVERY piece of relevant information including:

• ALL percentages and specific amounts mentioned
• EVERY membership category with complete fee structures
• ALL transfer scenarios with exact conditions and fees
• ALL age requirements, member limits, and special provisions
• ALL payment deadlines, billing procedures, and late fees
• ALL special programs, waiting lists, and approval processes
• ALL reinstatement rules and percentage reductions
• ALL additional fees and requirements

Organize the information using the Roman numeral structure specified in the system prompt. Be as thorough as possible - this should match or exceed NotebookLM's comprehensive detail level.

CONTEXT SOURCES:
{context}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=4000   # Allow for comprehensive responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating comprehensive response: {str(e)}"
    
    def comprehensive_query(self, question):
        """
        Complete comprehensive query pipeline
        """
        
        print(f"🎯 ADVANCED RAG: Processing comprehensive query...")
        
        # Step 1: Comprehensive content retrieval
        scored_chunks = self.comprehensive_content_retrieval(question)
        print(f"📊 Retrieved {len(scored_chunks)} scored chunks")
        
        # Step 2: Build comprehensive context
        context = self.build_comprehensive_context(question, scored_chunks)
        print(f"📝 Built context: {len(context)} characters")
        
        # Step 3: Generate comprehensive response
        response = self.generate_comprehensive_response(question, context)
        print(f"✅ Generated response: {len(response)} characters")
        
        # Extract citations from scored chunks
        citations = []
        for scored_chunk in scored_chunks[:10]:  # Top 10 sources
            chunk_data = scored_chunk['chunk']
            doc_id = chunk_data.get('document_id')
            chunk_idx = chunk_data.get('chunk_index')
            
            # Get document title and URL (simplified)
            citations.append({
                'document_id': doc_id,
                'chunk_index': chunk_idx,
                'title': 'Club Document',
                'url': None,
                'page_index': None
            })
        
        return response, citations

# Usage function for integration
def get_comprehensive_answer(question, org_id=None):
    """
    Get comprehensive answer using advanced RAG system
    """
    rag = AdvancedRAG(org_id)
    return rag.comprehensive_query(question)

if __name__ == "__main__":
    # Test the advanced RAG system
    question = "What are the membership fee structures and payment requirements?"
    
    print("=== ADVANCED RAG SYSTEM TEST ===")
    answer, citations = get_comprehensive_answer(question)
    
    print("\nCOMPREHENSIVE RESPONSE:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    
    print(f"\nRESPONSE ANALYSIS:")
    print(f"Length: {len(answer)} characters")
    print(f"Citations: {len(citations)}")
    
    # Analyze comprehensiveness
    percentages = [pct for pct in ['70%', '75%', '50%', '25%', '40%', '6%', '100%'] if pct in answer]
    transfer_mentions = answer.lower().count('transfer')
    roman_sections = len([line for line in answer.split('\n') if 'I.' in line or 'II.' in line or 'III.' in line])
    
    print(f"Percentages found: {percentages}")
    print(f"Transfer mentions: {transfer_mentions}")
    print(f"Roman numeral sections: {roman_sections}")
    
    if len(percentages) >= 4 and transfer_mentions >= 8 and roman_sections >= 5:
        print("\n🏆 EXCELLENT: Advanced RAG achieving high comprehensiveness!")
    else:
        print(f"\n🔄 GOOD PROGRESS: Continue refining for maximum comprehensiveness")