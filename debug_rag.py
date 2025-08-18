#!/usr/bin/env python3
"""
Debug script to find why RAG isn't extracting the same content as NotebookLM
"""

import os
from lib.supa import supa
from lib.rag import answer_question_md

# Set environment
os.environ['DEV_ORG_ID'] = '63602dc6-defe-4355-b66c-aa6b3b1273e3'
os.environ['DEV_USER_ID'] = 'debug-user'

def test_question():
    question = "How do membership structures and financial obligations vary across different categories at the club?"
    
    print("=== TESTING RAG RETRIEVAL ===")
    print(f"Question: {question}")
    print()
    
    # Test the full RAG pipeline
    try:
        answer, citations = answer_question_md('63602dc6-defe-4355-b66c-aa6b3b1273e3', question)
        print("RAG Answer:")
        print(answer)
        print(f"\nCitations: {len(citations)}")
        for i, c in enumerate(citations):
            print(f"  {i+1}. {c.get('title', 'Unknown')} - Chunk {c.get('chunk_index', '?')}")
    except Exception as e:
        print(f"RAG Error: {e}")
    
    print("\n=== CHECKING DATABASE DIRECTLY ===")
    
    # Check the FULL content of chunks with reinstatement details
    print("\n=== FULL REINSTATEMENT CONTENT ===")
    result = supa.table('doc_chunks').select('chunk_index,content').ilike('content', '%reinstatement%').limit(1).execute()
    if result.data:
        chunk = result.data[0]
        print(f"Full Chunk {chunk['chunk_index']} content:")
        print(chunk['content'])
        print("="*80)
    
    # Check chunks with percentage content
    result = supa.table('doc_chunks').select('chunk_index,content').ilike('content', '%75%').limit(1).execute()
    if result.data:
        chunk = result.data[0]
        print(f"Full Chunk {chunk['chunk_index']} with 75% content:")
        print(chunk['content'])
        print("="*80)

if __name__ == "__main__":
    test_question()