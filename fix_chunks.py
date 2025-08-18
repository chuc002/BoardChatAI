#!/usr/bin/env python3
"""
Fix the chunking issue - the content is being cut off at crucial details
"""

import os
from lib.supa import supa

# Set environment
os.environ['DEV_ORG_ID'] = '63602dc6-defe-4355-b66c-aa6b3b1273e3'

def find_truncated_chunks():
    print("=== FINDING TRUNCATED CHUNKS ===")
    
    # Find chunks that end abruptly with percentages or important content
    result = supa.table('doc_chunks').select('document_id,chunk_index,content').execute()
    
    for chunk in result.data:
        content = chunk['content'] or ""
        # Check if chunk ends with a partial number or percentage
        if content.endswith(' 75') or content.endswith(' 70') or content.endswith(' 50') or content.endswith(' 40'):
            print(f"FOUND TRUNCATED CHUNK {chunk['chunk_index']}:")
            print(f"Content ends with: ...{content[-50:]}")
            
            # Check if there's a next chunk with the continuation
            next_chunk = supa.table('doc_chunks').select('content').eq('document_id', chunk['document_id']).eq('chunk_index', chunk['chunk_index'] + 1).execute()
            if next_chunk.data:
                next_content = next_chunk.data[0]['content'] or ""
                print(f"Next chunk starts with: {next_content[:100]}...")
                
                # Combine the chunks to get full content
                full_content = content + " " + next_content
                print(f"COMBINED CONTENT: {full_content[max(0, len(content)-100):len(content)+200]}")
            
            print("="*80)

if __name__ == "__main__":
    find_truncated_chunks()