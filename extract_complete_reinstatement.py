#!/usr/bin/env python3
"""
Extract the complete reinstatement content that NotebookLM can find
"""

import os
from lib.supa import supa

# Set environment
os.environ['DEV_ORG_ID'] = '63602dc6-defe-4355-b66c-aa6b3b1273e3'

def extract_complete_reinstatement():
    print("=== EXTRACTING COMPLETE REINSTATEMENT CONTENT ===")
    
    # Get the truncated chunk and the next chunk
    chunk2 = supa.table('doc_chunks').select('document_id,content').eq('chunk_index', 2).limit(1).execute()
    chunk3 = supa.table('doc_chunks').select('document_id,content').eq('chunk_index', 3).limit(1).execute()
    
    if chunk2.data and chunk3.data:
        doc_id = chunk2.data[0]['document_id']
        content2 = chunk2.data[0]['content'] or ""
        content3 = chunk3.data[0]['content'] or ""
        
        print("CHUNK 2 (truncated):")
        print("..."+ content2[-200:])
        print("\nCHUNK 3 (continuation):")
        print(content3[:500] + "...")
        
        # Find where the reinstatement section is
        reinstatement_start = content2.find("(h) Reinstatement")
        if reinstatement_start >= 0:
            # Extract from reinstatement to the end of chunk2
            reinstatement_content = content2[reinstatement_start:]
            print(f"\nREINSTATEMENT SECTION FROM CHUNK 2:")
            print(reinstatement_content)
            
            # Extract continuation from chunk 3 until next section
            chunk3_continuation = content3.split("(i)")[0] if "(i)" in content3 else content3[:1000]
            print(f"\nCONTINUATION FROM CHUNK 3:")
            print(chunk3_continuation)
            
            # Combine for complete reinstatement info
            complete_reinstatement = reinstatement_content + " " + chunk3_continuation
            print(f"\n{'='*50}")
            print("COMPLETE REINSTATEMENT CONTENT:")
            print(complete_reinstatement)
            print(f"{'='*50}")
            
            # Update the chunk in database with complete content
            print("\nUpdating chunk 2 with complete reinstatement content...")
            updated_content = content2.replace(content2[reinstatement_start:], complete_reinstatement)
            
            try:
                supa.table('doc_chunks').update({'content': updated_content}).eq('document_id', doc_id).eq('chunk_index', 2).execute()
                print("✓ Successfully updated chunk 2 with complete reinstatement content")
                
                # Also update the summary to include the percentages
                summary = f"Foundation membership reinstatement fees reduced by: 75% within first year, 50% within second year, 25% within third year, no reduction after three years. [Doc:{doc_id}#Chunk:2]"
                supa.table('doc_chunks').update({'summary': summary}).eq('document_id', doc_id).eq('chunk_index', 2).execute()
                print("✓ Successfully updated summary with percentage details")
                
            except Exception as e:
                print(f"Error updating database: {e}")

if __name__ == "__main__":
    extract_complete_reinstatement()