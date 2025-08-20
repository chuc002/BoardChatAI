#!/usr/bin/env python3
"""
Fix unprocessed documents to ensure all uploaded documents are searchable.
This script downloads unprocessed documents from Supabase storage and processes them.
"""

import os
from lib.supa import supa, signed_url_for
from lib.ingest import upsert_document
import requests

def fix_unprocessed_documents():
    """Download and process all unprocessed documents"""
    
    print("🔧 FIXING UNPROCESSED DOCUMENTS")
    print("=" * 50)
    
    # Get all unprocessed documents
    unprocessed_docs = supa.table('documents').select(
        'id, title, filename, storage_path'
    ).eq('org_id', '63602dc6-defe-4355-b66c-aa6b3b1273e3').eq('processed', False).execute()
    
    if not unprocessed_docs.data:
        print("✅ All documents already processed!")
        return
        
    print(f"Found {len(unprocessed_docs.data)} unprocessed documents")
    
    for doc in unprocessed_docs.data:
        print(f"\n📄 Processing: {doc['title']}")
        
        try:
            # Update status to processing
            supa.table('documents').update({'status': 'processing'}).eq('id', doc['id']).execute()
            
            # Get signed URL for the document
            signed_url = signed_url_for(doc['id'])
            if not signed_url:
                print(f"   ❌ ERROR: Could not get signed URL for document")
                continue
            
            print(f"   📥 Downloading document from storage...")
            
            # Download the document
            response = requests.get(signed_url)
            if response.status_code != 200:
                print(f"   ❌ ERROR: Failed to download document (status: {response.status_code})")
                continue
                
            file_bytes = response.content
            print(f"   📊 Downloaded {len(file_bytes)} bytes")
            
            # Process the document using the correct function signature
            result = upsert_document(
                org_id='63602dc6-defe-4355-b66c-aa6b3b1273e3',
                filename=doc['filename'],
                file_bytes=file_bytes,
                mime_type='application/pdf'
            )
            
            print(f"   🧠 Processing result: {result}")
            
            # Mark as completed
            supa.table('documents').update({
                'status': 'ready', 
                'processed': True
            }).eq('id', doc['id']).execute()
            
            print(f"   ✅ SUCCESS: Document processed successfully!")
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            # Mark as error for manual investigation
            supa.table('documents').update({'status': 'error'}).eq('id', doc['id']).execute()
    
    # Final status check
    print("\n" + "="*60)
    print("📊 FINAL DOCUMENT STATUS")
    print("="*60)
    
    all_docs = supa.table('documents').select(
        'id, title, status, processed, created_at'
    ).eq('org_id', '63602dc6-defe-4355-b66c-aa6b3b1273e3').order('created_at', desc=True).execute()
    
    searchable_count = 0
    
    for i, doc in enumerate(all_docs.data, 1):
        # Count chunks for each document
        chunks = supa.table('doc_chunks').select('id').eq('document_id', doc['id']).execute()
        chunk_count = len(chunks.data) if chunks.data else 0
        
        if chunk_count > 0:
            searchable_count += 1
            status_icon = "✅"
            searchable_text = "SEARCHABLE"
        else:
            status_icon = "❌"
            searchable_text = "NOT SEARCHABLE"
        
        print(f"{i}. {status_icon} {doc['title']}")
        print(f"   Status: {doc['status']} | Processed: {doc['processed']}")
        print(f"   Chunks: {chunk_count} | {searchable_text}")
        print()
    
    # Summary
    total_docs = len(all_docs.data)
    print(f"📈 SUMMARY:")
    print(f"   Total documents uploaded: {total_docs}")
    print(f"   Documents that AI can reference: {searchable_count}")
    print(f"   Documents that AI cannot reference: {total_docs - searchable_count}")
    
    if searchable_count == total_docs:
        print(f"\n🎉 SUCCESS: All {total_docs} documents are now searchable!")
        print("   The AI system can now reference information from ALL uploaded documents.")
    else:
        print(f"\n⚠️ WARNING: {total_docs - searchable_count} documents are still not searchable.")
        print("   The AI system cannot reference information from these documents.")

if __name__ == "__main__":
    fix_unprocessed_documents()