#!/usr/bin/env python3
"""
Test script to validate the enhanced chunking system.
This specifically tests the IHCC rules document to ensure ALL reinstatement percentages
(75%, 50%, 25%) are captured correctly in a single chunk.
"""

import requests
import json

def test_document_processing():
    """Test document processing and validation."""
    
    print("🔍 Testing Enhanced Document Processing System")
    print("=" * 60)
    
    # Step 1: Get list of documents
    print("\n1. Fetching document list...")
    docs_response = requests.get("http://localhost:5000/docs")
    
    if docs_response.status_code != 200:
        print(f"❌ Failed to fetch documents: {docs_response.status_code}")
        return
    
    docs_data = docs_response.json()
    documents = docs_data.get("docs", [])
    
    if not documents:
        print("⚠️  No documents found. Please upload the IHCC rules document first.")
        return
    
    print(f"✅ Found {len(documents)} document(s)")
    
    # Step 2: Find IHCC rules document
    ihcc_doc = None
    for doc in documents:
        if "ihcc" in doc.get("filename", "").lower() or "rules" in doc.get("filename", "").lower():
            ihcc_doc = doc
            break
    
    if not ihcc_doc:
        print("⚠️  IHCC rules document not found. Using first available document for testing.")
        ihcc_doc = documents[0]
    
    print(f"📄 Testing document: {ihcc_doc.get('filename', 'Unknown')}")
    
    # Step 3: Test validation endpoint
    doc_id = ihcc_doc["id"]
    print(f"\n2. Validating document processing for doc_id: {doc_id}")
    
    validation_response = requests.get(f"http://localhost:5000/validate/{doc_id}")
    
    if validation_response.status_code != 200:
        print(f"❌ Validation failed: {validation_response.status_code}")
        print(f"Response: {validation_response.text}")
        return
    
    validation_data = validation_response.json()
    
    if not validation_data.get("ok"):
        print(f"❌ Validation error: {validation_data.get('error')}")
        return
    
    analysis = validation_data.get("analysis", {})
    
    # Step 4: Display results
    print("\n3. Validation Results:")
    print("-" * 30)
    
    print(f"Status: {analysis.get('status', 'unknown').upper()}")
    print(f"Total chunks: {analysis.get('total_chunks', 0)}")
    print(f"Reinstatement chunks: {analysis.get('reinstatement_chunks', 0)}")
    print(f"Chunks with percentages: {analysis.get('chunks_with_percentages', 0)}")
    print(f"Highest completeness score: {analysis.get('highest_completeness_score', 0):.1f}")
    
    required = analysis.get("required_percentages", [])
    found = analysis.get("found_percentages", [])
    missing = analysis.get("missing_percentages", [])
    
    print(f"\nPercentage Analysis:")
    print(f"Required: {', '.join(required)}")
    print(f"Found: {', '.join(found)}")
    
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
    else:
        print("✅ All required percentages found!")
    
    # Step 5: Show best chunk details
    if "best_reinstatement_chunk" in analysis:
        best = analysis["best_reinstatement_chunk"]
        print(f"\nBest Reinstatement Chunk:")
        print(f"Completeness score: {best.get('completeness_score', 0):.1f}")
        print(f"Contains all percentages: {'✅' if best.get('contains_all_percentages') else '❌'}")
        print(f"Content preview:\n{best.get('content_preview', 'No preview')}")
    
    # Step 6: Test chat functionality
    print(f"\n4. Testing chat functionality...")
    
    chat_response = requests.post(
        "http://localhost:5000/chat",
        json={"q": "What are the reinstatement percentages and requirements?"}
    )
    
    if chat_response.status_code == 200:
        chat_data = chat_response.json()
        if chat_data.get("ok"):
            markdown = chat_data.get("markdown", "")
            citations = chat_data.get("citations", [])
            
            print("✅ Chat response received")
            print(f"Response length: {len(markdown)} characters")
            print(f"Citations: {len(citations)}")
            
            # Check if response contains the percentages
            percentages_in_response = []
            for perc in ["75%", "50%", "25%"]:
                if perc in markdown:
                    percentages_in_response.append(perc)
            
            print(f"Percentages in response: {', '.join(percentages_in_response) if percentages_in_response else 'None found'}")
            
        else:
            print(f"❌ Chat error: {chat_data.get('error')}")
    else:
        print(f"❌ Chat request failed: {chat_response.status_code}")
    
    # Final assessment
    print(f"\n" + "=" * 60)
    if analysis.get("status") == "success" and not missing:
        print("🎉 ENHANCED CHUNKING SYSTEM: WORKING CORRECTLY!")
        print("All reinstatement percentages are properly captured.")
    else:
        print("⚠️  ENHANCED CHUNKING SYSTEM: NEEDS IMPROVEMENT")
        print("Some reinstatement percentages may not be captured correctly.")
    print("=" * 60)

if __name__ == "__main__":
    test_document_processing()