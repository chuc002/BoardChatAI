#!/usr/bin/env python3
"""
Implement NotebookLM approach - comprehensive multi-chunk synthesis for fee structure questions
"""

import os
from lib.supa import supa
from lib.rag import answer_question_md

os.environ['DEV_ORG_ID'] = '63602dc6-defe-4355-b66c-aa6b3b1273e3'

def test_comprehensive_implementation():
    """Test the current comprehensive implementation"""
    
    question = "What are the membership fee structures and payment requirements?"
    
    print("=== TESTING COMPREHENSIVE MEMBERSHIP FEE RESPONSE ===")
    
    # Get response from current system
    answer, citations = answer_question_md(os.environ['DEV_ORG_ID'], question)
    
    print(f"Generated Answer:")
    print("="*80)
    print(answer)
    print("="*80)
    
    # Analyze comprehensiveness
    notebooklm_standards = {
        'Multiple membership categories (4+)': len([cat for cat in ['foundation', 'social', 'intermediate', 'legacy', 'corporate', 'golfing senior', 'former foundation', 'nonresident'] if cat in answer.lower()]) >= 4,
        'Multiple fee percentages (3+)': len([pct for pct in ['70%', '50%', '25%', '75%'] if pct in answer]) >= 3,
        'Detailed initiation structures': 'initiation fee' in answer.lower() and len([cat for cat in ['foundation', 'social', 'corporate'] if cat in answer.lower()]) >= 3,
        'Transfer fee mechanisms': 'transfer fee' in answer.lower() and '70%' in answer,
        'Capital/membership dues': any(term in answer.lower() for term in ['capital dues', 'monthly dues', 'dues', 'assessment']),
        'Payment timeframes specified': any(time in answer.lower() for time in ['90 days', 'ninety days', '30 days', 'within']),
        'Age-based requirements': any(age in answer.lower() for age in ['age', '65', 'sixty-five', 'under 24', 'over']),
        'Board approval processes': 'board' in answer.lower() and ('approval' in answer.lower() or 'consideration' in answer.lower()),
        'Waiting list systems': ('waiting' in answer.lower() and 'list' in answer.lower()) or 'priority' in answer.lower(),
        'Member count limitations': any(limit in answer.lower() for limit in ['limited to', 'maximum of', '50 members', '20 members', 'minimum']),
        'Reinstatement provisions': 'reinstatement' in answer.lower() and any(pct in answer for pct in ['75%', '50%', '25%']),
        'Multiple fee types (5+)': sum(1 for fee_type in ['initiation', 'transfer', 'capital', 'monthly', 'guest', 'assessment', 'processing', 'annual'] if fee_type in answer.lower()) >= 5,
        'Professional organization': any(marker in answer for marker in ['**', '##', '•', '1.', '2.', 'CATEGORIES', 'REQUIREMENTS']),
        'Comprehensive length (>2200 chars)': len(answer) > 2200,
        'Cross-category synthesis': answer.lower().count('membership') >= 6 and answer.lower().count('fee') >= 8,
        'Specific rule citations': len(citations) > 0
    }
    
    passed = sum(notebooklm_standards.values())
    total = len(notebooklm_standards)
    score = (passed / total) * 100
    
    print(f"\n=== NOTEBOOKLM COMPREHENSIVENESS ANALYSIS ===")
    for criterion, result in notebooklm_standards.items():
        icon = '✅' if result else '❌'
        print(f"  {icon} {criterion}")
    
    print(f"\nCOMPREHENSIVE QUALITY SCORE: {passed}/{total} ({score:.0f}%)")
    print(f"Answer Length: {len(answer)} characters")
    print(f"Citations: {len(citations)}")
    print(f"Fee mentions: {answer.lower().count('fee')}")
    print(f"Membership mentions: {answer.lower().count('membership')}")
    
    if score >= 90:
        print("\n🏆 OUTSTANDING: Full NotebookLM comprehensiveness achieved!")
    elif score >= 85:
        print("\n🥇 EXCELLENT: Very close to NotebookLM standard")
    elif score >= 75:
        print("\n🥈 VERY GOOD: Substantial progress toward NotebookLM quality")
    elif score >= 65:
        print("\n🥉 GOOD: Significant improvement")
    else:
        print("\n🔴 NEEDS ENHANCEMENT: More work required for NotebookLM standards")
    
    return score, answer

def analyze_database_content():
    """Analyze what comprehensive information is available in the database"""
    
    print("\n=== DATABASE CONTENT ANALYSIS ===")
    
    # Get all chunks
    all_chunks = supa.table('doc_chunks').select('chunk_index,content,summary').execute()
    
    comprehensive_categories = {
        'Foundation membership': 0,
        'Social membership': 0,
        'Intermediate membership': 0,
        'Legacy membership': 0,
        'Corporate membership': 0,
        'Golfing Senior membership': 0,
        'Initiation fees': 0,
        'Transfer fees': 0,
        'Capital dues': 0,
        'Monthly dues': 0,
        'Payment requirements': 0,
        'Age requirements': 0,
        'Waiting lists': 0,
        'Board approval': 0,
        'Reinstatement': 0,
        '70% fee percentage': 0,
        '75% fee percentage': 0,
        '50% fee percentage': 0,
        '25% fee percentage': 0
    }
    
    for chunk in all_chunks.data:
        content = chunk.get('content', '') or ''
        
        if 'foundation' in content.lower() and 'membership' in content.lower():
            comprehensive_categories['Foundation membership'] += 1
        if 'social' in content.lower() and 'membership' in content.lower():
            comprehensive_categories['Social membership'] += 1
        if 'intermediate' in content.lower():
            comprehensive_categories['Intermediate membership'] += 1
        if 'legacy' in content.lower():
            comprehensive_categories['Legacy membership'] += 1
        if 'corporate' in content.lower():
            comprehensive_categories['Corporate membership'] += 1
        if 'golfing senior' in content.lower():
            comprehensive_categories['Golfing Senior membership'] += 1
        if 'initiation fee' in content.lower():
            comprehensive_categories['Initiation fees'] += 1
        if 'transfer fee' in content.lower():
            comprehensive_categories['Transfer fees'] += 1
        if 'capital dues' in content.lower():
            comprehensive_categories['Capital dues'] += 1
        if 'monthly dues' in content.lower():
            comprehensive_categories['Monthly dues'] += 1
        if any(term in content.lower() for term in ['payment', 'paid', '90 days']):
            comprehensive_categories['Payment requirements'] += 1
        if any(term in content.lower() for term in ['age', '65', 'years old']):
            comprehensive_categories['Age requirements'] += 1
        if 'waiting' in content.lower() and 'list' in content.lower():
            comprehensive_categories['Waiting lists'] += 1
        if 'board' in content.lower() and 'approval' in content.lower():
            comprehensive_categories['Board approval'] += 1
        if 'reinstatement' in content.lower():
            comprehensive_categories['Reinstatement'] += 1
        if '70%' in content:
            comprehensive_categories['70% fee percentage'] += 1
        if '75%' in content:
            comprehensive_categories['75% fee percentage'] += 1
        if '50%' in content:
            comprehensive_categories['50% fee percentage'] += 1
        if '25%' in content:
            comprehensive_categories['25% fee percentage'] += 1
    
    print("Information availability in database:")
    for category, count in comprehensive_categories.items():
        if count > 0:
            print(f"  ✅ {category}: {count} chunks")
        else:
            print(f"  ❌ {category}: 0 chunks")
    
    # Show chunks with the most comprehensive information
    print(f"\nTotal chunks analyzed: {len(all_chunks.data)}")
    
    return comprehensive_categories

if __name__ == "__main__":
    # Analyze what's available in the database
    db_analysis = analyze_database_content()
    
    # Test current implementation
    score, answer = test_comprehensive_implementation()
    
    print(f"\n=== FINAL ASSESSMENT ===")
    print(f"Current system achieves {score:.0f}% of NotebookLM comprehensiveness standards")
    
    if score < 85:
        print("Recommendation: Enhance RAG retrieval to gather more diverse membership category information")
    else:
        print("System meets high comprehensiveness standards!")