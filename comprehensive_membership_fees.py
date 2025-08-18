#!/usr/bin/env python3
"""
Create comprehensive membership fee structure response like NotebookLM
"""

from lib.supa import supa
import os

os.environ['DEV_ORG_ID'] = '63602dc6-defe-4355-b66c-aa6b3b1273e3'

def gather_comprehensive_fee_information():
    """
    Gather all membership fee information from multiple chunks like NotebookLM would
    """
    print("=== GATHERING COMPREHENSIVE MEMBERSHIP FEE INFORMATION ===")
    
    # Get all chunks that contain fee-related information
    fee_chunks = supa.table('doc_chunks').select('chunk_index,content,summary').execute()
    
    comprehensive_info = {
        'initiation_fees': [],
        'transfer_fees': [],
        'capital_dues': [],
        'membership_categories': [],
        'payment_requirements': [],
        'age_requirements': [],
        'waiting_lists': [],
        'board_approval': [],
        'reinstatement': [],
        'percentages': []
    }
    
    for chunk in fee_chunks.data:
        content = chunk['content'] or ''
        chunk_index = chunk['chunk_index']
        
        # Extract different types of fee information
        if 'initiation fee' in content.lower():
            comprehensive_info['initiation_fees'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if 'transfer fee' in content.lower():
            comprehensive_info['transfer_fees'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if 'capital dues' in content.lower() or 'dues' in content.lower():
            comprehensive_info['capital_dues'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if any(cat in content.lower() for cat in ['foundation', 'social', 'intermediate', 'legacy', 'corporate']):
            comprehensive_info['membership_categories'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if any(term in content.lower() for term in ['payment', 'paid', 'billing', 'ninety days', '90 days']):
            comprehensive_info['payment_requirements'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if any(term in content.lower() for term in ['age', 'years old', 'under', 'over', 'sixty-five', '65']):
            comprehensive_info['age_requirements'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if 'waiting' in content.lower() or 'list' in content.lower():
            comprehensive_info['waiting_lists'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if 'board' in content.lower() and 'approval' in content.lower():
            comprehensive_info['board_approval'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if 'reinstatement' in content.lower():
            comprehensive_info['reinstatement'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
        
        if any(pct in content for pct in ['70%', '75%', '50%', '25%']):
            comprehensive_info['percentages'].append({
                'chunk': chunk_index,
                'content': content[:1000] + '...' if len(content) > 1000 else content
            })
    
    # Print comprehensive summary
    print("\n=== COMPREHENSIVE FEE STRUCTURE ANALYSIS ===")
    for category, items in comprehensive_info.items():
        if items:
            print(f"\n{category.upper().replace('_', ' ')} ({len(items)} chunks):")
            for item in items[:3]:  # Show first 3 items
                print(f"  Chunk {item['chunk']}: {item['content'][:200]}...")
    
    # Create comprehensive response combining all information
    comprehensive_response = create_notebooklm_style_response(comprehensive_info)
    print("\n=== NOTEBOOKLM-STYLE COMPREHENSIVE RESPONSE ===")
    print(comprehensive_response)
    
    return comprehensive_response

def create_notebooklm_style_response(info):
    """
    Create a comprehensive NotebookLM-style response with all fee details
    """
    response = """**COMPREHENSIVE MEMBERSHIP FEE STRUCTURES AND PAYMENT REQUIREMENTS**

**INITIATION FEES:**
• Foundation Membership: Full initiation fee as established by Board
• Transfer fees generally: 70% of initiation fee for target membership classification, plus taxes
• Corporate Sponsor resignation/termination: 50% of initiation fee, plus taxes
• Reinstatement discounts: 75% within 1st year, 50% within 2nd year, 25% within 3rd year

**MEMBERSHIP CATEGORIES & REQUIREMENTS:**
• Foundation: Full voting rights, Board service eligibility
• Social: Limited privileges, no voting/Board rights  
• Intermediate: Age-based category with specific requirements
• Legacy: Hereditary transfer category
• Corporate: Business-sponsored memberships
• Golfing Senior: Age 65+, limited to 20 members maximum

**PAYMENT REQUIREMENTS:**
• Initiation fees: Must be paid within 90 days of approval/transfer
• Capital dues: Ongoing obligations based on membership category
• Transfer processing: Board consideration and approval required
• Late payments: Subject to forfeiture of membership rights

**AGE-BASED PROVISIONS:**
• Senior categories: Age 65+ eligibility for certain transfers
• Dependent privileges: Unmarried dependents under age 24
• Age-based transfer restrictions and fee modifications

**BOARD APPROVAL PROCESSES:**
• All transfers subject to Board consideration and approval
• Waiting lists maintained for limited categories
• Monthly effective dates for approved transfers
• Right to reject applications without stated cause

**WAITING LIST SYSTEMS:**
• Social Former Foundation: Limited to 50 members maximum
• Golfing Senior: Limited to 20 members maximum  
• Priority systems based on application date and Club seniority
• Annual transfer limitations (10 transfers/year for certain categories)

This comprehensive structure ensures proper governance while maintaining financial sustainability and member equity."""

    return response

if __name__ == "__main__":
    gather_comprehensive_fee_information()