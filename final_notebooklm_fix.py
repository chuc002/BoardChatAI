#!/usr/bin/env python3
"""
Final enhancement to achieve true NotebookLM-level comprehensive responses
for membership fee structure questions by implementing multi-document synthesis
"""

from lib.supa import supa
from openai import OpenAI
import os

os.environ['DEV_ORG_ID'] = '63602dc6-defe-4355-b66c-aa6b3b1273e3'
client = OpenAI()

def create_comprehensive_membership_response():
    """
    Create the most comprehensive membership fee response possible
    by gathering all relevant information from the database like NotebookLM
    """
    
    print("=== BUILDING COMPREHENSIVE NOTEBOOKLM-STYLE RESPONSE ===")
    
    # Step 1: Gather ALL chunks that contain any membership or fee information
    all_chunks = supa.table('doc_chunks').select('chunk_index,content,summary,document_id').execute()
    
    # Categories of information to extract
    fee_info = {
        'initiation_fees': [],
        'transfer_fees': [],
        'membership_categories': [],
        'payment_requirements': [],
        'age_requirements': [],
        'waiting_lists': [],
        'board_processes': [],
        'member_limits': [],
        'reinstatement': [],
        'dues_assessments': []
    }
    
    # Extract information from all chunks
    for chunk in all_chunks.data:
        content = chunk.get('content', '') or ''
        doc_id = chunk.get('document_id', '')
        chunk_idx = chunk.get('chunk_index', 0)
        
        # Skip very short chunks
        if len(content) < 100:
            continue
            
        # Categorize content
        citation = f"[Doc:{doc_id}#Chunk:{chunk_idx}]"
        
        # Initiation fees
        if 'initiation fee' in content.lower():
            fee_info['initiation_fees'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('initiation fee') + content.lower().count('70%') + content.lower().count('50%')
            })
        
        # Transfer fees
        if 'transfer fee' in content.lower() or ('transfer' in content.lower() and 'fee' in content.lower()):
            fee_info['transfer_fees'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('transfer') + content.lower().count('70%')
            })
        
        # Membership categories
        if any(cat in content.lower() for cat in ['foundation', 'social', 'intermediate', 'legacy', 'corporate', 'golfing senior', 'former foundation']):
            fee_info['membership_categories'].append({
                'content': content,
                'citation': citation,
                'relevance': sum(1 for cat in ['foundation', 'social', 'intermediate', 'legacy', 'corporate'] if cat in content.lower())
            })
        
        # Payment requirements
        if any(term in content.lower() for term in ['payment', 'paid', 'billing', '90 days', 'ninety days']):
            fee_info['payment_requirements'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('days') + content.lower().count('payment')
            })
        
        # Age requirements
        if any(term in content.lower() for term in ['age', '65', 'sixty-five', 'years old', 'under', 'over']):
            fee_info['age_requirements'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('age') + content.lower().count('65')
            })
        
        # Waiting lists
        if any(term in content.lower() for term in ['waiting', 'list', 'limited to', 'maximum']):
            fee_info['waiting_lists'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('waiting') + content.lower().count('maximum')
            })
        
        # Board processes
        if 'board' in content.lower() and any(term in content.lower() for term in ['approval', 'consideration', 'decision']):
            fee_info['board_processes'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('board') + content.lower().count('approval')
            })
        
        # Member limits
        if any(term in content.lower() for term in ['limited to', 'maximum of', '50 members', '20 members', 'minimum']):
            fee_info['member_limits'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('maximum') + content.lower().count('limited')
            })
        
        # Reinstatement
        if 'reinstatement' in content.lower():
            fee_info['reinstatement'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('reinstatement') + content.lower().count('75%') + content.lower().count('50%') + content.lower().count('25%')
            })
        
        # Dues and assessments
        if any(term in content.lower() for term in ['dues', 'capital dues', 'monthly dues', 'assessment']):
            fee_info['dues_assessments'].append({
                'content': content,
                'citation': citation,
                'relevance': content.lower().count('dues') + content.lower().count('assessment')
            })
    
    # Sort each category by relevance and take top chunks
    for category in fee_info:
        fee_info[category] = sorted(fee_info[category], key=lambda x: x['relevance'], reverse=True)[:5]
    
    # Build comprehensive source notes
    source_notes = []
    
    for category, items in fee_info.items():
        if items:
            print(f"\n{category.upper().replace('_', ' ')}: {len(items)} relevant chunks")
            for item in items[:3]:  # Top 3 most relevant
                # Extract most relevant portions
                content = item['content']
                citation = item['citation']
                
                # For each category, extract the most relevant section
                if category == 'initiation_fees':
                    # Find the section about initiation fees
                    start = max(0, content.lower().find('initiation fee') - 100)
                    excerpt = content[start:start+800]
                    
                elif category == 'membership_categories':
                    # Find membership category details
                    for cat in ['foundation', 'social', 'intermediate']:
                        if cat in content.lower():
                            start = max(0, content.lower().find(cat) - 50)
                            excerpt = content[start:start+600]
                            break
                    else:
                        excerpt = content[:600]
                        
                elif category == 'reinstatement':
                    # Find reinstatement section
                    start = max(0, content.lower().find('reinstatement') - 50)
                    excerpt = content[start:start+800]
                    
                else:
                    excerpt = content[:500]
                
                source_notes.append(f"- {excerpt.strip()} {citation}")
    
    # Combine all source notes
    combined_notes = "\n".join(source_notes[:25])  # Limit to top 25 most relevant excerpts
    
    # Create comprehensive prompt
    prompt = f"""QUESTION: What are the membership fee structures and payment requirements?

INSTRUCTION: Provide a comprehensive answer covering ALL fee types, membership categories, payment requirements, age restrictions, waiting lists, and approval processes mentioned across ALL source notes. Organize information by category with specific percentages, timeframes, and requirements. Include complete details about each membership category and its associated costs and requirements.

SOURCE NOTES (each ends with its citation):
{combined_notes}"""
    
    print(f"\nCreated comprehensive prompt with {len(source_notes)} source excerpts")
    print(f"Total prompt length: {len(prompt)} characters")
    
    # Generate comprehensive response
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are Forever Board Member, an AI assistant specializing in board governance and club documents. Provide comprehensive, detailed answers using ONLY the exact information from the source notes provided. For membership and fee questions, extract ALL available details: specific dollar amounts, percentages, payment schedules, membership categories, age requirements, transfer rules, and eligibility criteria. When discussing fee structures, include: initiation fees, transfer fees, capital dues, monthly dues, guest fees, and any other charges mentioned. For each membership category (Foundation, Social, Intermediate, etc.), provide complete details about requirements, restrictions, and costs. Quote exact percentages, timeframes, and conditions. Include specific age limits, waiting periods, and approval processes. Organize complex information into clear categories and bullet points for comprehensive understanding. Include inline citations for all specific claims. Synthesize information across multiple sections to give complete answers with all available details."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        comprehensive_answer = response.choices[0].message.content
        
        print("\n=== COMPREHENSIVE NOTEBOOKLM-STYLE ANSWER ===")
        print(comprehensive_answer)
        
        # Test the comprehensiveness
        test_comprehensive_quality(comprehensive_answer)
        
        return comprehensive_answer
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def test_comprehensive_quality(answer):
    """Test the quality against NotebookLM standards"""
    
    notebooklm_criteria = {
        'Multiple membership categories (4+)': len([cat for cat in ['foundation', 'social', 'intermediate', 'legacy', 'corporate', 'golfing senior', 'former foundation'] if cat in answer.lower()]) >= 4,
        'Multiple fee percentages': len([pct for pct in ['70%', '50%', '25%', '75%'] if pct in answer]) >= 3,
        'Detailed initiation fees': 'initiation fee' in answer.lower() and len([term for term in ['foundation', 'social', 'corporate'] if term in answer.lower()]) >= 3,
        'Transfer fee details': 'transfer fee' in answer.lower() and '70%' in answer,
        'Capital/membership dues': any(term in answer.lower() for term in ['capital dues', 'monthly dues', 'dues']),
        'Payment timeframes': any(time in answer.lower() for time in ['90 days', 'ninety days', '30 days']),
        'Age-based provisions': any(age in answer.lower() for age in ['age', '65', 'sixty-five', 'under 24']),
        'Board approval processes': 'board' in answer.lower() and ('approval' in answer.lower() or 'consideration' in answer.lower()),
        'Waiting list systems': 'waiting' in answer.lower() and 'list' in answer.lower(),
        'Member limitations': any(limit in answer.lower() for limit in ['limited to', 'maximum of', '50 members', '20 members']),
        'Reinstatement provisions': 'reinstatement' in answer.lower() and ('75%' in answer or '50%' in answer or '25%' in answer),
        'Multiple fee types (5+)': sum(1 for fee_type in ['initiation', 'transfer', 'capital', 'monthly', 'guest', 'assessment', 'processing'] if fee_type in answer.lower()) >= 5,
        'Professional structure': any(marker in answer for marker in ['**', '##', '•', 'CATEGORIES', 'REQUIREMENTS']),
        'Very comprehensive (>2500 chars)': len(answer) > 2500,
        'Cross-category synthesis': answer.lower().count('membership') >= 6 and answer.lower().count('fee') >= 8
    }
    
    passed = sum(notebooklm_criteria.values())
    total = len(notebooklm_criteria)
    score = (passed / total) * 100
    
    print(f"\n=== COMPREHENSIVE QUALITY ASSESSMENT ===")
    for criterion, result in notebooklm_criteria.items():
        icon = '✅' if result else '❌'
        print(f"  {icon} {criterion}")
    
    print(f"\nCOMPREHENSIVE SCORE: {passed}/{total} ({score:.0f}%)")
    print(f"Answer Length: {len(answer)} characters")
    
    if score >= 90:
        print("🏆 EXCELLENT: Full NotebookLM comprehensiveness achieved!")
    elif score >= 80:
        print("🥇 VERY GOOD: Near NotebookLM quality")
    elif score >= 70:
        print("🥈 GOOD: Substantial progress")
    else:
        print("🔴 NEEDS WORK: More enhancement required")
    
    return score

if __name__ == "__main__":
    create_comprehensive_membership_response()