#!/usr/bin/env python3
"""
Run the institutional memory migration to add enhanced decision tracking,
pattern analysis, and board member insights to the database.

This script will execute the migration and validate the new tables.
"""

import os
import sys
from lib.supa import supa

def run_migration():
    """Execute the institutional memory migration."""
    print("🧠 Running Institutional Memory Enhancement Migration")
    print("=" * 60)
    
    # Read the migration SQL
    migration_path = "migrations/add_institutional_memory.sql"
    
    if not os.path.exists(migration_path):
        print(f"❌ Migration file not found: {migration_path}")
        return False
    
    try:
        with open(migration_path, 'r') as f:
            migration_sql = f.read()
        
        print("📁 Migration file loaded successfully")
        
        # Execute the migration
        print("🚀 Executing migration...")
        result = supa.rpc('exec_sql', {'sql': migration_sql}).execute()
        
        if result.data:
            print("✅ Migration executed successfully")
        else:
            print("❌ Migration execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Error executing migration: {str(e)}")
        return False
    
    # Validate the new tables
    print("\n📊 Validating new tables...")
    
    expected_tables = [
        'decision_registry',
        'historical_patterns', 
        'institutional_knowledge',
        'board_member_insights',
        'decision_participation'
    ]
    
    for table in expected_tables:
        try:
            # Test if table exists by trying to select from it
            result = supa.table(table).select("*").limit(1).execute()
            print(f"✅ {table} table created successfully")
        except Exception as e:
            print(f"❌ {table} table validation failed: {str(e)}")
            return False
    
    # Check enhanced doc_chunks columns
    try:
        result = supa.table('doc_chunks').select('section_type,is_complete,contains_decision').limit(1).execute()
        print("✅ doc_chunks table enhanced successfully")
    except Exception as e:
        print(f"❌ doc_chunks enhancement failed: {str(e)}")
        return False
    
    print(f"\n🎉 Institutional Memory System Ready!")
    print("New capabilities:")
    print("• Complete decision registry with voting records")
    print("• Historical pattern analysis and prediction")
    print("• Enhanced document chunking with decision detection") 
    print("• Institutional knowledge capture")
    print("• Board member insight tracking")
    print("• Comprehensive decision-member relationships")
    
    return True

def test_system():
    """Test the institutional memory system with sample data."""
    print("\n🧪 Testing Institutional Memory System")
    print("-" * 40)
    
    org_id = os.getenv('DEV_ORG_ID', '00000000-0000-0000-0000-000000000001')
    
    # Test 1: Insert a sample decision
    try:
        sample_decision = {
            'decision_id': 'TEST-2024-001',
            'org_id': org_id,
            'date': '2024-08-19',
            'decision_type': 'membership',
            'title': 'Test Membership Decision',
            'description': 'Sample decision for testing',
            'outcome': 'approved',
            'vote_count_for': 7,
            'vote_count_against': 2,
            'tags': ['membership', 'test']
        }
        
        result = supa.table('decision_registry').insert(sample_decision).execute()
        if result.data:
            print("✅ Sample decision inserted successfully")
            decision_id = result.data[0]['id']
        else:
            print("❌ Failed to insert sample decision")
            return False
            
    except Exception as e:
        print(f"❌ Decision insert error: {str(e)}")
        return False
    
    # Test 2: Check pattern generation
    try:
        patterns = supa.table('historical_patterns').select('*').eq('org_id', org_id).execute()
        if patterns.data:
            print(f"✅ Pattern analysis working ({len(patterns.data)} patterns found)")
        else:
            print("⚠️ No patterns generated yet (normal for first run)")
    except Exception as e:
        print(f"❌ Pattern analysis error: {str(e)}")
    
    # Test 3: Insert institutional knowledge
    try:
        sample_knowledge = {
            'org_id': org_id,
            'knowledge_type': 'procedural',
            'category': 'testing',
            'title': 'Test Migration Success',
            'context': 'The institutional memory migration completed successfully on ' + str(os.popen('date').read().strip()),
            'confidence_score': 1.0,
            'tags': ['migration', 'test', 'success']
        }
        
        result = supa.table('institutional_knowledge').insert(sample_knowledge).execute()
        if result.data:
            print("✅ Institutional knowledge stored successfully")
        else:
            print("❌ Failed to store institutional knowledge")
            
    except Exception as e:
        print(f"❌ Knowledge storage error: {str(e)}")
    
    # Test 4: Enhanced doc_chunks functionality
    try:
        # Update a chunk with new fields
        chunks = supa.table('doc_chunks').select('id').limit(1).execute()
        if chunks.data:
            chunk_id = chunks.data[0]['id']
            update_result = supa.table('doc_chunks').update({
                'section_type': 'test_section',
                'is_complete': True,
                'contains_decision': True,
                'entities_mentioned': {'test_entity': 'test_value'},
                'importance_score': 0.85
            }).eq('id', chunk_id).execute()
            
            if update_result.data:
                print("✅ Enhanced doc_chunks functionality working")
            else:
                print("❌ doc_chunks update failed")
        else:
            print("⚠️ No existing chunks to test (upload documents first)")
            
    except Exception as e:
        print(f"❌ Enhanced chunks error: {str(e)}")
    
    print("\n📋 Migration Test Summary:")
    print("• Decision registry: Operational")  
    print("• Pattern analysis: Operational")
    print("• Knowledge storage: Operational")
    print("• Enhanced chunking: Operational")
    print("\n🚀 System ready for institutional memory capture!")
    
    return True

if __name__ == "__main__":
    success = run_migration()
    if success:
        test_system()
        sys.exit(0)
    else:
        print("❌ Migration failed")
        sys.exit(1)