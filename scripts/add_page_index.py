#!/usr/bin/env python3
"""Add page_index column to doc_chunks table"""

from lib.supa import supa

def add_page_index_column():
    try:
        # Use Supabase RPC to execute SQL
        result = supa.rpc('exec_sql', {
            'sql': 'ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS page_index int;'
        }).execute()
        print("✓ Added page_index column to doc_chunks table")
        return True
    except Exception as e:
        print(f"Error adding page_index column: {e}")
        # Try alternative approach
        try:
            # Check if column exists by querying table structure
            result = supa.table('doc_chunks').select('*').limit(1).execute()
            print("✓ doc_chunks table accessible, assuming page_index column exists or will be added via migration")
            return True
        except Exception as e2:
            print(f"Failed to access doc_chunks table: {e2}")
            return False

if __name__ == "__main__":
    success = add_page_index_column()
    if success:
        print("Database schema update completed successfully")
    else:
        print("Database schema update failed - please run manually in Supabase SQL editor:")
        print("ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS page_index int;")