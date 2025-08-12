# Board Continuity MVP Setup Instructions

## Database Setup Required

Your application is running successfully, but you need to set up the database tables in Supabase first.

### Step 1: Create Database Tables

1. Go to your Supabase dashboard: https://supabase.com/dashboard/projects
2. Select your project: `owaaeldudvrmpisoyfsz`
3. Click on "SQL Editor" in the left sidebar
4. Run the following SQL commands:

```sql
-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table (no users table needed for MVP)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE
);

-- Document chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    embedding VECTOR(3072)
);

-- Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    citations JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_org_id ON documents(org_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_org_id ON chat_messages(org_id);

-- Function for vector similarity search
CREATE OR REPLACE FUNCTION search_document_chunks(
    query_embedding VECTOR(3072),
    document_ids UUID[],
    similarity_threshold FLOAT DEFAULT 0.7,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    content TEXT,
    page_number INT,
    document_filename TEXT,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT 
        dc.id,
        dc.document_id,
        dc.content,
        dc.page_number,
        d.filename AS document_filename,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    JOIN documents d ON dc.document_id = d.id
    WHERE dc.document_id = ANY(document_ids)
    AND 1 - (dc.embedding <=> query_embedding) > similarity_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
$$;
```

### Step 2: Test Your Application

Once the database tables are created, your application will be fully functional:

1. **Homepage**: Visit your application to see the welcome page
2. **Register**: Create a new user account
3. **Upload PDFs**: Upload your PDF documents for AI-powered chat
4. **Chat**: Ask questions about your uploaded documents
5. **AI Responses**: Get intelligent responses with citations

## Features Overview

- **PDF Upload & Processing**: Upload PDFs up to 16MB
- **AI-Powered Chat**: Chat with your documents using OpenAI GPT-4o
- **Smart Citations**: Get responses with references to specific document sections
- **Vector Search**: Advanced semantic search through document content
- **User Management**: Secure authentication and personal document libraries

## Next Steps

After setting up the database:
1. Register a new account
2. Upload a test PDF document
3. Start chatting with your documents!

The application will automatically process your PDFs, extract text, create embeddings, and enable intelligent document conversations.