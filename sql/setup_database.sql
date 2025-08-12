-- Board Continuity MVP Database Schema for Supabase
-- Run this SQL in your Supabase SQL Editor

-- Enable pgvector extension for vector similarity search
create extension if not exists vector;

-- Users table
create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    email text unique not null,
    password_hash text not null,
    name text not null,
    created_at timestamptz default now()
);

-- Documents table
create table if not exists documents (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references users(id) on delete cascade,
    filename text not null,
    file_path text not null,
    uploaded_at timestamptz default now(),
    processed boolean default false
);

-- Document chunks table with vector embeddings
create table if not exists document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    content text not null,
    chunk_index integer not null,
    page_number integer,
    embedding vector(3072) -- text-embedding-3-large has 3072 dimensions
);

-- Chat messages table
create table if not exists chat_messages (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references users(id) on delete cascade,
    message text not null,
    response text not null,
    citations jsonb,
    created_at timestamptz default now()
);

-- Create indexes for better performance
create index if not exists idx_documents_user_id on documents(user_id);
create index if not exists idx_document_chunks_document_id on document_chunks(document_id);
create index if not exists idx_chat_messages_user_id on chat_messages(user_id);
create index if not exists idx_document_chunks_embedding on document_chunks using ivfflat (embedding vector_cosine_ops);

-- Function for vector similarity search
create or replace function search_document_chunks(
    query_embedding vector(3072),
    document_ids uuid[],
    similarity_threshold float default 0.7,
    match_count int default 5
)
returns table (
    id uuid,
    document_id uuid,
    content text,
    page_number int,
    document_filename text,
    similarity float
)
language sql stable
as $$
    select 
        dc.id,
        dc.document_id,
        dc.content,
        dc.page_number,
        d.filename as document_filename,
        1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    join documents d on dc.document_id = d.id
    where dc.document_id = any(document_ids)
    and 1 - (dc.embedding <=> query_embedding) > similarity_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;

-- Row Level Security (RLS) policies
alter table users enable row level security;
alter table documents enable row level security;
alter table document_chunks enable row level security;
alter table chat_messages enable row level security;

-- Users can only see their own data
create policy "Users can view own profile" on users for select using (auth.uid()::text = id::text);
create policy "Users can update own profile" on users for update using (auth.uid()::text = id::text);

create policy "Users can view own documents" on documents for select using (auth.uid()::text = user_id::text);
create policy "Users can insert own documents" on documents for insert with check (auth.uid()::text = user_id::text);
create policy "Users can update own documents" on documents for update using (auth.uid()::text = user_id::text);
create policy "Users can delete own documents" on documents for delete using (auth.uid()::text = user_id::text);

create policy "Users can view chunks of own documents" on document_chunks for select using (
    exists (
        select 1 from documents d 
        where d.id = document_chunks.document_id 
        and d.user_id::text = auth.uid()::text
    )
);
create policy "Users can insert chunks for own documents" on document_chunks for insert with check (
    exists (
        select 1 from documents d 
        where d.id = document_chunks.document_id 
        and d.user_id::text = auth.uid()::text
    )
);
create policy "Users can delete chunks of own documents" on document_chunks for delete using (
    exists (
        select 1 from documents d 
        where d.id = document_chunks.document_id 
        and d.user_id::text = auth.uid()::text
    )
);

create policy "Users can view own chat messages" on chat_messages for select using (auth.uid()::text = user_id::text);
create policy "Users can insert own chat messages" on chat_messages for insert with check (auth.uid()::text = user_id::text);