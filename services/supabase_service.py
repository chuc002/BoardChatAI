import os
import logging
import uuid
from datetime import datetime
from supabase import create_client, Client
from models import User, Document, DocumentChunk

def _get_env(name: str, *aliases: str, required: bool = True) -> str | None:
    for key in (name, *aliases):
        val = os.getenv(key)
        if val:
            return val
    if required:
        raise ValueError(f"Missing required environment variable(s): {name}" + (f" (aliases: {', '.join(aliases)})" if aliases else ""))
    return None

# Primary names + common aliases
SUPABASE_URL = _get_env("SUPABASE_URL", "SUPABASE_PROJECT_URL")
SUPABASE_SERVICE_ROLE = _get_env("SUPABASE_SERVICE_ROLE", "SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_ANON_KEY = _get_env("SUPABASE_ANON_KEY", required=False)

# Dev environment user settings
DEV_ORG_ID = _get_env("DEV_ORG_ID", required=False) or "dev-org-001"
DEV_USER_ID = _get_env("DEV_USER_ID", required=False) or "dev-user-001"

supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

def get_current_user():
    """Get current user for MVP - uses environment variables"""
    return User(
        id=DEV_USER_ID,
        email="dev@example.com",
        name="Development User",
        created_at=datetime.utcnow().isoformat()
    )

def get_current_org_id():
    """Get current organization ID for MVP"""
    return DEV_ORG_ID

# Remove all user authentication functions for MVP - using DEV environment variables instead

def save_document(filename, file_path):
    """Save document record to database"""
    try:
        # Calculate SHA256 hash of the file
        import hashlib
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            sha = sha256_hash.hexdigest()
        except Exception:
            sha = "unknown"
        
        # Determine MIME type
        mime_type = "application/pdf" if filename.lower().endswith('.pdf') else "application/octet-stream"
        
        document_data = {
            "id": str(uuid.uuid4()),
            "org_id": get_current_org_id(),
            "created_by": DEV_USER_ID,
            "title": filename,
            "name": filename,
            "filename": filename,
            "storage_path": file_path,
            "file_path": file_path,
            "sha256": sha,
            "mime_type": mime_type,
            "status": "processing",
            "created_at": datetime.utcnow().isoformat(),
            "processed": False
        }
        
        result = supa.table("documents").insert(document_data).execute()
        
        if result.data:
            doc_record = result.data[0]
            return Document(
                id=doc_record["id"],
                user_id=doc_record["created_by"],
                filename=doc_record["filename"],
                file_path=doc_record["file_path"],
                uploaded_at=doc_record["created_at"],
                processed=doc_record["processed"]
            )
        else:
            raise Exception("Failed to save document")
            
    except Exception as e:
        logging.error(f"Save document failed: {str(e)}")
        raise

def get_user_documents():
    """Get all documents for current organization"""
    try:
        result = supa.table("documents").select("*").eq("org_id", get_current_org_id()).order("created_at", desc=True).execute()
        
        documents = []
        for doc_record in result.data:
            documents.append(Document(
                id=doc_record["id"],
                user_id=doc_record.get("created_by", DEV_USER_ID),
                filename=doc_record["filename"],
                file_path=doc_record["file_path"],
                uploaded_at=doc_record["created_at"],
                processed=doc_record["processed"]
            ))
        
        return documents
        
    except Exception as e:
        logging.error(f"Get user documents failed: {str(e)}")
        return []

def delete_document(document_id):
    """Delete a document and its chunks"""
    try:
        # First, delete document chunks
        supa.table("document_chunks").delete().eq("document_id", document_id).execute()
        
        # Then delete the document (ensure it belongs to current org)
        result = supa.table("documents").delete().eq("id", document_id).eq("org_id", get_current_org_id()).execute()
        
        return True
        
    except Exception as e:
        logging.error(f"Delete document failed: {str(e)}")
        return False

def save_document_chunks(document_id, chunks):
    """Save document chunks with embeddings to database"""
    try:
        chunk_records = []
        
        for chunk in chunks:
            chunk_data = {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk["content"],
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "embedding": chunk["embedding"]
            }
            chunk_records.append(chunk_data)
        
        # Insert chunks in batches
        batch_size = 100
        for i in range(0, len(chunk_records), batch_size):
            batch = chunk_records[i:i + batch_size]
            supa.table("document_chunks").insert(batch).execute()
        
        # Mark document as processed
        supa.table("documents").update({"processed": True}).eq("id", document_id).execute()
        
        logging.info(f"Successfully saved {len(chunk_records)} chunks for document {document_id}")
        
    except Exception as e:
        logging.error(f"Save document chunks failed: {str(e)}")
        raise

def search_similar_chunks(query_embedding, limit=5):
    """Search for similar chunks using vector similarity"""
    try:
        # Get current org's documents
        org_docs = supa.table("documents").select("id").eq("org_id", get_current_org_id()).execute()
        doc_ids = [doc["id"] for doc in org_docs.data] if org_docs.data else []
        
        if not doc_ids:
            return []
        
        # Use Supabase's vector search functionality
        try:
            result = supa.rpc("search_document_chunks", {
                "query_embedding": query_embedding,
                "document_ids": doc_ids,
                "similarity_threshold": 0.7,
                "match_count": limit
            }).execute()
            
            return result.data if result.data else []
        except Exception as e:
            logging.warning(f"Vector search failed: {str(e)}")
            # Fallback: return recent chunks from org's documents
            if not doc_ids:
                return []
            result = supa.table("document_chunks").select("""
                id, document_id, content, page_number,
                documents!inner(filename, org_id)
            """).in_("document_id", doc_ids).limit(limit).execute()
            
            chunks = []
            for chunk in result.data:
                chunks.append({
                    'id': chunk['id'],
                    'document_id': chunk['document_id'],
                    'content': chunk['content'],
                    'page_number': chunk['page_number'],
                    'document_filename': chunk['documents']['filename']
                })
            return chunks
        
    except Exception as e:
        logging.error(f"Search similar chunks failed: {str(e)}")
        return []

def save_chat_message(message, response, citations):
    """Save chat message to database"""
    try:
        chat_data = {
            "id": str(uuid.uuid4()),
            "org_id": get_current_org_id(),
            "created_by": DEV_USER_ID,
            "message": message,
            "response": response,
            "citations": citations,
            "created_at": datetime.utcnow().isoformat()
        }
        
        supa.table("chat_messages").insert(chat_data).execute()
        
    except Exception as e:
        logging.error(f"Save chat message failed: {str(e)}")
        raise

def get_document_by_id(document_id):
    """Get document by ID"""
    try:
        result = supa.table("documents").select("*").eq("id", document_id).eq("org_id", get_current_org_id()).execute()
        
        if result.data and len(result.data) > 0:
            doc_record = result.data[0]
            return Document(
                id=doc_record["id"],
                user_id=doc_record.get("created_by", DEV_USER_ID),
                filename=doc_record["filename"],
                file_path=doc_record["file_path"],
                uploaded_at=doc_record["created_at"],
                processed=doc_record["processed"]
            )
        
        return None
        
    except Exception as e:
        logging.error(f"Get document by ID failed: {str(e)}")
        return None

__all__ = ["supa", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE", "SUPABASE_ANON_KEY", 
           "get_current_user", "get_current_org_id", "save_document", 
           "get_user_documents", "delete_document", "save_document_chunks", 
           "search_similar_chunks", "save_chat_message", "get_document_by_id"]