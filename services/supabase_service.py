import os
import logging
import uuid
import hashlib
from datetime import datetime
from supabase import create_client, Client
from models import User, Document, DocumentChunk

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE environment variables are required")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def hash_password(password):
    """Simple password hashing - in production, use proper hashing like bcrypt"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password, name):
    """Create a new user"""
    try:
        # Check if user already exists
        existing_user = supabase.table("users").select("*").eq("email", email).execute()
        if existing_user.data:
            raise Exception("User with this email already exists")
        
        user_data = {
            "id": str(uuid.uuid4()),
            "email": email,
            "password_hash": hash_password(password),
            "name": name,
            "created_at": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("users").insert(user_data).execute()
        
        if result.data:
            user_record = result.data[0]
            return User(
                id=user_record["id"],
                email=user_record["email"],
                name=user_record["name"],
                created_at=user_record["created_at"]
            )
        else:
            raise Exception("Failed to create user")
            
    except Exception as e:
        logging.error(f"User creation failed: {str(e)}")
        raise

def authenticate_user(email, password):
    """Authenticate user and return User object"""
    try:
        result = supabase.table("users").select("*").eq("email", email).execute()
        
        if result.data and len(result.data) > 0:
            user_record = result.data[0]
            if user_record["password_hash"] == hash_password(password):
                return User(
                    id=user_record["id"],
                    email=user_record["email"],
                    name=user_record["name"],
                    created_at=user_record["created_at"]
                )
        
        return None
        
    except Exception as e:
        logging.error(f"Authentication failed: {str(e)}")
        raise

def get_user_by_id(user_id):
    """Get user by ID for Flask-Login"""
    try:
        result = supabase.table("users").select("*").eq("id", user_id).execute()
        
        if result.data and len(result.data) > 0:
            user_record = result.data[0]
            return User(
                id=user_record["id"],
                email=user_record["email"],
                name=user_record["name"],
                created_at=user_record["created_at"]
            )
        
        return None
        
    except Exception as e:
        logging.error(f"Get user by ID failed: {str(e)}")
        return None

def save_document(user_id, filename, file_path):
    """Save document record to database"""
    try:
        document_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "filename": filename,
            "file_path": file_path,
            "uploaded_at": datetime.utcnow().isoformat(),
            "processed": False
        }
        
        result = supabase.table("documents").insert(document_data).execute()
        
        if result.data:
            doc_record = result.data[0]
            return Document(
                id=doc_record["id"],
                user_id=doc_record["user_id"],
                filename=doc_record["filename"],
                file_path=doc_record["file_path"],
                uploaded_at=doc_record["uploaded_at"],
                processed=doc_record["processed"]
            )
        else:
            raise Exception("Failed to save document")
            
    except Exception as e:
        logging.error(f"Save document failed: {str(e)}")
        raise

def get_user_documents(user_id):
    """Get all documents for a user"""
    try:
        result = supabase.table("documents").select("*").eq("user_id", user_id).order("uploaded_at", desc=True).execute()
        
        documents = []
        for doc_record in result.data:
            documents.append(Document(
                id=doc_record["id"],
                user_id=doc_record["user_id"],
                filename=doc_record["filename"],
                file_path=doc_record["file_path"],
                uploaded_at=doc_record["uploaded_at"],
                processed=doc_record["processed"]
            ))
        
        return documents
        
    except Exception as e:
        logging.error(f"Get user documents failed: {str(e)}")
        return []

def delete_document(document_id, user_id):
    """Delete a document and its chunks"""
    try:
        # First, delete document chunks
        supabase.table("document_chunks").delete().eq("document_id", document_id).execute()
        
        # Then delete the document
        result = supabase.table("documents").delete().eq("id", document_id).eq("user_id", user_id).execute()
        
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
            supabase.table("document_chunks").insert(batch).execute()
        
        # Mark document as processed
        supabase.table("documents").update({"processed": True}).eq("id", document_id).execute()
        
        logging.info(f"Successfully saved {len(chunk_records)} chunks for document {document_id}")
        
    except Exception as e:
        logging.error(f"Save document chunks failed: {str(e)}")
        raise

def search_similar_chunks(user_id, query_embedding, limit=5):
    """Search for similar chunks using vector similarity"""
    try:
        # Get user's documents
        user_docs = supabase.table("documents").select("id").eq("user_id", user_id).execute()
        doc_ids = [doc["id"] for doc in user_docs.data] if user_docs.data else []
        
        if not doc_ids:
            return []
        
        # Use Supabase's vector search functionality
        # This assumes you have set up the pgvector extension and similarity functions
        result = supabase.rpc("search_document_chunks", {
            "query_embedding": query_embedding,
            "document_ids": doc_ids,
            "similarity_threshold": 0.7,
            "match_count": limit
        }).execute()
        
        return result.data if result.data else []
        
    except Exception as e:
        logging.error(f"Vector search failed: {str(e)}")
        # Fallback: return recent chunks from user's documents
        try:
            if not doc_ids:
                return []
            result = supabase.table("document_chunks").select("""
                id, document_id, content, page_number,
                documents!inner(filename, user_id)
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
        except:
            return []

def save_chat_message(user_id, message, response, citations):
    """Save chat message to database"""
    try:
        chat_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "message": message,
            "response": response,
            "citations": citations,
            "created_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("chat_messages").insert(chat_data).execute()
        
    except Exception as e:
        logging.error(f"Save chat message failed: {str(e)}")
        raise

def get_document_by_id(document_id):
    """Get document by ID"""
    try:
        result = supabase.table("documents").select("*").eq("id", document_id).execute()
        
        if result.data and len(result.data) > 0:
            doc_record = result.data[0]
            return Document(
                id=doc_record["id"],
                user_id=doc_record["user_id"],
                filename=doc_record["filename"],
                file_path=doc_record["file_path"],
                uploaded_at=doc_record["uploaded_at"],
                processed=doc_record["processed"]
            )
        
        return None
        
    except Exception as e:
        logging.error(f"Get document by ID failed: {str(e)}")
        return None
