import os
import logging
import json
from openai import OpenAI
from services.supabase_service import search_similar_chunks, save_chat_message

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_embeddings(texts):
    """Generate embeddings for a list of texts using text-embedding-3-large"""
    try:
        if isinstance(texts, str):
            texts = [texts]
        
        response = client.embeddings.create(
            model="text-embedding-3-small",  # 1536 dimensions to match database schema
            input=texts
        )
        
        return [item.embedding for item in response.data]
    
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {str(e)}")
        raise

def expand_query_terms(message):
    """Expand query with related terms for better recall"""
    expanded_terms = []
    message_lower = message.lower()
    
    # Query expansion for "reserved" terms
    if "reserved" in message_lower:
        synonyms = ["reserved periods", "open times", "Primary Golfers", "Ladies' 18", "Ladies' 9", "Juniors"]
        expanded_terms.extend(synonyms)
    
    # Return original message plus expanded terms
    if expanded_terms:
        return message + " " + " ".join(expanded_terms)
    return message

def chat_with_documents(message):
    """Chat with documents using GPT-4o and vector search for context"""
    try:
        # Expand query terms for better recall
        expanded_message = expand_query_terms(message)
        
        # Generate embedding for the expanded user's message
        message_embedding = get_embeddings([expanded_message])[0]
        
        # Search for relevant document chunks with increased limit and hybrid fallback
        relevant_chunks = search_similar_chunks(message_embedding, limit=40, fallback_query=message)
        
        # Prepare context from relevant chunks
        context_parts = []
        citations = []
        
        for chunk in relevant_chunks:
            # Include document_id and chunk_index in the context format
            chunk_id = chunk.get('id', 'unknown')
            doc_id = chunk.get('document_id', 'unknown')
            context_parts.append(f"[Doc:{doc_id}#Chunk:{chunk_id}] {chunk['content']}")
            citations.append({
                'document_id': chunk['document_id'],
                'document_filename': chunk['document_filename'],
                'page_number': chunk['page_number'],
                'chunk_id': chunk_id,
                'content_preview': chunk['content'][:150] + "..." if len(chunk['content']) > 150 else chunk['content']
            })
        
        context = "\n\n".join(context_parts)
        
        # Create the chat prompt
        system_prompt = """You are an intelligent document assistant. You help users understand and analyze their uploaded documents. 

When responding:
1. Base your answers primarily on the provided document context
2. Be specific and cite which documents and pages you're referencing
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Provide detailed, helpful responses that demonstrate understanding of the content
5. Use a professional but conversational tone
6. Prefer table rows when present

The user has uploaded documents and you have access to relevant excerpts based on their question. Each excerpt includes document and chunk identifiers for precise citation."""

        user_prompt = f"""Based on the following document excerpts, please answer this question: {message}

Document context:
{context}

Question: {message}"""

        # Generate response using GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Save the chat message
        try:
            save_chat_message(message, ai_response, citations)
        except Exception as e:
            logging.warning(f"Failed to save chat message: {str(e)}")
        
        return {
            'response': ai_response,
            'citations': citations
        }
        
    except Exception as e:
        logging.error(f"Chat with documents failed: {str(e)}")
        raise
