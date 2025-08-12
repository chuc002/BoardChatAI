import logging
import PyPDF2
from io import BytesIO
from services.supabase_service import save_document_chunks
from services.openai_service import get_embeddings

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append({
                            'page': page_num,
                            'text': page_text
                        })
                except Exception as e:
                    logging.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                    continue
            
            return text_content
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {str(e)}")
        raise

def chunk_text(text, max_chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for better context preservation"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # If we're not at the end, try to break at a sentence or paragraph
        if end < len(text):
            # Look for sentence endings
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            if last_period > start + max_chunk_size // 2:
                end = last_period + 1
            elif last_newline > start + max_chunk_size // 2:
                end = last_newline
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start <= 0:
            start = end
    
    return chunks

def process_pdf(document_id, file_path):
    """Process PDF: extract text, create chunks, generate embeddings, and save to database"""
    try:
        logging.info(f"Starting PDF processing for document {document_id}")
        
        # Extract text from PDF
        page_contents = extract_text_from_pdf(file_path)
        if not page_contents:
            raise Exception("No text could be extracted from the PDF")
        
        all_chunks = []
        chunk_index = 0
        
        # Process each page
        for page_data in page_contents:
            page_num = page_data['page']
            page_text = page_data['text']
            
            # Create chunks from page text
            page_chunks = chunk_text(page_text)
            
            for chunk_text_content in page_chunks:
                if chunk_text_content.strip():
                    all_chunks.append({
                        'content': chunk_text_content,
                        'page_number': page_num,
                        'chunk_index': chunk_index
                    })
                    chunk_index += 1
        
        if not all_chunks:
            raise Exception("No valid text chunks could be created from the PDF")
        
        logging.info(f"Created {len(all_chunks)} chunks from {len(page_contents)} pages")
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        embeddings = get_embeddings(chunk_texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(all_chunks):
            chunk['embedding'] = embeddings[i]
        
        # Save chunks to database
        save_document_chunks(document_id, all_chunks)
        
        logging.info(f"Successfully processed PDF for document {document_id}")
        
    except Exception as e:
        logging.error(f"PDF processing failed for document {document_id}: {str(e)}")
        raise
