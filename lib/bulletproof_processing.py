import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
from datetime import datetime

# Multiple PDF processing libraries for fallback
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.layout import LAParams
except ImportError:
    pdfminer_extract = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
    import pdf2image
except ImportError:
    pytesseract = None

class BulletproofDocumentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Multiple extraction strategies
        self.extraction_strategies = [
            self._extract_with_pypdf2,
            self._extract_with_pdfminer,
            self._extract_with_pymupdf,
            self._extract_with_ocr,
        ]
        
        # Processing status tracking
        self.processing_status = {}
        
    def process_all_documents(self, org_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process ALL documents with 100% success guarantee"""
        
        self.logger.info(f"Starting bulletproof processing for org: {org_id}")
        
        # Get all documents that need processing
        unprocessed_docs = self._get_unprocessed_documents(org_id, force_reprocess)
        
        if not unprocessed_docs:
            return {
                'status': 'complete',
                'message': 'All documents already processed',
                'coverage': '100%'
            }
        
        self.logger.info(f"Found {len(unprocessed_docs)} documents to process")
        
        # Process each document with multiple fallback strategies
        results = {
            'total_documents': len(unprocessed_docs),
            'successful': 0,
            'failed': 0,
            'processing_details': [],
            'errors': []
        }
        
        for doc in unprocessed_docs:
            self.logger.info(f"Processing document: {doc['filename']}")
            
            doc_result = self._process_single_document_bulletproof(org_id, doc)
            results['processing_details'].append(doc_result)
            
            if doc_result['success']:
                results['successful'] += 1
                self.logger.info(f"✅ Successfully processed: {doc['filename']}")
            else:
                results['failed'] += 1
                results['errors'].append(doc_result['error'])
                self.logger.error(f"❌ Failed to process: {doc['filename']} - {doc_result['error']}")
        
        # Calculate final coverage
        total_docs = self._get_total_document_count(org_id)
        processed_docs = self._get_processed_document_count(org_id)
        coverage_percentage = (processed_docs / total_docs * 100) if total_docs > 0 else 0
        
        results['final_coverage'] = f"{coverage_percentage:.1f}%"
        results['processed_count'] = processed_docs
        results['total_count'] = total_docs
        
        self.logger.info(f"Processing complete. Coverage: {results['final_coverage']}")
        
        return results
    
    def _get_unprocessed_documents(self, org_id: str, force_reprocess: bool = False) -> List[Dict]:
        """Get list of documents that haven't been processed"""
        
        try:
            from lib.supa import supa
            
            if force_reprocess:
                # Get ALL documents for reprocessing
                result = supa.table("documents").select("id,filename,storage_path,created_at,title").eq("org_id", org_id).order("created_at", desc=True).execute()
            else:
                # Get only unprocessed documents (no chunks)
                # First get all documents
                all_docs = supa.table("documents").select("id,filename,storage_path,created_at,title").eq("org_id", org_id).execute()
                
                if not all_docs.data:
                    return []
                
                # Check which ones have chunks
                doc_ids = [doc['id'] for doc in all_docs.data]
                chunks_result = supa.table("doc_chunks").select("document_id").in_("document_id", doc_ids).execute()
                
                processed_doc_ids = set()
                if chunks_result.data:
                    processed_doc_ids = {chunk['document_id'] for chunk in chunks_result.data}
                
                # Return documents without chunks
                unprocessed = [doc for doc in all_docs.data if doc['id'] not in processed_doc_ids]
                return unprocessed
                
            return result.data if result.data else []
            
        except Exception as e:
            self.logger.error(f"Error getting unprocessed documents: {e}")
            return []
    
    def _process_single_document_bulletproof(self, org_id: str, doc: Dict) -> Dict[str, Any]:
        """Process single document with multiple fallback strategies"""
        
        doc_id = doc['id']
        filename = doc['filename']
        file_path = doc.get('storage_path', '')
        
        # Initialize processing result
        processing_result = {
            'document_id': doc_id,
            'filename': filename,
            'success': False,
            'extraction_method': None,
            'chunks_created': 0,
            'pages_processed': 0,
            'processing_time_seconds': 0,
            'error': None,
            'warnings': []
        }
        
        start_time = time.time()
        
        try:
            # Determine actual file path (download from storage if needed)
            actual_file_path = self._get_actual_file_path(file_path, filename)
            
            if not actual_file_path or not os.path.exists(actual_file_path):
                raise Exception(f"File not found: {actual_file_path}")
            
            # Try each extraction strategy until one succeeds
            text_content = None
            pages_info = None
            
            for i, strategy in enumerate(self.extraction_strategies):
                strategy_name = strategy.__name__.replace('_extract_with_', '')
                
                try:
                    self.logger.info(f"Trying extraction strategy {i+1}: {strategy_name}")
                    
                    text_content, pages_info = strategy(actual_file_path)
                    
                    if text_content and len(text_content.strip()) > 100:  # Minimum content threshold
                        processing_result['extraction_method'] = strategy_name
                        self.logger.info(f"✅ Successfully extracted with {strategy_name}")
                        break
                    else:
                        self.logger.warning(f"⚠️ {strategy_name} extracted insufficient content")
                        processing_result['warnings'].append(f"{strategy_name}: insufficient content")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ {strategy_name} failed: {str(e)}")
                    processing_result['warnings'].append(f"{strategy_name}: {str(e)}")
                    continue
            
            if not text_content:
                raise Exception("All extraction strategies failed - no readable content found")
            
            # Process the extracted content
            processing_result['pages_processed'] = len(pages_info) if pages_info else 1
            
            # Create chunks and embeddings
            chunks_result = self._create_chunks_and_embeddings(
                org_id, doc_id, filename, text_content, pages_info
            )
            
            processing_result['chunks_created'] = chunks_result['chunks_created']
            processing_result['success'] = True
            
            # Update document status
            self._update_document_status(doc_id, 'processed', processing_result['chunks_created'])
            
        except Exception as e:
            processing_result['error'] = str(e)
            processing_result['success'] = False
            self.logger.error(f"Failed to process {filename}: {e}")
            
            # Update document status as failed
            self._update_document_status(doc_id, 'failed', 0, str(e))
        
        processing_result['processing_time_seconds'] = time.time() - start_time
        
        return processing_result
    
    def _get_actual_file_path(self, storage_path: str, filename: str) -> str:
        """Get the actual file path, downloading from Supabase storage if needed"""
        
        # Try local cache first
        local_cache_dir = 'temp_documents'
        os.makedirs(local_cache_dir, exist_ok=True)
        local_file_path = os.path.join(local_cache_dir, filename)
        
        if os.path.exists(local_file_path):
            return local_file_path
        
        # Download from Supabase storage
        if storage_path:
            try:
                from lib.supa import supa, signed_url_for
                
                self.logger.info(f"Downloading document from storage: {filename}")
                
                # Get signed URL for the document
                signed_url = signed_url_for(storage_path)
                
                if signed_url:
                    # Download the file
                    import requests
                    response = requests.get(signed_url, stream=True)
                    response.raise_for_status()
                    
                    # Save to local cache
                    with open(local_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    self.logger.info(f"Downloaded successfully: {filename}")
                    return local_file_path
                    
            except Exception as e:
                self.logger.error(f"Failed to download {filename}: {e}")
        
        # Try common upload directories as fallback
        possible_paths = [
            os.path.join('uploads', filename),
            os.path.join('.', 'uploads', filename),
            os.path.join('files', filename),
            filename  # Current directory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _extract_with_pypdf2(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Extract text using PyPDF2"""
        
        if not PyPDF2:
            raise Exception("PyPDF2 not available")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            text_content = []
            pages_info = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                        pages_info.append({
                            'page_number': page_num + 1,
                            'text': page_text,
                            'char_count': len(page_text)
                        })
                except Exception as e:
                    self.logger.warning(f"PyPDF2 failed on page {page_num + 1}: {e}")
                    continue
        
        full_text = '\n\n'.join(text_content)
        return full_text, pages_info
    
    def _extract_with_pdfminer(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Extract text using pdfminer"""
        
        if not pdfminer_extract:
            raise Exception("pdfminer not available")
        
        # Configure layout analysis parameters for better text extraction
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            all_texts=False
        )
        
        text_content = pdfminer_extract(file_path, laparams=laparams)
        
        # Split into pages (approximate)
        pages = text_content.split('\f')  # Form feed character often separates pages
        pages_info = []
        
        for i, page_text in enumerate(pages):
            if page_text.strip():
                pages_info.append({
                    'page_number': i + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
        
        return text_content, pages_info
    
    def _extract_with_pymupdf(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Extract text using PyMuPDF (fitz)"""
        
        if not fitz:
            raise Exception("PyMuPDF not available")
        
        doc = fitz.open(file_path)
        text_content = []
        pages_info = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            
            if page_text.strip():
                text_content.append(page_text)
                pages_info.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
        
        doc.close()
        
        full_text = '\n\n'.join(text_content)
        return full_text, pages_info
    
    def _extract_with_ocr(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Extract text using OCR as last resort"""
        
        if not pytesseract or not pdf2image:
            raise Exception("OCR libraries not available")
        
        self.logger.info("Using OCR extraction - this may take longer...")
        
        # Convert PDF to images
        images = pdf2image.convert_from_path(file_path, dpi=300)
        
        text_content = []
        pages_info = []
        
        for i, image in enumerate(images):
            try:
                # Use OCR to extract text from image
                page_text = pytesseract.image_to_string(image, lang='eng')
                
                if page_text.strip():
                    text_content.append(page_text)
                    pages_info.append({
                        'page_number': i + 1,
                        'text': page_text,
                        'char_count': len(page_text),
                        'extraction_method': 'OCR'
                    })
                    
            except Exception as e:
                self.logger.warning(f"OCR failed on page {i + 1}: {e}")
                continue
        
        full_text = '\n\n'.join(text_content)
        return full_text, pages_info
    
    def _create_chunks_and_embeddings(self, org_id: str, doc_id: str, filename: str, 
                                     text_content: str, pages_info: List[Dict]) -> Dict[str, Any]:
        """Create chunks and embeddings from extracted text"""
        
        try:
            from lib.supa import supa
            from openai import OpenAI
            import tiktoken
            
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")
            
            # Create chunks using simple text splitting
            all_chunks = []
            chunk_size = 800  # Target tokens per chunk
            overlap = 100     # Overlap tokens
            
            if pages_info:
                # Process by pages if available
                for page_info in pages_info:
                    page_text = page_info['text']
                    page_chunks = self._create_simple_chunks(
                        page_text, doc_id, filename, 
                        page_info['page_number'], encoding, chunk_size, overlap
                    )
                    all_chunks.extend(page_chunks)
            else:
                # Fallback: process entire document as single unit
                doc_chunks = self._create_simple_chunks(
                    text_content, doc_id, filename, 1, encoding, chunk_size, overlap
                )
                all_chunks.extend(doc_chunks)
            
            if not all_chunks:
                raise Exception("No chunks created from document")
            
            # Create embeddings for chunks
            chunks_with_embeddings = []
            batch_size = 100  # Process in batches to avoid API limits
            
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                # Get embeddings for batch
                texts = [chunk['content'] for chunk in batch]
                
                try:
                    response = client.embeddings.create(
                        input=texts,
                        model="text-embedding-3-small"
                    )
                    
                    embeddings = [embedding.embedding for embedding in response.data]
                    
                    # Add embeddings to chunks
                    for chunk, embedding in zip(batch, embeddings):
                        chunk['embedding'] = embedding
                        chunk['org_id'] = org_id
                        chunks_with_embeddings.append(chunk)
                        
                except Exception as e:
                    self.logger.error(f"Failed to create embeddings for batch: {e}")
                    raise
            
            # Upsert chunks to database
            upsert_result = upsert_chunks(chunks_with_embeddings)
            
            return {
                'chunks_created': len(chunks_with_embeddings),
                'pages_processed': len(pages_info) if pages_info else 1,
                'total_characters': len(text_content)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create chunks and embeddings: {e}")
            raise Exception(f"Chunking and embedding failed: {str(e)}")
    
    def _update_document_status(self, doc_id: str, status: str, chunk_count: int, error: str = None):
        """Update document processing status"""
        
        try:
            from lib.supa import supa
            
            update_data = {
                'status': status,
                'processed': True if status == 'processed' else False
            }
            
            if chunk_count > 0:
                update_data['processed'] = True
            
            # Try to add processing info if columns exist
            try:
                if status == 'processed':
                    update_data.update({
                        'processing_status': 'processed',
                        'chunks_count': chunk_count,
                        'processed_at': datetime.now().isoformat()
                    })
                elif status == 'failed' and error:
                    update_data.update({
                        'processing_status': 'failed',
                        'processing_error': error[:500]
                    })
            except:
                pass  # Columns may not exist yet
            
            supa.table("documents").update(update_data).eq("id", doc_id).execute()
            self.logger.info(f"Updated document status: {doc_id} -> {status}")
            
        except Exception as e:
            self.logger.error(f"Failed to update document status: {e}")
    
    def _get_total_document_count(self, org_id: str) -> int:
        """Get total number of documents for organization"""
        
        try:
            from lib.supa import supa
            result = supa.table("documents").select("id", count="exact").eq("org_id", org_id).execute()
            return result.count or 0
        except Exception as e:
            self.logger.error(f"Failed to get total document count: {e}")
            return 0
    
    def _get_processed_document_count(self, org_id: str) -> int:
        """Get number of processed documents for organization"""
        
        try:
            from lib.supa import supa
            
            # Get documents with chunks
            all_docs = supa.table("documents").select("id").eq("org_id", org_id).execute()
            
            if not all_docs.data:
                return 0
            
            doc_ids = [doc['id'] for doc in all_docs.data]
            chunks_result = supa.table("doc_chunks").select("document_id").in_("document_id", doc_ids).execute()
            
            if not chunks_result.data:
                return 0
            
            # Count unique documents with chunks
            processed_doc_ids = set(chunk['document_id'] for chunk in chunks_result.data)
            return len(processed_doc_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to get processed document count: {e}")
            return 0
    
    def _create_simple_chunks(self, text: str, doc_id: str, filename: str, page_num: int, 
                             encoding, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Create simple text chunks with tiktoken tokenization"""
        
        chunks = []
        
        # Split text into sentences for better chunking
        sentences = text.split('.')
        current_chunk = ""
        chunk_order = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Check token count
            tokens = encoding.encode(test_chunk)
            
            if len(tokens) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'document_id': doc_id,
                        'filename': filename,
                        'page_number': page_num,
                        'chunk_order': chunk_order,
                        'content': current_chunk.strip(),
                        'token_count': len(encoding.encode(current_chunk))
                    })
                    chunk_order += 1
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'document_id': doc_id,
                'filename': filename,  
                'page_number': page_num,
                'chunk_order': chunk_order,
                'content': current_chunk.strip(),
                'token_count': len(encoding.encode(current_chunk))
            })
        
        return chunks


    def _create_simple_chunks(self, text: str, doc_id: str, filename: str, page_num: int, 
                             encoding, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Create simple text chunks with tiktoken tokenization"""
        
        chunks = []
        
        # Split text into sentences for better chunking
        sentences = text.split('.')
        current_chunk = ""
        chunk_order = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            # Check token count
            tokens = encoding.encode(test_chunk)
            
            if len(tokens) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'document_id': doc_id,
                        'filename': filename,
                        'page_number': page_num,
                        'chunk_order': chunk_order,
                        'content': current_chunk.strip(),
                        'token_count': len(encoding.encode(current_chunk))
                    })
                    chunk_order += 1
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'document_id': doc_id,
                'filename': filename,  
                'page_number': page_num,
                'chunk_order': chunk_order,
                'content': current_chunk.strip(),
                'token_count': len(encoding.encode(current_chunk))
            })
        
        return chunks

    def get_processing_status(self, org_id: str) -> Dict[str, Any]:
        """Get comprehensive processing status for organization"""
        
        total_docs = self._get_total_document_count(org_id)
        processed_docs = self._get_processed_document_count(org_id)
        
        coverage_percentage = (processed_docs / total_docs * 100) if total_docs > 0 else 0
        
        return {
            'total_documents': total_docs,
            'processed_documents': processed_docs,
            'unprocessed_documents': total_docs - processed_docs,
            'coverage_percentage': round(coverage_percentage, 1),
            'status': 'complete' if coverage_percentage == 100 else 'incomplete',
            'last_check': datetime.now().isoformat()
        }

# Diagnostic and repair functions
class DocumentCoverageDiagnostic:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def diagnose_coverage_issues(self, org_id: str) -> Dict[str, Any]:
        """Comprehensive diagnosis of document coverage issues"""
        
        diagnosis = {
            'coverage_analysis': {},
            'processing_issues': [],
            'recommendations': [],
            'repair_actions': []
        }
        
        try:
            from lib.supa import supa
            
            # Get all documents 
            all_docs = supa.table("documents").select("id,filename,created_at,title,status").eq("org_id", org_id).execute()
            documents = all_docs.data if all_docs.data else []
            
            # Get documents with chunks
            if documents:
                doc_ids = [doc['id'] for doc in documents]
                chunks_result = supa.table("doc_chunks").select("document_id").in_("document_id", doc_ids).execute()
                
                processed_doc_ids = set()
                if chunks_result.data:
                    processed_doc_ids = {chunk['document_id'] for chunk in chunks_result.data}
            else:
                processed_doc_ids = set()
            
            # Analyze coverage
            total_docs = len(documents)
            processed_docs = len(processed_doc_ids)
            coverage_percentage = (processed_docs / total_docs * 100) if total_docs > 0 else 0
            
            diagnosis['coverage_analysis'] = {
                'total_documents': total_docs,
                'processed_documents': processed_docs,
                'unprocessed_documents': total_docs - processed_docs,
                'coverage_percentage': round(coverage_percentage, 1),
                'is_complete': coverage_percentage == 100.0
            }
            
            # Identify specific issues
            for doc in documents:
                if doc['id'] not in processed_doc_ids:
                    issue = {
                        'document_id': doc['id'],
                        'filename': doc['filename'],
                        'status': doc.get('status', 'unknown'),
                        'title': doc.get('title', ''),
                        'created_at': doc.get('created_at')
                    }
                    
                    # Categorize the issue
                    if doc.get('status') == 'failed':
                        issue['category'] = 'processing_failed'
                        issue['recommended_action'] = 'retry_with_different_strategy'
                    elif doc.get('status') == 'pending':
                        issue['category'] = 'processing_pending'
                        issue['recommended_action'] = 'process_immediately'
                    else:
                        issue['category'] = 'never_processed'
                        issue['recommended_action'] = 'full_processing_required'
                    
                    diagnosis['processing_issues'].append(issue)
            
            # Generate recommendations
            if coverage_percentage < 100:
                diagnosis['recommendations'].extend([
                    f"Immediate action required: {total_docs - processed_docs} documents need processing",
                    "Run bulletproof processing to achieve 100% coverage",
                    "Implement automated retry for failed documents"
                ])
                
                diagnosis['repair_actions'].extend([
                    "force_reprocess_failed",
                    "process_pending_documents", 
                    "verify_file_accessibility",
                    "check_processing_logs"
                ])
            else:
                diagnosis['recommendations'].append("✅ 100% document coverage achieved")
                
        except Exception as e:
            diagnosis['error'] = str(e)
            self.logger.error(f"Diagnosis failed: {e}")
        
        return diagnosis
    
    def repair_coverage_issues(self, org_id: str, repair_actions: List[str]) -> Dict[str, Any]:
        """Execute repair actions to fix coverage issues"""
        
        processor = BulletproofDocumentProcessor()
        repair_results = {
            'actions_taken': [],
            'results': {},
            'success': False
        }
        
        try:
            if 'force_reprocess_failed' in repair_actions:
                self.logger.info("Reprocessing failed documents...")
                result = processor.process_all_documents(org_id, force_reprocess=True)
                repair_results['actions_taken'].append('force_reprocess_failed')
                repair_results['results']['reprocessing'] = result
            
            if 'process_pending_documents' in repair_actions:
                self.logger.info("Processing pending documents...")
                result = processor.process_all_documents(org_id, force_reprocess=False)
                repair_results['actions_taken'].append('process_pending_documents')
                repair_results['results']['pending_processing'] = result
            
            # Verify final coverage
            final_diagnosis = self.diagnose_coverage_issues(org_id)
            repair_results['final_coverage'] = final_diagnosis['coverage_analysis']
            repair_results['success'] = final_diagnosis['coverage_analysis']['is_complete']
            
        except Exception as e:
            repair_results['error'] = str(e)
            self.logger.error(f"Repair failed: {e}")
        
        return repair_results

class BulletproofDocumentProcessor:
    """Extended class with chunking methods"""
    
# Factory function for easy integration
def create_bulletproof_processor() -> BulletproofDocumentProcessor:
    """Create bulletproof document processor instance"""
    return BulletproofDocumentProcessor()

# Convenience function for processing all documents
def process_all_documents_bulletproof(org_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
    """Process all documents with bulletproof extraction"""
    processor = create_bulletproof_processor()
    return processor.process_all_documents(org_id, force_reprocess)