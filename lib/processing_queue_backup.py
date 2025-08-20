import time
import threading
import queue
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

class DocumentProcessingQueue:
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.processing_queue = []
        self.is_processing = False
        self.processing_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'session_started': None,
            'last_activity': None
        }
        
    async def add_documents_to_queue(self, org_id: str, document_ids: List[str] = None):
        """Add documents to processing queue"""
        
        try:
            from lib.supa import supa
            
            # Get unprocessed documents
            if document_ids:
                # Process specific documents
                documents_query = supa.table("documents").select("id,filename,storage_path,title").eq("org_id", org_id).in_("id", document_ids).execute()
            else:
                # Get all documents for org
                all_docs = supa.table("documents").select("id,filename,storage_path,title").eq("org_id", org_id).execute()
                
                if not all_docs.data:
                    return {'added_to_queue': 0, 'queue_size': 0}
                
                # Get documents with chunks to exclude them
                doc_ids = [doc['id'] for doc in all_docs.data]
                chunks_result = supa.table("doc_chunks").select("document_id").in_("document_id", doc_ids).execute()
                
                processed_doc_ids = set()
                if chunks_result.data:
                    processed_doc_ids = set(chunk['document_id'] for chunk in chunks_result.data)
                
                # Filter to unprocessed documents
                documents_query = {'data': [doc for doc in all_docs.data if doc['id'] not in processed_doc_ids]}
            
            documents = documents_query.get('data', []) if isinstance(documents_query, dict) else documents_query.data or []
            
            # Add to queue
            for doc in documents:
                self.processing_queue.append({
                    'document_id': doc['id'],
                    'filename': doc['filename'],
                    'storage_path': doc.get('storage_path', ''),
                    'title': doc.get('title', ''),
                    'org_id': org_id,
                    'added_at': datetime.now(),
                    'priority': 1  # Normal priority
                })
            
            self.logger.info(f"Added {len(documents)} documents to processing queue")
            
            # Start processing if not already running
            if not self.is_processing:
                asyncio.create_task(self.process_queue())
            
            return {
                'added_to_queue': len(documents),
                'queue_size': len(self.processing_queue)
            }
            
        except Exception as e:
            self.logger.error(f"Error adding documents to queue: {e}")
            raise
    
    async def process_queue(self):
        """Process documents in queue with parallel workers"""
        
        if self.is_processing:
            self.logger.info("Processing already in progress")
            return
        
        self.is_processing = True
        self.processing_stats['session_started'] = datetime.now()
        self.logger.info(f"Starting queue processing with {self.max_workers} workers")
        
        try:
            from lib.bulletproof_processing import create_bulletproof_processor
            processor = create_bulletproof_processor()
            
            # Process documents in batches
            batch_size = self.max_workers
            processed_count = 0
            failed_count = 0
            
            while self.processing_queue:
                # Get next batch
                batch = self.processing_queue[:batch_size]
                self.processing_queue = self.processing_queue[batch_size:]
                
                # Process batch in parallel using asyncio instead of ThreadPoolExecutor
                tasks = []
                for doc in batch:
                    task = asyncio.create_task(self._process_document_async(processor, doc))
                    tasks.append(task)
                
                # Wait for batch completion
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    doc = batch[i]
                    if isinstance(result, Exception):
                        failed_count += 1
                        self.logger.error(f"❌ Exception processing {doc['filename']}: {result}")
                    elif result and result.get('success'):
                        processed_count += 1
                        self.logger.info(f"✅ Successfully processed: {doc['filename']}")
                    else:
                        failed_count += 1
                        self.logger.error(f"❌ Failed to process: {doc['filename']}")
                
                # Brief pause between batches
                await asyncio.sleep(1)
                self.processing_stats['last_activity'] = datetime.now()
            
            # Update final stats
            self.processing_stats['total_processed'] += processed_count
            self.processing_stats['total_failed'] += failed_count
            
            self.logger.info(f"Queue processing complete: {processed_count} successful, {failed_count} failed")
            
            return {
                'processed_count': processed_count,
                'failed_count': failed_count,
                'total_documents': processed_count + failed_count
            }
            
        except Exception as e:
            self.logger.error(f"Queue processing error: {e}")
            return {
                'processed_count': 0,
                'failed_count': 0,
                'error': str(e)
            }
        finally:
            self.is_processing = False
    
    async def _process_document_async(self, processor, doc: Dict[str, Any]):
        """Process a single document asynchronously"""
        
        try:
            # Convert document format for processor
            doc_data = {
                'id': doc['document_id'],
                'filename': doc['filename'],
                'storage_path': doc['storage_path'],
                'title': doc.get('title', '')
            }
            
            # Process using bulletproof processor
            result = processor._process_single_document_bulletproof(doc['org_id'], doc_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in async document processing: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        
        return {
            'queue_size': len(self.processing_queue),
            'is_processing': self.is_processing,
            'processing_stats': self.processing_stats,
            'pending_documents': [
                {
                    'filename': doc['filename'],
                    'added_at': doc['added_at'].isoformat(),
                    'priority': doc['priority']
                } for doc in self.processing_queue[:10]  # Show first 10
            ]
        }
    
    def clear_queue(self):
        """Clear the processing queue"""
        self.processing_queue.clear()
        self.logger.info("Processing queue cleared")
    
    def pause_processing(self):
        """Pause processing (will finish current batch)"""
        self.is_processing = False
        self.logger.info("Processing paused")

# Singleton queue instance
document_queue = DocumentProcessingQueue()

def get_document_queue():
    """Get the singleton document queue instance"""
    return document_queue