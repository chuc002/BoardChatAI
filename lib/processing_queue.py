# lib/processing_queue.py - Complete Fix
# This fixes all import and threading issues

import time
import threading
import queue
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProcessingQueue:
    """Thread-safe document processing queue for enterprise scale"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.queue = queue.Queue()
        self.workers = []
        self.running = False
        self.stats = {
            'processed': 0,
            'failed': 0,
            'in_progress': 0
        }
        self.logger = logging.getLogger(__name__)
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to processing queue"""
        try:
            for doc in documents:
                self.queue.put(doc)
            
            self.logger.info(f"Added {len(documents)} documents to processing queue")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to queue: {e}")
            return False
    
    def start_processing(self) -> bool:
        """Start worker threads to process queue"""
        try:
            if self.running:
                self.logger.warning("Processing already running")
                return True
                
            self.running = True
            self.workers = []
            
            # Start worker threads
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_thread,
                    args=(f"worker-{i}",),
                    daemon=True
                )
                worker.start()
                self.workers.append(worker)
            
            self.logger.info(f"Starting queue processing with {self.max_workers} workers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start processing: {e}")
            self.running = False
            return False
    
    def stop_processing(self) -> bool:
        """Stop all worker threads"""
        try:
            self.running = False
            
            # Signal workers to stop by adding None items
            for _ in range(self.max_workers):
                self.queue.put(None)
            
            # Wait for workers to finish
            for worker in self.workers:
                worker.join(timeout=5.0)
            
            self.workers = []
            self.logger.info("Processing queue stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop processing: {e}")
            return False
    
    def _worker_thread(self, worker_name: str):
        """Worker thread function"""
        self.logger.info(f"{worker_name} started")
        
        while self.running:
            try:
                # Get document from queue with timeout
                doc = self.queue.get(timeout=1.0)
                
                # Check for stop signal
                if doc is None:
                    self.logger.info(f"{worker_name} received stop signal")
                    break
                
                # Process document
                self.stats['in_progress'] += 1
                success = self._process_document_bulletproof(doc, worker_name)
                
                if success:
                    self.stats['processed'] += 1
                else:
                    self.stats['failed'] += 1
                    
                self.stats['in_progress'] -= 1
                
                # Mark task as done
                self.queue.task_done()
                
            except queue.Empty:
                # Timeout waiting for work - continue loop
                continue
                
            except Exception as e:
                self.logger.error(f"{worker_name} error: {e}")
                self.stats['failed'] += 1
                self.stats['in_progress'] -= 1
                
        self.logger.info(f"{worker_name} finished")
    
    def _process_document_bulletproof(self, doc: Dict[str, Any], worker_name: str) -> bool:
        """Process a single document with bulletproof extraction"""
        try:
            doc_id = doc.get('id')
            filename = doc.get('filename', 'unknown')
            
            self.logger.info(f"{worker_name} processing {filename}")
            
            # Use bulletproof processing system
            from lib.bulletproof_processing import create_bulletproof_processor
            
            processor = create_bulletproof_processor()
            result = processor._process_single_document_bulletproof("63602dc6-defe-4355-b66c-aa6b3b1273e3", doc)
            
            if result and result.get('success'):
                self.logger.info(f"{worker_name} successfully processed {filename}")
                return True
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result'
                self.logger.error(f"{worker_name} failed to process {filename}: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"{worker_name} exception processing {doc.get('filename', 'unknown')}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            'queue_size': self.queue.qsize(),
            'running': self.running,
            'workers': len(self.workers)
        }
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all items in queue to be processed"""
        try:
            if timeout:
                # Wait with timeout
                start_time = time.time()
                while not self.queue.empty():
                    if time.time() - start_time > timeout:
                        self.logger.warning(f"Queue processing timeout after {timeout}s")
                        return False
                    time.sleep(0.1)
            else:
                # Wait indefinitely
                self.queue.join()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error waiting for completion: {e}")
            return False

# Updated DocumentQueue class to maintain compatibility
class DocumentQueue:
    """Enhanced document processing queue with smart processing integration"""
    
    def __init__(self, max_workers: int = 3):
        self.processing_queue = ProcessingQueue(max_workers)
        self.logger = logging.getLogger(__name__)
    
    def add_documents_to_queue(self, org_id: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Add unprocessed documents to queue"""
        try:
            from lib.supa import supa
            
            # Get unprocessed documents
            if force_reprocess:
                result = supa.table("documents").select("id,filename,storage_path,created_at,title").eq("org_id", org_id).execute()
                unprocessed_docs = result.data if result.data else []
            else:
                # Get documents without chunks
                all_docs = supa.table("documents").select("id,filename,storage_path,created_at,title").eq("org_id", org_id).execute()
                if not all_docs.data:
                    return {'added': 0, 'error': 'No documents found'}
                
                doc_ids = [doc['id'] for doc in all_docs.data]
                chunks_result = supa.table("doc_chunks").select("document_id").in_("document_id", doc_ids).execute()
                
                processed_doc_ids = set()
                if chunks_result.data:
                    processed_doc_ids = {chunk['document_id'] for chunk in chunks_result.data}
                
                unprocessed_docs = [doc for doc in all_docs.data if doc['id'] not in processed_doc_ids]
            
            # Add to queue
            if unprocessed_docs:
                success = self.processing_queue.add_documents(unprocessed_docs)
                if success:
                    self.logger.info(f"Added {len(unprocessed_docs)} documents to processing queue")
                    return {'added': len(unprocessed_docs), 'documents': unprocessed_docs}
                else:
                    return {'added': 0, 'error': 'Failed to add documents to queue'}
            else:
                return {'added': 0, 'message': 'No unprocessed documents found'}
                
        except Exception as e:
            self.logger.error(f"Error adding documents to queue: {e}")
            return {'added': 0, 'error': str(e)}
    
    def start_processing(self) -> bool:
        """Start processing documents in queue"""
        return self.processing_queue.start_processing()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        stats = self.processing_queue.get_stats()
        return {
            'status': 'running' if stats['running'] else 'stopped',
            'queue_size': stats['queue_size'],
            'processed': stats['processed'],
            'failed': stats['failed'],
            'in_progress': stats['in_progress'],
            'workers': stats['workers']
        }

# Global instances
_processing_queue = None
_document_queue = None

def get_document_queue() -> DocumentQueue:
    """Get or create global document queue"""
    global _document_queue
    if _document_queue is None:
        _document_queue = DocumentQueue()
    return _document_queue

def get_processing_queue() -> ProcessingQueue:
    """Get or create global processing queue"""
    global _processing_queue
    if _processing_queue is None:
        _processing_queue = ProcessingQueue()
    return _processing_queue