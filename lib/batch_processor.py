"""
Enterprise Batch Document Processing System
Handles large-scale document ingestion with memory management and performance optimization.
"""

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import psutil
import os

class BatchDocumentProcessor:
    def __init__(self, max_workers=4, batch_size=10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
    
    async def process_documents_batch(self, org_id: str, document_files: list) -> Dict[str, Any]:
        """Process multiple documents efficiently with memory management"""
        
        results = {
            'total_documents': len(document_files),
            'successful': 0,
            'failed': 0,
            'processing_time_seconds': 0,
            'total_chunks_created': 0,
            'memory_usage_mb': 0,
            'errors': [],
            'batch_details': []
        }
        
        start_time = time.time()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        self.logger.info(f"🚀 Starting batch processing of {len(document_files)} documents")
        self.logger.info(f"📊 Configuration: {self.max_workers} workers, batch size {self.batch_size}")
        
        # Process in batches to avoid memory issues
        batches = [document_files[i:i + self.batch_size] for i in range(0, len(document_files), self.batch_size)]
        
        for batch_num, batch in enumerate(batches, 1):
            batch_start_time = time.time()
            self.logger.info(f"📦 Processing batch {batch_num}/{len(batches)} ({len(batch)} documents)")
            
            batch_results = await self._process_batch(org_id, batch, batch_num)
            
            # Update overall results
            results['successful'] += batch_results['successful']
            results['failed'] += batch_results['failed']
            results['total_chunks_created'] += batch_results['total_chunks_created']
            results['errors'].extend(batch_results['errors'])
            results['batch_details'].append({
                'batch_number': batch_num,
                'documents_count': len(batch),
                'successful': batch_results['successful'],
                'failed': batch_results['failed'],
                'processing_time_seconds': time.time() - batch_start_time,
                'chunks_created': batch_results['total_chunks_created']
            })
            
            # Memory monitoring
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            self.logger.info(f"✅ Batch {batch_num} complete: {batch_results['successful']}/{len(batch)} successful")
            self.logger.info(f"🧠 Memory usage: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Brief pause between batches to prevent resource exhaustion
            if batch_num < len(batches):
                await asyncio.sleep(2)
        
        # Final calculations
        results['processing_time_seconds'] = time.time() - start_time
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        results['memory_usage_mb'] = final_memory - initial_memory
        
        # Log final results
        self.logger.info(f"🎯 Batch processing complete!")
        self.logger.info(f"📊 Results: {results['successful']}/{results['total_documents']} successful")
        self.logger.info(f"⏱️ Total time: {results['processing_time_seconds']:.1f}s")
        self.logger.info(f"🧠 Memory increase: {results['memory_usage_mb']:.1f}MB")
        self.logger.info(f"📚 Total chunks created: {results['total_chunks_created']}")
        
        return results
    
    async def _process_batch(self, org_id: str, batch: list, batch_num: int) -> Dict[str, Any]:
        """Process a single batch of documents"""
        
        batch_results = {
            'successful': 0,
            'failed': 0,
            'total_chunks_created': 0,
            'errors': []
        }
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_single_document, org_id, doc_file, f"{batch_num}-{i+1}")
                for i, doc_file in enumerate(batch)
            ]
            
            for future in futures:
                try:
                    doc_result = future.result(timeout=300)  # 5 minute timeout per doc
                    if doc_result['success']:
                        batch_results['successful'] += 1
                        batch_results['total_chunks_created'] += doc_result['chunks_created']
                        self.logger.info(f"   ✅ {doc_result['filename']}: {doc_result['chunks_created']} chunks, {doc_result['processing_time']:.0f}ms")
                    else:
                        batch_results['failed'] += 1
                        batch_results['errors'].append(doc_result['error'])
                        self.logger.error(f"   ❌ {doc_result['filename']}: {doc_result['error']}")
                        
                except Exception as e:
                    batch_results['failed'] += 1
                    error_msg = f"Processing timeout or error: {str(e)}"
                    batch_results['errors'].append(error_msg)
                    self.logger.error(f"   ❌ Batch processing error: {error_msg}")
        
        return batch_results
    
    def _process_single_document(self, org_id: str, document_file, doc_id: str) -> Dict[str, Any]:
        """Process a single document with comprehensive error handling"""
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from lib.enhanced_ingest import process_document_complete
            
            # Get filename safely
            filename = getattr(document_file, 'filename', f'document_{doc_id}')
            
            # Process the document
            result = process_document_complete(document_file, org_id)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'filename': filename,
                'chunks_created': result.get('chunks_count', 0),
                'processing_time': processing_time,
                'document_id': result.get('document_id', None)
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            filename = getattr(document_file, 'filename', f'document_{doc_id}')
            
            error_msg = f"Failed to process {filename}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'filename': filename,
                'error': error_msg,
                'processing_time': processing_time
            }
    
    def get_processing_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed processing statistics"""
        
        if results['total_documents'] == 0:
            return {'error': 'No documents processed'}
        
        success_rate = (results['successful'] / results['total_documents']) * 100
        avg_chunks_per_doc = results['total_chunks_created'] / results['successful'] if results['successful'] > 0 else 0
        docs_per_second = results['total_documents'] / results['processing_time_seconds'] if results['processing_time_seconds'] > 0 else 0
        
        stats = {
            'success_rate_percent': success_rate,
            'average_chunks_per_document': avg_chunks_per_doc,
            'documents_per_second': docs_per_second,
            'memory_efficiency_mb_per_doc': results['memory_usage_mb'] / results['total_documents'],
            'total_processing_time_minutes': results['processing_time_seconds'] / 60,
            'enterprise_ready': success_rate >= 95 and results['memory_usage_mb'] < 500,
            'batch_performance': []
        }
        
        # Analyze batch performance
        for batch_detail in results.get('batch_details', []):
            batch_success_rate = (batch_detail['successful'] / batch_detail['documents_count']) * 100
            stats['batch_performance'].append({
                'batch_number': batch_detail['batch_number'],
                'success_rate_percent': batch_success_rate,
                'processing_time_seconds': batch_detail['processing_time_seconds'],
                'chunks_per_second': batch_detail['chunks_created'] / batch_detail['processing_time_seconds'] if batch_detail['processing_time_seconds'] > 0 else 0
            })
        
        return stats

# Utility functions for enterprise document management
def estimate_processing_time(document_count: int, avg_doc_size_mb: float = 5) -> Dict[str, float]:
    """Estimate processing time for enterprise document batches"""
    
    # Based on empirical testing (adjust based on actual performance)
    base_time_per_doc = 30  # seconds per document
    memory_overhead = document_count * 0.5  # MB per document
    
    estimated_time_minutes = (document_count * base_time_per_doc) / 60
    estimated_memory_mb = memory_overhead + (document_count * avg_doc_size_mb * 0.2)  # 20% overhead
    
    return {
        'estimated_processing_minutes': estimated_time_minutes,
        'estimated_memory_mb': estimated_memory_mb,
        'recommended_batch_size': min(20, max(5, int(500 / avg_doc_size_mb))),  # Based on memory constraints
        'recommended_workers': min(6, max(2, int(document_count / 10)))  # Scale workers with document count
    }

def validate_enterprise_capacity(current_docs: int, target_docs: int) -> Dict[str, Any]:
    """Validate system capacity for enterprise document loads"""
    
    capacity_assessment = {
        'current_documents': current_docs,
        'target_documents': target_docs,
        'capacity_available': target_docs <= 1000,  # Conservative enterprise limit
        'scaling_required': target_docs > current_docs * 2,
        'recommendations': []
    }
    
    if target_docs > 500:
        capacity_assessment['recommendations'].append("Consider database indexing optimization")
    
    if target_docs > 200:
        capacity_assessment['recommendations'].append("Enable batch processing for uploads")
    
    if target_docs > 100:
        capacity_assessment['recommendations'].append("Monitor memory usage during processing")
    
    return capacity_assessment