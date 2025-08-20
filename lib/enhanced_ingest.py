import logging
from typing import Dict, Any, List
from lib.smart_processing import create_smart_processor
from lib.supa import supa

logger = logging.getLogger(__name__)

class EnhancedIngestPipeline:
    """Enhanced ingestion pipeline with smart processing integration"""
    
    def __init__(self):
        self.smart_processor = create_smart_processor()
    
    def process_document_with_smart_extraction(self, document_id: str, text_content: str, 
                                             doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process document with smart entity extraction and visual element detection"""
        
        try:
            # Apply smart processing
            processing_results = self.smart_processor.process_document_intelligently(
                text_content, doc_metadata
            )
            
            # Store processing results in database
            enhanced_metadata = {
                **doc_metadata,
                'smart_processing': processing_results,
                'document_classification': processing_results['document_type'],
                'processing_confidence': processing_results['processing_confidence'],
                'governance_score': processing_results['governance_actions'].get('governance_score', 0.0),
                'visual_content_score': processing_results['visual_elements'].get('visual_content_score', 0.0)
            }
            
            # Update document metadata with smart processing results
            self._update_document_metadata(document_id, enhanced_metadata)
            
            logger.info(f"Enhanced processing completed for {doc_metadata.get('filename')} "
                       f"with {processing_results['processing_confidence']:.0%} confidence")
            
            return processing_results
            
        except Exception as e:
            logger.error(f"Enhanced processing failed for {document_id}: {e}")
            return self.smart_processor._create_fallback_result(doc_metadata, str(e))
    
    def _update_document_metadata(self, document_id: str, enhanced_metadata: Dict[str, Any]):
        """Update document record with enhanced metadata"""
        
        try:
            # Extract key metadata for database storage
            update_data = {
                'document_type': enhanced_metadata['document_classification'],
                'processing_confidence': enhanced_metadata['processing_confidence'],
                'smart_extraction_summary': enhanced_metadata['smart_processing']['extraction_summary'],
                'governance_entities_count': sum(len(entities) for entities in 
                                               enhanced_metadata['smart_processing']['extracted_entities'].values()),
                'visual_elements_detected': enhanced_metadata['visual_content_score'] > 0.3,
                'governance_actions_count': len(enhanced_metadata['smart_processing']['governance_actions'].get('motions_identified', [])) +
                                          len(enhanced_metadata['smart_processing']['governance_actions'].get('voting_results', []))
            }
            
            # Update document record
            result = supa.table("documents").update(update_data).eq("id", document_id).execute()
            
            if result.data:
                logger.info(f"Updated document metadata for {document_id}")
            else:
                logger.warning(f"No document found to update for {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to update document metadata for {document_id}: {e}")

def create_enhanced_ingest_pipeline() -> EnhancedIngestPipeline:
    """Factory function to create enhanced ingest pipeline"""
    return EnhancedIngestPipeline()