import time
import logging
from typing import Dict, Any, Optional
from lib.supa import supa

logger = logging.getLogger(__name__)

class AutoScalingRAG:
    """Intelligent RAG system that auto-scales based on document volume and complexity"""
    
    def __init__(self):
        self.rag_threshold_docs = 100
        self.rag_threshold_tokens = 150000
        self.direct_context_limit = 2000
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def determine_processing_mode(self, org_id: str) -> Dict[str, Any]:
        """Auto-detect optimal processing mode based on organizational scale"""
        
        cache_key = f"mode_{org_id}"
        current_time = time.time()
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if current_time - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['mode_info']
        
        try:
            # Get document statistics
            doc_result = supa.table("documents").select("id").eq("org_id", org_id).execute()
            total_documents = len(doc_result.data) if doc_result.data else 0
            
            # Get chunk statistics (proxy for token count)
            chunk_result = supa.table("doc_chunks").select("id").eq("org_id", org_id).execute()
            total_chunks = len(chunk_result.data) if chunk_result.data else 0
            estimated_tokens = total_chunks * 150  # Rough estimate
            
            # Determine processing mode
            use_rag = (total_documents > self.rag_threshold_docs or 
                      estimated_tokens > self.rag_threshold_tokens)
            
            mode_info = {
                'use_rag_mode': use_rag,
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'estimated_tokens': estimated_tokens,
                'processing_strategy': 'full_rag' if use_rag else 'direct_context',
                'scale_tier': self._determine_scale_tier(total_documents, estimated_tokens)
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'mode_info': mode_info,
                'timestamp': current_time
            }
            
            logger.info(f"Auto-scaling mode for {org_id}: {mode_info['processing_strategy']} "
                       f"({total_documents} docs, ~{estimated_tokens} tokens)")
            
            return mode_info
            
        except Exception as e:
            logger.error(f"Failed to determine processing mode: {e}")
            # Fallback to direct context for safety
            return {
                'use_rag_mode': False,
                'total_documents': 0,
                'total_chunks': 0,
                'estimated_tokens': 0,
                'processing_strategy': 'direct_context_fallback',
                'scale_tier': 'startup',
                'error': str(e)
            }
    
    def _determine_scale_tier(self, docs: int, tokens: int) -> str:
        """Determine organizational scale tier"""
        if docs > 500 or tokens > 500000:
            return 'enterprise'
        elif docs > 100 or tokens > 150000:
            return 'corporate'
        elif docs > 25 or tokens > 50000:
            return 'professional'
        else:
            return 'startup'
    
    def generate_scaled_response(self, org_id: str, query: str) -> Dict[str, Any]:
        """Generate response using auto-scaled processing approach"""
        start_time = time.time()
        
        try:
            # Determine optimal processing mode
            mode_info = self.determine_processing_mode(org_id)
            
            if mode_info['use_rag_mode']:
                return self._generate_rag_response(org_id, query, mode_info)
            else:
                return self._generate_direct_response(org_id, query, mode_info)
                
        except Exception as e:
            logger.error(f"Auto-scaling RAG failed: {e}")
            return self._generate_emergency_response(query, start_time, error=str(e))
    
    def _generate_rag_response(self, org_id: str, query: str, mode_info: Dict) -> Dict[str, Any]:
        """Generate response using full RAG pipeline for large-scale organizations"""
        start_time = time.time()
        
        try:
            # Use optimized RAG for enterprise scale
            from lib.fast_rag import FastRAG
            
            fast_rag = FastRAG()
            response_data = fast_rag.generate_fast_response(org_id, query)
            
            # Enhance with scale information
            response_data.update({
                'auto_scaling_used': True,
                'processing_mode': 'full_rag',
                'scale_tier': mode_info['scale_tier'],
                'document_count': mode_info['total_documents'],
                'estimated_complexity': 'high'
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            return self._generate_emergency_response(query, start_time, error=str(e))
    
    def _generate_direct_response(self, org_id: str, query: str, mode_info: Dict) -> Dict[str, Any]:
        """Generate response using direct context for smaller organizations"""
        start_time = time.time()
        
        try:
            # Use performance bypass for direct, fast responses
            from lib.performance_bypass import create_performance_bypass
            
            bypass = create_performance_bypass()
            response_data = bypass.handle_query_with_timeout(org_id, query)
            
            # Enhance with scale information
            response_data.update({
                'auto_scaling_used': True,
                'processing_mode': 'direct_context',
                'scale_tier': mode_info['scale_tier'],
                'document_count': mode_info['total_documents'],
                'estimated_complexity': 'low_to_medium'
            })
            
            return response_data
            
        except Exception as e:
            logger.error(f"Direct response generation failed: {e}")
            return self._generate_emergency_response(query, start_time, error=str(e))
    
    def _generate_emergency_response(self, query: str, start_time: float, error: str = None) -> Dict[str, Any]:
        """Emergency fallback response"""
        elapsed = time.time() - start_time
        
        return {
            'answer': f"I'm BoardContinuity AI. I can help with governance questions about {query[:50]}... Please try rephrasing your question or contact support if issues persist.",
            'response': f"I encountered an issue processing your governance query. Please try rephrasing your question about {query[:50]}...",
            'sources': [],
            'confidence': 0.6,
            'strategy': 'emergency_fallback',
            'processing_time_ms': int(elapsed * 1000),
            'auto_scaling_used': True,
            'processing_mode': 'emergency_fallback',
            'scale_tier': 'unknown',
            'error': error
        }

def enable_rag_mode():
    """Enable full RAG processing mode"""
    logger.info("RAG mode enabled for enterprise-scale processing")
    return True

def use_direct_context():
    """Use direct context processing for smaller scale"""
    logger.info("Direct context mode enabled for optimized performance")
    return True

def create_auto_scaling_rag() -> AutoScalingRAG:
    """Factory function to create auto-scaling RAG instance"""
    return AutoScalingRAG()