#!/usr/bin/env python3
"""
Enterprise Scale Performance Testing for BoardContinuity MVP
Tests system performance with large document sets and validates enterprise readiness.
"""

import time
import psutil
import os
import logging
from typing import Dict, List, Any
from lib.perfect_rag import PerfectRAG
from lib.supa import supa

class TestEnterpriseScale:
    def __init__(self):
        self.rag = PerfectRAG()
        self.test_org_id = "63602dc6-defe-4355-b66c-aa6b3b1273e3"  # Using existing org
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def test_current_capacity_assessment(self):
        """Assess current database capacity and document counts"""
        
        print("🎯 CURRENT CAPACITY ASSESSMENT")
        print("=" * 50)
        
        try:
            # Check document counts
            docs_response = supa.table("documents").select("*").eq("org_id", self.test_org_id).execute()
            total_documents = len(docs_response.data) if docs_response.data else 0
            
            # Check chunk counts
            chunks_response = supa.table("doc_chunks").select("document_id").eq("org_id", self.test_org_id).execute()
            total_chunks = len(chunks_response.data) if chunks_response.data else 0
            
            # Calculate averages
            avg_chunks_per_doc = total_chunks / total_documents if total_documents > 0 else 0
            
            print(f"📊 Database Metrics:")
            print(f"   Total documents: {total_documents}")
            print(f"   Total chunks: {total_chunks}")
            print(f"   Average chunks per document: {avg_chunks_per_doc:.1f}")
            print(f"   Enterprise scale ready: {total_documents >= 10}")  # Reasonable threshold
            
            return {
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'avg_chunks_per_doc': avg_chunks_per_doc,
                'enterprise_ready': total_documents >= 10
            }
            
        except Exception as e:
            print(f"❌ Capacity assessment failed: {str(e)}")
            return {'error': str(e)}
    
    def test_large_document_set_performance(self):
        """Test performance with current document set"""
        
        print("\n⚡ PERFORMANCE BENCHMARKING")
        print("=" * 50)
        
        # Enterprise-grade test queries
        test_queries = [
            "What are all our membership fees and transfer rules?",
            "Show me decision patterns for budget approvals over 65",
            "What vendor relationships have had issues?",
            "Compare our governance approaches and committee structures",
            "What are the eligibility requirements for different membership categories?"
        ]
        
        performance_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}/5: {query[:60]}...")
            start_time = time.time()
            
            try:
                response = self.rag.generate_perfect_response(self.test_org_id, query)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Extract context information
                context_used = response.get('context_used', {})
                primary_contexts = context_used.get('primary_contexts', [])
                sources = response.get('sources', [])
                
                performance_results.append({
                    'query': query,
                    'response_time_ms': response_time,
                    'success': True,
                    'context_count': len(primary_contexts),
                    'sources_count': len(sources),
                    'confidence': response.get('confidence', 0)
                })
                
                print(f"   ✅ Response time: {response_time:.0f}ms")
                print(f"   📚 Contexts retrieved: {len(primary_contexts)}")
                print(f"   📄 Sources found: {len(sources)}")
                
            except Exception as e:
                performance_results.append({
                    'query': query,
                    'response_time_ms': None,
                    'success': False,
                    'error': str(e)
                })
                print(f"   ❌ Query failed: {str(e)}")
        
        # Analyze results
        successful_queries = [r for r in performance_results if r['success']]
        if successful_queries:
            avg_response_time = sum(r['response_time_ms'] for r in successful_queries) / len(successful_queries)
            max_response_time = max(r['response_time_ms'] for r in successful_queries)
            min_response_time = min(r['response_time_ms'] for r in successful_queries)
            
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Successful queries: {len(successful_queries)}/{len(test_queries)}")
            print(f"   Average response time: {avg_response_time:.0f}ms")
            print(f"   Min response time: {min_response_time:.0f}ms")
            print(f"   Max response time: {max_response_time:.0f}ms")
            print(f"   Enterprise performance: {'✅ READY' if avg_response_time < 5000 else '⚠️ NEEDS OPTIMIZATION'}")
        
        return performance_results
    
    def test_memory_usage_with_large_dataset(self):
        """Test memory usage with complex query processing"""
        
        print("\n🧠 MEMORY USAGE ANALYSIS")
        print("=" * 50)
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run complex query that should retrieve many contexts
        complex_query = """Provide a comprehensive analysis of all our major financial decisions, 
        membership categories, governance patterns, and committee structures with specific 
        examples, fee amounts, and historical context from our institutional documents."""
        
        print(f"🔍 Complex Query: {complex_query[:80]}...")
        
        try:
            start_time = time.time()
            response = self.rag.generate_perfect_response(self.test_org_id, complex_query)
            end_time = time.time()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            response_time = (end_time - start_time) * 1000
            
            # Extract context information
            context_used = response.get('context_used', {})
            primary_contexts = context_used.get('primary_contexts', [])
            
            print(f"   Initial memory: {initial_memory:.1f}MB")
            print(f"   Final memory: {final_memory:.1f}MB")
            print(f"   Memory increase: {memory_increase:.1f}MB")
            print(f"   Response time: {response_time:.0f}ms")
            print(f"   Contexts retrieved: {len(primary_contexts)}")
            print(f"   Memory efficient: {'✅ YES' if memory_increase < 150 else '⚠️ NEEDS OPTIMIZATION'}")
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase,
                'response_time_ms': response_time,
                'contexts_retrieved': len(primary_contexts),
                'memory_efficient': memory_increase < 150
            }
            
        except Exception as e:
            print(f"   ❌ Memory test failed: {str(e)}")
            return {'error': str(e)}
    
    def test_concurrent_query_handling(self):
        """Test system performance under concurrent load"""
        
        print("\n🔄 CONCURRENT LOAD TESTING")
        print("=" * 50)
        
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        concurrent_queries = [
            "What are our membership fees?",
            "Show me budget approval patterns",
            "What are committee structures?",
            "Tell me about governance rules",
            "What are eligibility requirements?"
        ]
        
        def run_query(query):
            start_time = time.time()
            try:
                response = self.rag.generate_perfect_response(self.test_org_id, query)
                end_time = time.time()
                return {
                    'query': query,
                    'success': True,
                    'response_time_ms': (end_time - start_time) * 1000,
                    'sources_count': len(response.get('sources', []))
                }
            except Exception as e:
                return {
                    'query': query,
                    'success': False,
                    'error': str(e)
                }
        
        print(f"🚀 Running {len(concurrent_queries)} concurrent queries...")
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {executor.submit(run_query, query): query for query in concurrent_queries}
            
            for future in as_completed(future_to_query):
                result = future.result()
                results.append(result)
                status = "✅" if result['success'] else "❌"
                time_str = f"{result.get('response_time_ms', 0):.0f}ms" if result['success'] else "FAILED"
                print(f"   {status} {result['query'][:40]}... - {time_str}")
        
        total_time = (time.time() - start_time) * 1000
        successful_results = [r for r in results if r['success']]
        
        print(f"\n📊 Concurrent Performance:")
        print(f"   Total execution time: {total_time:.0f}ms")
        print(f"   Successful queries: {len(successful_results)}/{len(concurrent_queries)}")
        if successful_results:
            avg_time = sum(r['response_time_ms'] for r in successful_results) / len(successful_results)
            print(f"   Average response time: {avg_time:.0f}ms")
            print(f"   Concurrent handling: {'✅ READY' if len(successful_results) == len(concurrent_queries) else '⚠️ NEEDS OPTIMIZATION'}")
        
        return results
    
    def generate_enterprise_report(self):
        """Generate comprehensive enterprise readiness report"""
        
        print("\n" + "="*70)
        print("🏢 ENTERPRISE READINESS ASSESSMENT REPORT")
        print("="*70)
        
        # Run all tests
        capacity_results = self.test_current_capacity_assessment()
        performance_results = self.test_large_document_set_performance()
        memory_results = self.test_memory_usage_with_large_dataset()
        concurrent_results = self.test_concurrent_query_handling()
        
        # Generate overall assessment
        print(f"\n🎯 FINAL ENTERPRISE ASSESSMENT:")
        print("-" * 40)
        
        # Capacity assessment
        if 'error' not in capacity_results:
            capacity_ready = capacity_results.get('enterprise_ready', False)
            print(f"   📊 Data Capacity: {'✅ READY' if capacity_ready else '⚠️ NEEDS MORE DOCUMENTS'}")
            print(f"       Documents: {capacity_results.get('total_documents', 0)}")
            print(f"       Chunks: {capacity_results.get('total_chunks', 0)}")
        
        # Performance assessment
        successful_queries = [r for r in performance_results if r.get('success', False)]
        if successful_queries:
            avg_time = sum(r['response_time_ms'] for r in successful_queries) / len(successful_queries)
            performance_ready = avg_time < 5000 and len(successful_queries) == len(performance_results)
            print(f"   ⚡ Query Performance: {'✅ READY' if performance_ready else '⚠️ NEEDS OPTIMIZATION'}")
            print(f"       Average response: {avg_time:.0f}ms")
            print(f"       Success rate: {len(successful_queries)}/{len(performance_results)}")
        
        # Memory assessment
        if 'error' not in memory_results:
            memory_ready = memory_results.get('memory_efficient', False)
            print(f"   🧠 Memory Efficiency: {'✅ READY' if memory_ready else '⚠️ NEEDS OPTIMIZATION'}")
            print(f"       Memory increase: {memory_results.get('memory_increase_mb', 0):.1f}MB")
        
        # Concurrent assessment
        concurrent_successful = [r for r in concurrent_results if r.get('success', False)]
        concurrent_ready = len(concurrent_successful) == len(concurrent_results)
        print(f"   🔄 Concurrent Load: {'✅ READY' if concurrent_ready else '⚠️ NEEDS OPTIMIZATION'}")
        print(f"       Concurrent success: {len(concurrent_successful)}/{len(concurrent_results)}")
        
        # Overall verdict
        all_ready = all([
            capacity_results.get('enterprise_ready', False),
            len(successful_queries) == len(performance_results) if successful_queries else False,
            memory_results.get('memory_efficient', False) if 'error' not in memory_results else False,
            concurrent_ready
        ])
        
        print(f"\n🏆 OVERALL ENTERPRISE READINESS: {'✅ READY FOR PRODUCTION' if all_ready else '⚠️ OPTIMIZATION RECOMMENDED'}")
        
        return {
            'capacity': capacity_results,
            'performance': performance_results,
            'memory': memory_results,
            'concurrent': concurrent_results,
            'enterprise_ready': all_ready
        }

# Run the enterprise scale test
if __name__ == "__main__":
    print("🚀 BoardContinuity Enterprise Scale Testing")
    print("Testing system performance and enterprise readiness...")
    print()
    
    tester = TestEnterpriseScale()
    results = tester.generate_enterprise_report()
    
    print(f"\n✨ Enterprise testing complete!")
    print(f"Review the assessment above for optimization recommendations.")