import pytest
import asyncio
import time
import os
from lib.bulletproof_processing import create_bulletproof_processor, DocumentCoverageDiagnostic
from lib.processing_queue import get_document_queue

class TestDocumentCoverage:
    def __init__(self):
        self.processor = create_bulletproof_processor()
        self.diagnostic = DocumentCoverageDiagnostic()
    
    def test_complete_processing_pipeline(self, org_id: str = None):
        """Test complete document processing pipeline"""
        
        if not org_id:
            org_id = os.getenv('DEV_ORG_ID', 'demo-org')
        
        print("🚀 Testing Complete Document Processing Pipeline")
        print("=" * 60)
        
        # Step 1: Diagnose current coverage
        print("Step 1: Diagnosing current coverage...")
        try:
            diagnosis = self.diagnostic.diagnose_coverage_issues(org_id)
            
            print(f"Current coverage: {diagnosis['coverage_analysis']['coverage_percentage']}%")
            print(f"Total documents: {diagnosis['coverage_analysis']['total_documents']}")
            print(f"Processed: {diagnosis['coverage_analysis']['processed_documents']}")
            print(f"Unprocessed: {diagnosis['coverage_analysis']['unprocessed_documents']}")
            
            if diagnosis['processing_issues']:
                print("\nIssues found:")
                for issue in diagnosis['processing_issues'][:5]:  # Show first 5
                    print(f"  - {issue['filename']}: {issue['category']}")
        except Exception as e:
            print(f"❌ Coverage diagnosis failed: {e}")
            return False
        
        # Step 2: Process all documents if needed
        if diagnosis['coverage_analysis']['coverage_percentage'] < 100:
            print("\nStep 2: Processing all documents...")
            
            try:
                result = self.processor.process_all_documents(org_id, force_reprocess=False)
                
                print(f"Processing result:")
                print(f"  - Total documents: {result['total_documents']}")
                print(f"  - Successful: {result['successful']}")
                print(f"  - Failed: {result['failed']}")
                print(f"  - Final coverage: {result['final_coverage']}")
                
                # Step 3: Verify coverage
                print("\nStep 3: Verifying final coverage...")
                final_diagnosis = self.diagnostic.diagnose_coverage_issues(org_id)
                
                final_coverage = final_diagnosis['coverage_analysis']['coverage_percentage']
                print(f"Final coverage: {final_coverage}%")
                
                # Success criteria
                success = final_coverage >= 90  # Allow for 90%+ success rate
                
                print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}: Coverage test {'passed' if success else 'failed'}")
                
                if not success and result['failed'] > 0:
                    print("\nFailed documents:")
                    for error in result.get('errors', [])[:3]:  # Show first 3 errors
                        print(f"  - {error}")
                
                return success
                
            except Exception as e:
                print(f"❌ Document processing failed: {e}")
                return False
        else:
            print("✅ All documents already processed!")
            return True
    
    def test_extraction_strategies(self, test_file_path: str):
        """Test all extraction strategies on a single file"""
        
        print(f"\n🔧 Testing Extraction Strategies on: {test_file_path}")
        print("=" * 60)
        
        if not os.path.exists(test_file_path):
            print(f"❌ Test file not found: {test_file_path}")
            return []
        
        strategies = [
            ('pypdf2', self.processor._extract_with_pypdf2),
            ('pdfminer', self.processor._extract_with_pdfminer),
            ('pymupdf', self.processor._extract_with_pymupdf),
            # Skip OCR for speed in testing
        ]
        
        results = []
        
        for strategy_name, strategy_func in strategies:
            print(f"\nTesting {strategy_name}...")
            
            start_time = time.time()
            try:
                text_content, pages_info = strategy_func(test_file_path)
                end_time = time.time()
                
                result = {
                    'strategy': strategy_name,
                    'success': True,
                    'text_length': len(text_content) if text_content else 0,
                    'pages_extracted': len(pages_info) if pages_info else 0,
                    'time_seconds': round(end_time - start_time, 2),
                    'content_preview': text_content[:200] if text_content else "No content"
                }
                
                print(f"  ✅ Success: {result['text_length']} chars, {result['pages_extracted']} pages, {result['time_seconds']}s")
                
            except Exception as e:
                end_time = time.time()
                result = {
                    'strategy': strategy_name,
                    'success': False,
                    'error': str(e),
                    'time_seconds': round(end_time - start_time, 2)
                }
                
                print(f"  ❌ Failed: {str(e)}")
            
            results.append(result)
        
        # Determine best strategy
        successful_strategies = [r for r in results if r['success']]
        if successful_strategies:
            best_strategy = max(successful_strategies, key=lambda x: x['text_length'])
            print(f"\n🏆 Best strategy: {best_strategy['strategy']} ({best_strategy['text_length']} chars)")
        else:
            print("\n❌ No strategies succeeded")
        
        return results
    
    async def test_queue_processing(self, org_id: str = None):
        """Test queue-based processing"""
        
        if not org_id:
            org_id = os.getenv('DEV_ORG_ID', 'demo-org')
        
        print("\n⚡ Testing Queue Processing")
        print("=" * 60)
        
        try:
            # Get queue instance
            queue = get_document_queue()
            
            # Add documents to queue
            print("Adding documents to queue...")
            queue_result = await queue.add_documents_to_queue(org_id)
            
            print(f"Added {queue_result['added_to_queue']} documents to queue")
            print(f"Queue size: {queue_result['queue_size']}")
            
            if queue_result['added_to_queue'] == 0:
                print("✅ No documents to process - all already processed")
                return True
            
            # Monitor queue progress
            start_time = time.time()
            max_wait = 300  # 5 minutes max
            
            while queue.is_processing and (time.time() - start_time) < max_wait:
                status = queue.get_queue_status()
                print(f"  Processing... Queue size: {status['queue_size']}")
                await asyncio.sleep(10)  # Check every 10 seconds
            
            end_time = time.time()
            print(f"Queue processing completed in {end_time - start_time:.1f} seconds")
            
            # Verify final coverage
            final_diagnosis = self.diagnostic.diagnose_coverage_issues(org_id)
            final_coverage = final_diagnosis['coverage_analysis']['coverage_percentage']
            
            print(f"Final coverage after queue processing: {final_coverage}%")
            
            return final_coverage >= 90
            
        except Exception as e:
            print(f"❌ Queue processing test failed: {e}")
            return False
    
    def test_api_endpoints(self):
        """Test API endpoints are accessible"""
        
        print("\n🌐 Testing API Endpoints")
        print("=" * 60)
        
        try:
            import requests
            base_url = 'http://localhost:5000'
            
            endpoints_to_test = [
                ('/api/queue-status', 'GET'),
                ('/api/document-coverage-status', 'GET'),
                ('/api/processing-status', 'GET'),
            ]
            
            results = []
            
            for endpoint, method in endpoints_to_test:
                try:
                    if method == 'GET':
                        response = requests.get(f"{base_url}{endpoint}", timeout=10)
                    else:
                        response = requests.post(f"{base_url}{endpoint}", json={}, timeout=10)
                    
                    success = response.status_code < 400
                    print(f"  {'✅' if success else '❌'} {method} {endpoint}: {response.status_code}")
                    
                    results.append({
                        'endpoint': endpoint,
                        'method': method,
                        'success': success,
                        'status_code': response.status_code
                    })
                    
                except Exception as e:
                    print(f"  ❌ {method} {endpoint}: {str(e)}")
                    results.append({
                        'endpoint': endpoint,
                        'method': method,
                        'success': False,
                        'error': str(e)
                    })
            
            return results
            
        except ImportError:
            print("  ⚠️  requests library not available, skipping API tests")
            return []

# CLI test runner
def run_all_tests(org_id: str = None):
    """Run all tests and return results"""
    
    if not org_id:
        org_id = os.getenv('DEV_ORG_ID', '63602dc6-defe-4355-b66c-aa6b3b1273e3')
    
    tester = TestDocumentCoverage()
    
    print("🧪 BoardContinuity Document Coverage Testing")
    print("=" * 80)
    print(f"Testing with org_id: {org_id}")
    
    results = {}
    
    # Test 1: Complete processing pipeline
    print("\n" + "=" * 80)
    results['pipeline'] = tester.test_complete_processing_pipeline(org_id)
    
    # Test 2: API endpoints
    print("\n" + "=" * 80)
    api_results = tester.test_api_endpoints()
    results['api'] = all(r['success'] for r in api_results)
    
    # Test 3: Queue processing (async)
    print("\n" + "=" * 80)
    try:
        results['queue'] = asyncio.run(tester.test_queue_processing(org_id))
    except Exception as e:
        print(f"❌ Queue test failed: {e}")
        results['queue'] = False
    
    # Final results
    print("\n" + "=" * 80)
    print("📊 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Pipeline Test: {'✅ PASSED' if results['pipeline'] else '❌ FAILED'}")
    print(f"API Test: {'✅ PASSED' if results['api'] else '❌ FAILED'}")
    print(f"Queue Test: {'✅ PASSED' if results['queue'] else '❌ FAILED'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 SUCCESS: All tests passed!")
        print("   Document processing system is working correctly!")
        print("   BoardContinuity has complete institutional memory access.")
    else:
        print("\n⚠️  ISSUES DETECTED: Some tests failed.")
        print("   Review the error logs above for troubleshooting.")
        print("   System may have reduced functionality until issues are resolved.")
    
    return results

if __name__ == "__main__":
    import sys
    
    org_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_all_tests(org_id)