#!/usr/bin/env python3
"""
API Validation Tests for BoardContinuity AI
Tests all endpoints for proper functionality
"""

import requests
import json
import time
import os
from typing import Dict, List, Any

class APIValidator:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.org_id = os.getenv('DEV_ORG_ID', '63602dc6-defe-4355-b66c-aa6b3b1273e3')
        self.test_results = []
    
    def test_endpoint(self, endpoint: str, method: str = 'GET', payload: Dict = None, expected_status: int = 200) -> bool:
        """Test a single API endpoint"""
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            start_time = time.time()
            
            if method.upper() == 'GET':
                response = requests.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, json=payload or {}, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.time()
            response_time = round(end_time - start_time, 3)
            
            success = response.status_code == expected_status
            
            result = {
                'endpoint': endpoint,
                'method': method,
                'success': success,
                'status_code': response.status_code,
                'response_time': response_time,
                'content_type': response.headers.get('content-type', ''),
                'response_size': len(response.content)
            }
            
            # Try to parse JSON response
            try:
                result['json_response'] = response.json()
            except:
                result['json_response'] = None
            
            # Add error details if failed
            if not success:
                result['error'] = response.text[:500]  # First 500 chars
            
            self.test_results.append(result)
            
            status_icon = "✅" if success else "❌"
            print(f"{status_icon} {method} {endpoint} - {response.status_code} ({response_time}s)")
            
            return success
            
        except Exception as e:
            result = {
                'endpoint': endpoint,
                'method': method,
                'success': False,
                'error': str(e),
                'response_time': 0
            }
            
            self.test_results.append(result)
            print(f"❌ {method} {endpoint} - Exception: {str(e)}")
            return False
    
    def test_core_endpoints(self) -> Dict[str, bool]:
        """Test core application endpoints"""
        
        print("\n🏠 Testing Core Endpoints")
        print("-" * 40)
        
        results = {}
        
        # Home page
        results['home'] = self.test_endpoint('/', 'GET', expected_status=200)
        
        # Health check (if exists)
        results['health'] = self.test_endpoint('/health', 'GET', expected_status=200)
        
        return results
    
    def test_document_processing_endpoints(self) -> Dict[str, bool]:
        """Test document processing endpoints"""
        
        print("\n📄 Testing Document Processing Endpoints")
        print("-" * 40)
        
        results = {}
        
        # Document coverage status
        results['coverage_status'] = self.test_endpoint(
            f'/api/document-coverage-status?org_id={self.org_id}', 
            'GET'
        )
        
        # Processing status
        results['processing_status'] = self.test_endpoint(
            '/api/processing-status', 
            'POST',
            {'org_id': self.org_id}
        )
        
        # Diagnose coverage
        results['diagnose_coverage'] = self.test_endpoint(
            '/api/diagnose-coverage',
            'POST', 
            {'org_id': self.org_id}
        )
        
        return results
    
    def test_queue_endpoints(self) -> Dict[str, bool]:
        """Test processing queue endpoints"""
        
        print("\n⚡ Testing Queue Endpoints")
        print("-" * 40)
        
        results = {}
        
        # Queue status
        results['queue_status'] = self.test_endpoint('/api/queue-status', 'GET')
        
        # Queue documents (add to queue)
        results['queue_documents'] = self.test_endpoint(
            '/api/queue-documents',
            'POST',
            {'org_id': self.org_id, 'document_ids': []}
        )
        
        return results
    
    def test_ai_chat_endpoints(self) -> Dict[str, bool]:
        """Test AI chat and query endpoints"""
        
        print("\n🤖 Testing AI Chat Endpoints") 
        print("-" * 40)
        
        results = {}
        
        # Basic query
        results['query'] = self.test_endpoint(
            '/api/query',
            'POST',
            {
                'question': 'What are the membership categories?',
                'org_id': self.org_id
            }
        )
        
        # Enterprise query (if available)
        results['enterprise_query'] = self.test_endpoint(
            '/api/enterprise-query',
            'POST',
            {
                'question': 'What are the membership fees?',
                'org_id': self.org_id,
                'use_committee_agents': True
            }
        )
        
        return results
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests"""
        
        print("🧪 BoardContinuity API Comprehensive Validation")
        print("=" * 80)
        print(f"Testing against: {self.base_url}")
        print(f"Using org_id: {self.org_id}")
        
        # Test all endpoint categories
        core_results = self.test_core_endpoints()
        doc_results = self.test_document_processing_endpoints()
        queue_results = self.test_queue_endpoints()
        ai_results = self.test_ai_chat_endpoints()
        
        # Compile summary
        all_results = {
            'core': core_results,
            'document_processing': doc_results,
            'queue': queue_results,
            'ai_chat': ai_results
        }
        
        # Calculate statistics
        total_tests = sum(len(category) for category in all_results.values())
        passed_tests = sum(
            sum(1 for result in category.values() if result)
            for category in all_results.values()
        )
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        
        for category_name, category_results in all_results.items():
            category_passed = sum(1 for r in category_results.values() if r)
            category_total = len(category_results)
            category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
            
            status = "✅ PASSED" if category_rate == 100 else f"⚠️  {category_rate:.1f}% PASSED"
            print(f"{category_name.upper()}: {status} ({category_passed}/{category_total})")
        
        print(f"\nOVERALL: {success_rate:.1f}% PASSED ({passed_tests}/{total_tests})")
        
        # Performance analysis
        if self.test_results:
            avg_response_time = sum(r.get('response_time', 0) for r in self.test_results) / len(self.test_results)
            max_response_time = max(r.get('response_time', 0) for r in self.test_results)
            
            print(f"\nPERFORMANCE:")
            print(f"Average response time: {avg_response_time:.3f}s")
            print(f"Maximum response time: {max_response_time:.3f}s")
        
        # Final verdict
        if success_rate >= 90:
            print(f"\n🎉 EXCELLENT: API validation passed with {success_rate:.1f}% success rate")
            print("   All critical endpoints are functional")
        elif success_rate >= 75:
            print(f"\n✅ GOOD: API validation passed with {success_rate:.1f}% success rate")
            print("   Most endpoints are functional, some minor issues detected")
        elif success_rate >= 50:
            print(f"\n⚠️  FAIR: API validation partial with {success_rate:.1f}% success rate")
            print("   Core functionality works but several endpoints have issues")
        else:
            print(f"\n❌ POOR: API validation failed with {success_rate:.1f}% success rate")
            print("   Significant issues detected, system may not be fully functional")
        
        return {
            'results': all_results,
            'statistics': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time if self.test_results else 0,
                'max_response_time': max_response_time if self.test_results else 0
            },
            'detailed_results': self.test_results
        }

def main():
    """Main validation runner"""
    
    # Allow custom base URL from command line
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    validator = APIValidator(base_url)
    results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    success_rate = results['statistics']['success_rate']
    exit_code = 0 if success_rate >= 75 else 1
    
    return exit_code

if __name__ == "__main__":
    exit(main())