#!/usr/bin/env python3
"""
BoardContinuity AI Enterprise Deployment Verification System
Comprehensive pre-deployment checks for production readiness
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentVerifier:
    """Comprehensive deployment verification for enterprise BoardContinuity AI"""
    
    def __init__(self):
        self.results = []
        self.critical_failures = []
        self.warnings = []
        
    def verify_environment_variables(self) -> Tuple[bool, str]:
        """Verify all required environment variables are set"""
        required_vars = [
            'OPENAI_API_KEY',
            'DATABASE_URL',
            'SUPABASE_URL', 
            'SUPABASE_SERVICE_ROLE',
            'DEV_ORG_ID',
            'DEV_USER_ID'
        ]
        
        missing = []
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            return False, f"Missing environment variables: {', '.join(missing)}"
        return True, "All environment variables configured"
    
    def verify_database_connectivity(self) -> Tuple[bool, str]:
        """Test database connection and basic operations"""
        try:
            from lib.supa import supa
            
            # Test basic query
            result = supa.table("documents").select("id").limit(1).execute()
            
            # Test document table structure
            if result.data is not None:
                return True, "Database connectivity verified"
            else:
                return False, "Database query returned no data structure"
                
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"
    
    def verify_ai_services(self) -> Tuple[bool, str]:
        """Test OpenAI API connectivity and model access"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Test API connectivity with minimal request
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            
            if response.choices and response.choices[0].message:
                return True, "OpenAI API connectivity verified"
            else:
                return False, "OpenAI API response invalid"
                
        except Exception as e:
            return False, f"OpenAI API test failed: {str(e)}"
    
    def verify_enterprise_agent_system(self) -> Tuple[bool, str]:
        """Test enterprise agent initialization and basic functionality"""
        try:
            from lib.enterprise_rag_agent import create_enterprise_rag_agent
            
            # Initialize agent
            agent = create_enterprise_rag_agent()
            
            # Check all subsystems
            checks = []
            if hasattr(agent, 'guardrails_enabled') and agent.guardrails_enabled:
                checks.append("Guardrails")
            if hasattr(agent, 'committee_agents_enabled') and agent.committee_agents_enabled:
                checks.append("Committee Agents")
            if hasattr(agent, 'intervention_enabled') and agent.intervention_enabled:
                checks.append("Human Intervention")
            if hasattr(agent, 'monitoring_enabled') and agent.monitoring_enabled:
                checks.append("Performance Monitoring")
            
            if len(checks) >= 3:  # At least 3 of 4 systems should be operational
                return True, f"Enterprise agent operational with: {', '.join(checks)}"
            else:
                return False, f"Insufficient enterprise systems active: {', '.join(checks)}"
                
        except Exception as e:
            return False, f"Enterprise agent initialization failed: {str(e)}"
    
    def verify_document_processing(self) -> Tuple[bool, str]:
        """Verify document processing capabilities"""
        try:
            from lib.supa import supa
            
            # Check for existing documents
            docs_result = supa.table("documents").select("id,status").limit(10).execute()
            
            if docs_result.data:
                processed_count = len([d for d in docs_result.data if d.get('status') == 'processed'])
                total_count = len(docs_result.data)
                
                if processed_count > 0:
                    return True, f"Document processing verified: {processed_count}/{total_count} documents processed"
                else:
                    return False, f"No processed documents found (0/{total_count})"
            else:
                return True, "No documents found - processing system ready"
                
        except Exception as e:
            return False, f"Document processing verification failed: {str(e)}"
    
    def verify_api_endpoints(self) -> Tuple[bool, str]:
        """Verify critical API endpoints are accessible"""
        try:
            # Import Flask app to check route registration
            from app import app
            
            # Check for critical enterprise endpoints
            critical_routes = [
                '/api/query',
                '/api/enterprise-query', 
                '/api/agent-status',
                '/api/health'
            ]
            
            registered_routes = [rule.rule for rule in app.url_map.iter_rules()]
            missing_routes = [route for route in critical_routes if route not in registered_routes]
            
            if not missing_routes:
                return True, f"All critical API endpoints registered ({len(critical_routes)} endpoints)"
            else:
                return False, f"Missing API endpoints: {', '.join(missing_routes)}"
                
        except Exception as e:
            return False, f"API endpoint verification failed: {str(e)}"
    
    def verify_performance_requirements(self) -> Tuple[bool, str]:
        """Test performance requirements for enterprise deployment"""
        try:
            from lib.enterprise_rag_agent import create_enterprise_rag_agent
            
            agent = create_enterprise_rag_agent()
            test_query = "What is our governance structure?"
            
            # Performance test
            start_time = time.time()
            response = agent.run(os.getenv('DEV_ORG_ID'), test_query)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Check performance criteria
            if response_time < 5000:  # 5 second max for deployment test
                confidence = response.get('confidence', 0)
                if confidence > 0.5:  # Reasonable confidence threshold
                    return True, f"Performance verified: {response_time:.0f}ms response, {confidence:.2f} confidence"
                else:
                    return False, f"Low confidence response: {confidence:.2f}"
            else:
                return False, f"Slow response time: {response_time:.0f}ms (>5000ms)"
                
        except Exception as e:
            return False, f"Performance test failed: {str(e)}"
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all verification checks and generate deployment report"""
        
        print("🔍 BOARDCONTINUITY AI ENTERPRISE DEPLOYMENT VERIFICATION")
        print("=" * 65)
        
        verifications = [
            ("Environment Variables", self.verify_environment_variables),
            ("Database Connectivity", self.verify_database_connectivity),
            ("AI Services", self.verify_ai_services),
            ("Enterprise Agent System", self.verify_enterprise_agent_system),
            ("Document Processing", self.verify_document_processing),
            ("API Endpoints", self.verify_api_endpoints),
            ("Performance Requirements", self.verify_performance_requirements)
        ]
        
        results = {}
        passed = 0
        failed = 0
        
        for check_name, check_func in verifications:
            try:
                success, message = check_func()
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{status} {check_name}: {message}")
                
                results[check_name] = {
                    'passed': success,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                
                if success:
                    passed += 1
                else:
                    failed += 1
                    if check_name in ["Environment Variables", "Database Connectivity", "AI Services"]:
                        self.critical_failures.append(f"{check_name}: {message}")
                        
            except Exception as e:
                error_msg = f"Verification error: {str(e)}"
                print(f"❌ FAIL {check_name}: {error_msg}")
                results[check_name] = {
                    'passed': False,
                    'message': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
                failed += 1
                self.critical_failures.append(f"{check_name}: {error_msg}")
        
        # Overall assessment
        total_checks = len(verifications)
        pass_rate = passed / total_checks
        
        print("\n" + "=" * 65)
        print(f"📊 VERIFICATION SUMMARY: {passed}/{total_checks} checks passed ({pass_rate:.1%})")
        
        deployment_ready = pass_rate >= 0.85 and len(self.critical_failures) == 0
        
        if deployment_ready:
            print("🚀 DEPLOYMENT STATUS: READY FOR PRODUCTION")
            print("   All critical systems verified and performance meets enterprise standards")
        else:
            print("⚠️  DEPLOYMENT STATUS: NOT READY")
            if self.critical_failures:
                print("   Critical failures detected:")
                for failure in self.critical_failures:
                    print(f"   • {failure}")
        
        return {
            'deployment_ready': deployment_ready,
            'pass_rate': pass_rate,
            'passed_checks': passed,
            'failed_checks': failed,
            'total_checks': total_checks,
            'critical_failures': self.critical_failures,
            'detailed_results': results,
            'verification_timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    verifier = DeploymentVerifier()
    report = verifier.run_comprehensive_verification()
    
    # Exit with appropriate code
    sys.exit(0 if report['deployment_ready'] else 1)