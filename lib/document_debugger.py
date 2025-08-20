import os
import logging
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime
import mimetypes

class DocumentDebugger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def comprehensive_document_analysis(self, org_id: str) -> Dict[str, Any]:
        """Comprehensive analysis of all documents and processing issues"""
        
        analysis = {
            'document_inventory': [],
            'processing_analysis': {},
            'file_system_check': {},
            'database_integrity': {},
            'error_analysis': [],
            'recommendations': []
        }
        
        try:
            # Get complete document inventory from database
            analysis['document_inventory'] = self._get_complete_document_inventory(org_id)
            
            # Analyze each document's processing status
            analysis['processing_analysis'] = self._analyze_processing_status(analysis['document_inventory'])
            
            # Check file system accessibility
            analysis['file_system_check'] = self._check_file_system_access(analysis['document_inventory'])
            
            # Check database integrity
            analysis['database_integrity'] = self._check_database_integrity(org_id)
            
            # Analyze specific errors
            analysis['error_analysis'] = self._analyze_processing_errors(analysis['document_inventory'])
            
            # Generate actionable recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
        except Exception as e:
            analysis['critical_error'] = str(e)
            self.logger.error(f"Document analysis failed: {e}")
            traceback.print_exc()
        
        return analysis
    
    def _get_complete_document_inventory(self, org_id: str) -> List[Dict[str, Any]]:
        """Get complete inventory of all documents"""
        
        try:
            from lib.supa import supa
            
            # Get all documents with their processing status
            documents_response = supa.table('documents').select('*').eq('org_id', org_id).execute()
            documents = documents_response.data if documents_response.data else []
            
            # Get chunk counts for each document
            for doc in documents:
                chunks_response = supa.table('doc_chunks').select('id').eq('document_id', doc['id']).execute()
                doc['actual_chunk_count'] = len(chunks_response.data) if chunks_response.data else 0
                
                # Add file analysis
                doc['file_analysis'] = self._analyze_single_file(doc)
                
                # Add processing diagnosis
                doc['processing_diagnosis'] = self._diagnose_processing_issue(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to get document inventory: {e}")
            return []
    
    def _analyze_single_file(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single file for processing issues"""
        
        file_path = doc.get('file_path', '')
        filename = doc.get('filename', '')
        storage_path = doc.get('storage_path', '')
        
        analysis = {
            'exists': False,
            'readable': False,
            'size_bytes': 0,
            'mime_type': None,
            'is_pdf': False,
            'issues': []
        }
        
        try:
            # Check if local file exists
            if file_path and os.path.exists(file_path):
                analysis['exists'] = True
                
                try:
                    with open(file_path, 'rb') as f:
                        first_kb = f.read(1024)
                        analysis['readable'] = len(first_kb) > 0
                        
                    analysis['size_bytes'] = os.path.getsize(file_path)
                    
                    mime_type, _ = mimetypes.guess_type(file_path)
                    analysis['mime_type'] = mime_type
                    analysis['is_pdf'] = mime_type == 'application/pdf' or filename.lower().endswith('.pdf')
                    
                    if analysis['size_bytes'] == 0:
                        analysis['issues'].append('File is empty (0 bytes)')
                    elif analysis['size_bytes'] < 1000:
                        analysis['issues'].append(f'File very small ({analysis["size_bytes"]} bytes)')
                    
                    if not analysis['is_pdf']:
                        analysis['issues'].append(f'Not a PDF file (detected: {mime_type})')
                    
                    if analysis['is_pdf'] and not first_kb.startswith(b'%PDF'):
                        analysis['issues'].append('PDF header missing - file may be corrupted')
                    
                except Exception as e:
                    analysis['readable'] = False
                    analysis['issues'].append(f'Cannot read file: {str(e)}')
            
            # Check if it's in Supabase storage
            elif storage_path:
                analysis['issues'].append('File stored in Supabase - need to download for processing')
                analysis['is_pdf'] = filename.lower().endswith('.pdf')
            else:
                analysis['issues'].append(f'No file path or storage path found')
                
        except Exception as e:
            analysis['issues'].append(f'File analysis error: {str(e)}')
        
        return analysis
    
    def _diagnose_processing_issue(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose why a document failed to process"""
        
        diagnosis = {
            'status': doc.get('status', 'unknown'),
            'issue_category': 'unknown',
            'specific_issue': None,
            'recommended_action': None,
            'can_retry': False
        }
        
        actual_chunks = doc.get('actual_chunk_count', 0)
        file_analysis = doc.get('file_analysis', {})
        
        # Categorize the issue
        if not file_analysis.get('exists', False) and not doc.get('storage_path'):
            diagnosis['issue_category'] = 'file_missing'
            diagnosis['specific_issue'] = 'File not found on disk or in storage'
            diagnosis['recommended_action'] = 'Delete document record or re-upload file'
            diagnosis['can_retry'] = False
            
        elif file_analysis.get('issues'):
            diagnosis['issue_category'] = 'file_quality'
            diagnosis['specific_issue'] = '; '.join(file_analysis['issues'])
            diagnosis['recommended_action'] = 'Process with bulletproof extraction system'
            diagnosis['can_retry'] = True
            
        elif actual_chunks == 0:
            diagnosis['issue_category'] = 'never_processed'
            diagnosis['specific_issue'] = 'Document uploaded but never processed'
            diagnosis['recommended_action'] = 'Process immediately with automated queue'
            diagnosis['can_retry'] = True
            
        else:
            diagnosis['issue_category'] = 'processed_successfully'
            diagnosis['specific_issue'] = None
            diagnosis['recommended_action'] = None
            diagnosis['can_retry'] = False
        
        return diagnosis
    
    def _analyze_processing_status(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze overall processing status"""
        
        analysis = {
            'total_documents': len(documents),
            'processed_successfully': 0,
            'never_processed': 0,
            'failed_processing': 0,
            'file_issues': 0,
            'can_retry_count': 0
        }
        
        for doc in documents:
            diagnosis = doc.get('processing_diagnosis', {})
            category = diagnosis.get('issue_category', 'unknown')
            
            if category == 'processed_successfully':
                analysis['processed_successfully'] += 1
            elif category == 'never_processed':
                analysis['never_processed'] += 1
            elif category in ['file_quality', 'processing_error']:
                analysis['failed_processing'] += 1
            else:
                analysis['file_issues'] += 1
            
            if diagnosis.get('can_retry', False):
                analysis['can_retry_count'] += 1
        
        analysis['coverage_percentage'] = (analysis['processed_successfully'] / analysis['total_documents'] * 100) if analysis['total_documents'] > 0 else 0
        
        return analysis
    
    def _check_file_system_access(self, documents: List[Dict]) -> Dict[str, Any]:
        """Check file system access issues"""
        
        check = {
            'total_files': len(documents),
            'files_exist': 0,
            'files_readable': 0,
            'files_missing': 0,
            'storage_only': 0,
            'path_issues': []
        }
        
        for doc in documents:
            file_analysis = doc.get('file_analysis', {})
            
            if file_analysis.get('exists'):
                check['files_exist'] += 1
                if file_analysis.get('readable'):
                    check['files_readable'] += 1
            elif doc.get('storage_path'):
                check['storage_only'] += 1
            else:
                check['files_missing'] += 1
                check['path_issues'].append({
                    'filename': doc.get('filename'),
                    'path': doc.get('file_path'),
                    'issue': 'File not found'
                })
        
        return check
    
    def _check_database_integrity(self, org_id: str) -> Dict[str, Any]:
        """Check database integrity issues"""
        
        integrity = {
            'orphaned_chunks': 0,
            'documents_without_chunks': 0,
            'issues': []
        }
        
        try:
            from lib.supa import supa
            
            # Check for documents without chunks
            documents_response = supa.table('documents').select('id').eq('org_id', org_id).execute()
            documents = documents_response.data if documents_response.data else []
            
            for doc in documents:
                chunks_response = supa.table('doc_chunks').select('id').eq('document_id', doc['id']).execute()
                if not chunks_response.data:
                    integrity['documents_without_chunks'] += 1
            
        except Exception as e:
            integrity['issues'].append(f"Database check failed: {str(e)}")
        
        return integrity
    
    def _analyze_processing_errors(self, documents: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze processing errors to find patterns"""
        
        errors = []
        
        for doc in documents:
            diagnosis = doc.get('processing_diagnosis', {})
            
            if diagnosis.get('issue_category') not in ['processed_successfully', 'unknown']:
                errors.append({
                    'filename': doc.get('filename'),
                    'category': diagnosis.get('issue_category'),
                    'issue': diagnosis.get('specific_issue'),
                    'action': diagnosis.get('recommended_action'),
                    'can_retry': diagnosis.get('can_retry', False)
                })
        
        return errors
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        processing = analysis.get('processing_analysis', {})
        file_check = analysis.get('file_system_check', {})
        errors = analysis.get('error_analysis', [])
        
        # Coverage recommendations
        coverage = processing.get('coverage_percentage', 0)
        if coverage < 100:
            recommendations.append(f"PRIORITY: Achieve 100% document coverage (currently {coverage:.1f}%)")
        
        # Processing recommendations
        never_processed = processing.get('never_processed', 0)
        if never_processed > 0:
            recommendations.append(f"Process {never_processed} documents that have never been processed")
        
        failed_processing = processing.get('failed_processing', 0)
        if failed_processing > 0:
            recommendations.append(f"Retry processing for {failed_processing} documents that failed")
        
        # File system recommendations
        storage_only = file_check.get('storage_only', 0)
        if storage_only > 0:
            recommendations.append(f"Download and process {storage_only} documents from Supabase storage")
        
        # Specific error recommendations
        file_quality_issues = [e for e in errors if e['category'] == 'file_quality']
        if file_quality_issues:
            recommendations.append(f"Use bulletproof processing for {len(file_quality_issues)} files with quality issues")
        
        # Immediate actions
        can_retry = processing.get('can_retry_count', 0)
        if can_retry > 0:
            recommendations.append(f"IMMEDIATE: Run automated processing queue to fix {can_retry} documents")
        
        return recommendations

def create_document_debugger() -> DocumentDebugger:
    """Factory function to create document debugger"""
    return DocumentDebugger()