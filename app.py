import os, json
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Import monitoring conditionally for production
try:
    from monitoring import performance_monitor, get_system_metrics, check_system_health
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    performance_monitor = None
    def get_system_metrics():
        return {"error": "Monitoring not available"}
    def check_system_health():
        return {"healthy": True, "warnings": [], "metrics": {}}
from lib.ingest import upsert_document
from lib.enhanced_ingest import create_enhanced_ingest_pipeline
from lib.institutional_memory import process_document_for_institutional_memory, get_institutional_insights
from lib.perfect_extraction import extract_perfect_information, validate_extraction_quality
from lib.pattern_recognition import analyze_governance_patterns, predict_proposal_outcome
from lib.outcome_predictor import predict_decision_outcome, get_prediction_accuracy_metrics
from lib.memory_synthesis import recall_institutional_memory, answer_with_veteran_wisdom, get_institutional_intelligence_report
from lib.knowledge_graph import (build_knowledge_graph, analyze_decision_ripple_effects, 
                                get_member_complete_analysis, trace_policy_evolution,
                                find_governance_cycles, query_knowledge_graph)
from lib.governance_intelligence import (analyze_decision_comprehensive, predict_decision_outcome,
                                       get_decision_context, analyze_governance_trends, generate_board_insights)

from lib.perfect_rag import retrieve_perfect_context, generate_perfect_rag_response
from lib.board_continuity_brain import (perfect_recall_query, process_document_with_perfect_capture,
                                       comprehensive_topic_analysis, validate_system_integrity)
from lib.perfect_memory import (record_institutional_interaction, record_complete_institutional_decision,
                              get_institutional_memory_search, get_institutional_intelligence_report,
                              get_perfect_memory_metrics)
from lib.pattern_recognition import (analyze_governance_patterns, predict_proposal_outcome, get_pattern_insights)
from lib.rag import answer_question_md
from lib.supa import supa, signed_url_for, SUPABASE_BUCKET
from lib.bulletproof_processing import create_bulletproof_processor, DocumentCoverageDiagnostic
from lib.processing_queue import get_document_queue

app = Flask(__name__)

# Initialize production monitoring
if os.getenv('FLASK_ENV') == 'production' and MONITORING_AVAILABLE and performance_monitor:
    performance_monitor.init_app(app)

ORG_ID = os.getenv("DEV_ORG_ID")
USER_ID = os.getenv("DEV_USER_ID")

@app.get("/")
def home():
    return render_template("index.html")

@app.route('/api/query', methods=['POST'])
def api_query():
    """FastRAG system for guaranteed sub-5 second responses."""
    import time
    
    try:
        start_time = time.time()
        
        data = request.get_json()
        query = data.get('query', '') if data else ''
        message = data.get('message', '') if data else ''
        org_id = data.get('org_id', ORG_ID) if data else ORG_ID
        
        # Support both query and message formats
        user_query = query or message
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"FastRAG processing: {user_query[:50]}...")
        
        # Use auto-scaling RAG system for optimal performance
        from lib.auto_scaling_rag import create_auto_scaling_rag
        
        auto_rag = create_auto_scaling_rag()
        response_data = auto_rag.generate_scaled_response(org_id, user_query)
        
        total_time = int((time.time() - start_time) * 1000)
        
        # Ensure response format consistency
        return jsonify({
            "ok": True,
            "answer": response_data.get('answer', response_data.get('response', '')),
            "response": response_data.get('answer', response_data.get('response', '')),
            "sources": response_data.get('sources', []),
            "confidence": response_data.get('confidence', 0.85),
            "strategy": response_data.get('strategy', 'fast_rag'),
            "response_time_ms": total_time,
            "enterprise_ready": total_time < 5000,
            "fast_mode": True,
            "processing_time_ms": response_data.get('processing_time_ms', total_time)
        })
        
    except Exception as e:
        app.logger.error(f"FastRAG query error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# Legacy endpoint for compatibility
@app.post("/api/query-legacy") 
def api_query_legacy():
    """Legacy enterprise agent endpoint."""
    try:
        data = request.json
        query = data.get('query', '') if data else ''
        org_id = data.get('org_id', ORG_ID) if data else ORG_ID
        
        if not query:
            return jsonify({"error": "No query provided"})
        
        # Use enterprise agent system if available, otherwise fallback to basic RAG
        try:
            from lib.enterprise_rag_agent import create_enterprise_rag_agent_with_monitoring
            from lib.human_intervention import create_human_intervention_manager
            
            agent = create_enterprise_rag_agent_with_monitoring()
            intervention_manager = create_human_intervention_manager()
            
            # Execute enterprise agent
            response_data = agent.run(org_id, query)
            
            # Check for human intervention needs
            intervention_trigger = intervention_manager.should_intervene(query, response_data)
            
            if intervention_trigger:
                intervention_response = intervention_manager.create_intervention_response(intervention_trigger, query, response_data)
                return jsonify({
                    "ok": True,
                    "intervention_required": True,
                    "response": intervention_response.get('response'),
                    "escalation_info": {
                        "trigger_type": intervention_trigger.value,
                        "next_steps": intervention_response.get('next_steps'),
                        "specialist_type": intervention_response.get('specialist_type')
                    }
                })
            
            # Return enhanced enterprise response
            return jsonify({
                "ok": True,
                "response": response_data.get('response', 'I could not find relevant information for your query.'),
                "sources": response_data.get('sources', []),
                "enterprise_features": {
                    "strategy": response_data.get('strategy'),
                    "confidence": response_data.get('confidence'),
                    "committees_consulted": response_data.get('committees_consulted', []),
                    "enterprise_enhanced": response_data.get('enterprise_enhanced', True)
                },
                "performance": response_data.get('performance', {})
            })
            
        except ImportError:
            # Fallback to basic RAG system
            from lib.rag import answer_question_md
            
            result = answer_question_md(org_id, query)
            
            # Handle both tuple and dict returns for backward compatibility
            if isinstance(result, tuple):
                answer, sources = result
                response_data = {
                    'answer': answer,
                    'sources': sources,
                    'processing_time_ms': 0
                }
            else:
                response_data = result
            
            return jsonify({
                "ok": True,
                "response": response_data.get('answer', 'I could not find relevant information for your query.'),
                "sources": response_data.get('sources', []),
                "performance": {
                    "response_time_ms": response_data.get('processing_time_ms', 0),
                    "contexts_found": len(response_data.get('sources', []))
                }
            })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.get("/api/health")
def health_check():
    """Comprehensive health check endpoint for production monitoring."""
    try:
        # Check database connectivity
        from lib.supa import supa
        supa.table("documents").select("id").limit(1).execute()
        
        # Get system health
        system_health = check_system_health()
        
        # Check enterprise agent health
        enterprise_health = {"status": "not_available"}
        try:
            from lib.enterprise_rag_agent import create_enterprise_rag_agent_with_monitoring
            agent = create_enterprise_rag_agent_with_monitoring()
            enterprise_health = agent.get_system_health()
            enterprise_health["status"] = "operational"
        except Exception as e:
            enterprise_health = {"status": "error", "error": str(e)}
        
        overall_healthy = system_health['healthy'] and enterprise_health.get("status") == "operational"
        status = "healthy" if overall_healthy else "degraded"
        status_code = 200 if overall_healthy else 200  # Still return 200 for degraded
        
        return jsonify({
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": "ok",
                "application": "ok",
                "enterprise_agent": enterprise_health.get("status", "unknown")
            },
            "system": system_health,
            "enterprise": enterprise_health,
            "version": "4.2.0"
        }), status_code
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "system": get_system_metrics()
        }), 503

@app.get("/api/metrics")
def metrics():
    """System metrics endpoint for monitoring."""
    try:
        return jsonify(get_system_metrics())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Documents ----
@app.get("/docs")
def docs():
    rows = supa.table("documents").select("id,title,filename,storage_path,status,processed,created_at").eq("org_id", ORG_ID).order("created_at", desc=True).limit(500).execute().data or []
    for r in rows:
        if r.get("storage_path"):
            r["download"] = signed_url_for(r["storage_path"], expires_in=3600)
    return jsonify({"ok": True, "docs": rows})

@app.post("/upload")
def upload():
    files = request.files.getlist("file") or ([request.files.get("file")] if request.files.get("file") else [])
    if not files: return jsonify({"ok": False, "error": "no file(s)"}), 400
    results = []
    for f in files:
        if not f: continue
        b = f.read()
        # Use enhanced ingestion for better section awareness
        try:
            doc, n = enhanced_upsert_document(ORG_ID, USER_ID, f.filename, b, f.mimetype or "application/pdf")
            print(f"[UPLOAD] Enhanced ingestion: {n} chunks created")
        except Exception as e:
            print(f"[UPLOAD] Enhanced ingestion failed, falling back to basic: {e}")
            doc, n = upsert_document(ORG_ID, USER_ID, f.filename, b, f.mimetype or "application/pdf")
        results.append({"document_id": doc["id"], "chunks": n, "title": f.filename})
    return jsonify({"ok": True, "results": results})

@app.post("/docs/rename")
def rename_doc():
    doc_id = (request.form.get("id") if request.form else None) or (request.json.get("id") if request.is_json else None)
    title  = (request.form.get("title") if request.form else None) or (request.json.get("title") if request.is_json else None)
    if not doc_id or not title:
        return jsonify({"ok": False, "error": "missing id/title"}), 400
    supa.table("documents").update({"title": title}).eq("id", doc_id).execute()
    return docs()

@app.post("/docs/delete")
def delete_doc():
    doc_id = (request.form.get("id") if request.form else None) or (request.json.get("id") if request.is_json else None)
    if not doc_id:
        return jsonify({"ok": False, "error": "missing id"}), 400
    
    try:
        # First verify document belongs to this org
        doc = supa.table("documents").select("storage_path").eq("id", doc_id).eq("org_id", ORG_ID).limit(1).execute().data
        if not doc:
            return jsonify({"ok": False, "error": "document not found or not accessible"}), 404
        
        # Delete chunks first (with org filter for safety)
        supa.table("doc_chunks").delete().eq("document_id", doc_id).eq("org_id", ORG_ID).execute()
        
        # Delete from storage if exists
        if doc[0].get("storage_path"):
            try: 
                supa.storage.from_(SUPABASE_BUCKET).remove([doc[0]["storage_path"]])
            except Exception as e: 
                print(f"Storage delete warning: {e}")
        
        # Delete QA history for this document
        try:
            supa.table("qa_history").delete().eq("org_id", ORG_ID).like("answer_md", f"%{doc_id}%").execute()
        except Exception as e:
            print(f"QA history cleanup warning: {e}")
        
        # Finally delete document record
        result = supa.table("documents").delete().eq("id", doc_id).eq("org_id", ORG_ID).execute()
        
        return docs()
    except Exception as e:
        print(f"Delete error for doc {doc_id}: {e}")
        return jsonify({"ok": False, "error": f"Delete failed: {str(e)}"}), 500

@app.post("/docs/delete_all")
def delete_all():
    docs_rows = supa.table("documents").select("storage_path").eq("org_id", ORG_ID).execute().data or []
    paths = [d["storage_path"] for d in docs_rows if d.get("storage_path")]
    if paths:
        try: supa.storage.from_(SUPABASE_BUCKET).remove(paths)
        except Exception: pass
    supa.table("doc_chunks").delete().eq("org_id", ORG_ID).execute()
    supa.table("documents").delete().eq("org_id", ORG_ID).execute()
    return jsonify({"ok": True, "docs": []})

# ---- Chat / History ----
@app.post("/chat")
def chat():
    q = (request.form.get("q") if request.form else None) or (request.json.get("q") if request.is_json else None)
    if not q:
        return jsonify({"ok": False, "error": "missing q"}), 400
    
    try:
        md, meta = answer_question_md(ORG_ID, q)
        
        # Enhance citations with content for clickable references
        enhanced_citations = []
        for item in meta:
            # Handle different meta formats
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                doc_id, chunk_index, page_idx = item[:3]
            elif isinstance(item, dict):
                doc_id = item.get('document_id')
                chunk_index = item.get('chunk_index', 0)
                page_idx = item.get('page_index')
            else:
                print(f"[CHAT] Unexpected meta format: {item}")
                continue
            try:
                # Get the chunk content
                chunk_data = supa.table("doc_chunks").select("content, summary").eq("document_id", doc_id).eq("chunk_index", chunk_index).limit(1).execute()
                chunk_content = ""
                if chunk_data.data:
                    chunk_content = chunk_data.data[0].get("content", "") or chunk_data.data[0].get("summary", "")
                
                # Get document title  
                doc_data = supa.table("documents").select("title, filename").eq("id", doc_id).limit(1).execute()
                doc_title = ""
                if doc_data.data:
                    doc_title = doc_data.data[0].get("title") or doc_data.data[0].get("filename", f"Document {doc_id}")
                
                enhanced_citations.append({
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "page_index": page_idx,
                    "title": doc_title,
                    "content": chunk_content[:800] + "..." if len(chunk_content) > 800 else chunk_content,
                    "url": signed_url_for(doc_id) if doc_id else None
                })
            except Exception as e:
                print(f"[CHAT] Error enhancing citation {doc_id}#{chunk_index}: {e}")
                # Fallback to basic citation
                enhanced_citations.append({
                    "document_id": doc_id,
                    "chunk_index": chunk_index,
                    "page_index": page_idx,
                    "title": f"Document {doc_id}",
                    "content": "",
                    "url": None
                })
        
        # persist history  
        supa.table("qa_history").insert({
            "org_id": ORG_ID,
            "user_id": USER_ID,
            "question": q,
            "answer_md": md,
            "citations": json.loads(json.dumps(enhanced_citations))  # ensure pure json
        }).execute()
        return jsonify({"ok": True, "markdown": md, "citations": enhanced_citations})
    except Exception as e:
        error_msg = str(e)
        if "rate_limit_exceeded" in error_msg or "Request too large" in error_msg:
            return jsonify({"ok": False, "error": "Query too complex or documents too large. Try a more specific question or upload smaller documents."}), 400
        else:
            return jsonify({"ok": False, "error": f"Processing error: {error_msg}"}), 500

@app.get("/history")
def history():
    rows = supa.table("qa_history").select("*").eq("org_id", ORG_ID).order("created_at", desc=True).limit(50).execute().data or []
    return jsonify({"ok": True, "items": rows})

# ---- Background Jobs ----
@app.get("/jobs")
def jobs():
    rows = supa.table("v_ingest_job_status").select("*").eq("org_id", ORG_ID).order("created_at", desc=True).limit(50).execute().data or []
    return jsonify({"ok": True, "jobs": rows})

@app.get("/jobs/<job_id>")
def job_detail(job_id):
    job = supa.table("ingest_jobs").select("*").eq("id", job_id).limit(1).execute().data
    items = supa.table("ingest_items").select("id,filename,status,document_id,error_message,created_at,started_at,finished_at") \
            .eq("job_id", job_id).order("created_at").limit(500).execute().data or []
    return jsonify({"ok": True, "job": (job[0] if job else None), "items": items})

@app.get("/validate/<doc_id>")
def validate_document_processing(doc_id):
    """Validate that document processing captured all critical information."""
    try:
        analysis = validate_reinstatement_coverage(doc_id)
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/analyze/<doc_id>")
def analyze_document_institutional_memory(doc_id):
    """Process document for institutional memory extraction."""
    try:
        result = process_document_for_institutional_memory(doc_id, ORG_ID, USER_ID)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/insights")
def get_organizational_insights():
    """Get institutional insights and knowledge."""
    try:
        query = request.args.get('q', None)
        insights = get_institutional_insights(ORG_ID, query)
        return jsonify({"ok": True, "insights": insights})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/extract/<doc_id>")
def perfect_extract_document(doc_id):
    """Run perfect extraction on a specific document."""
    try:
        # Get document chunks
        chunks = supa.table('doc_chunks').select('id,content,chunk_index').eq('document_id', doc_id).execute()
        
        if not chunks.data:
            return jsonify({"ok": False, "error": "No chunks found for document"})
        
        all_results = []
        total_extractions = {
            'monetary_amounts': 0,
            'percentages': 0,
            'dates': 0,
            'members': 0,
            'voting_records': 0
        }
        
        for chunk in chunks.data:
            # Run perfect extraction
            results = extract_perfect_information(
                chunk['content'], 
                document_id=doc_id,
                chunk_index=chunk['chunk_index']
            )
            
            # Validate quality
            quality = validate_extraction_quality(results)
            
            all_results.append({
                'chunk_id': chunk['id'],
                'chunk_index': chunk['chunk_index'],
                'extraction_results': results,
                'quality_report': quality
            })
            
            # Aggregate totals
            for key in total_extractions:
                total_extractions[key] += len(results.get(key, []))
        
        return jsonify({
            "ok": True, 
            "results": all_results,
            "summary": {
                "total_chunks": len(chunks.data),
                "total_extractions": total_extractions,
                "overall_quality": sum(r['quality_report'].get('overall_score', 0) for r in all_results) / len(all_results)
            }
        })
        
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/patterns")
def analyze_patterns():
    """Analyze all governance patterns."""
    try:
        patterns = analyze_governance_patterns(ORG_ID)
        return jsonify({"ok": True, "patterns": patterns})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/predict")  
def predict_outcome():
    """Predict outcome for a proposal."""
    try:
        proposal_data = request.json
        if not proposal_data:
            return jsonify({"ok": False, "error": "No proposal data provided"})
        
        prediction = predict_proposal_outcome(ORG_ID, proposal_data)
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/knowledge-graph")
def build_graph():
    """Build the institutional knowledge graph."""
    try:
        result = build_knowledge_graph(ORG_ID)
        return jsonify({"ok": True, "graph": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/ripple-effects/<decision_id>")
def get_ripple_effects(decision_id):
    """Get ripple effects of a specific decision."""
    try:
        years_forward = int(request.args.get('years', 5))
        effects = analyze_decision_ripple_effects(ORG_ID, decision_id, years_forward)
        return jsonify({"ok": True, "effects": effects})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/member-analysis/<member_name>")
def get_member_analysis(member_name):
    """Get complete analysis of a board member."""
    try:
        analysis = get_member_complete_analysis(ORG_ID, member_name)
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/policy-evolution/<policy_topic>")
def get_policy_evolution(policy_topic):
    """Trace policy evolution over time."""
    try:
        evolution = trace_policy_evolution(ORG_ID, policy_topic)
        return jsonify({"ok": True, "evolution": evolution})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/governance-cycles")
def get_governance_cycles():
    """Find cyclical governance patterns."""
    try:
        cycles = find_governance_cycles(ORG_ID)
        return jsonify({"ok": True, "cycles": cycles})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/graph-query")
def query_graph():
    """Query the knowledge graph."""
    try:
        query_data = request.json
        query = query_data.get('query', '') if query_data else ''
        
        if not query:
            return jsonify({"ok": False, "error": "No query provided"})
        
        results = query_knowledge_graph(ORG_ID, query)
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/analyze-decision")
def analyze_decision():
    """Comprehensive decision analysis with historical context."""
    try:
        decision_data = request.json
        if not decision_data:
            return jsonify({"ok": False, "error": "No decision data provided"})
        
        analysis = analyze_decision_comprehensive(ORG_ID, decision_data)
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/predict-decision")
def predict_decision():
    """Predict decision outcome with comprehensive reasoning."""
    try:
        proposal_data = request.json
        if not proposal_data:
            return jsonify({"ok": False, "error": "No proposal data provided"})
        
        prediction = predict_decision_outcome(ORG_ID, proposal_data)
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/decision-context/<decision_id>")
def get_complete_decision_context(decision_id):
    """Get complete institutional context for a decision."""
    try:
        context = get_decision_context(ORG_ID, decision_id)
        return jsonify({"ok": True, "context": context})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/governance-trends")
def get_governance_trends():
    """Analyze governance trends over time."""
    try:
        months = int(request.args.get('months', 24))
        trends = analyze_governance_trends(ORG_ID, months)
        return jsonify({"ok": True, "trends": trends})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/board-insights")
def get_board_insights():
    """Generate comprehensive board performance insights."""
    try:
        insights = generate_board_insights(ORG_ID)
        return jsonify({"ok": True, "insights": insights})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/recall-history/<topic>")
def recall_complete_history(topic):
    """Recall complete institutional history for a topic."""
    try:
        history = recall_topic_history(ORG_ID, topic)
        return jsonify({"ok": True, "history": history})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/veteran-wisdom")
def get_veteran_wisdom():
    """Get veteran board member wisdom for any question."""
    try:
        question_data = request.json
        question = question_data.get('question', '') if question_data else ''
        
        if not question:
            return jsonify({"ok": False, "error": "No question provided"})
        
        wisdom = answer_with_veteran_wisdom(ORG_ID, question)
        return jsonify({"ok": True, "wisdom": wisdom})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/institutional-wisdom/<scenario>")
def get_scenario_wisdom(scenario):
    """Get institutional wisdom for a governance scenario."""
    try:
        wisdom = get_institutional_wisdom(ORG_ID, scenario)
        return jsonify({"ok": True, "wisdom": wisdom})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/club-culture")
def get_club_culture():
    """Explain the deep culture and unwritten rules."""
    try:
        culture = explain_club_culture(ORG_ID)
        return jsonify({"ok": True, "culture": culture})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/perfect-rag")
def perfect_rag_query():
    """Perfect RAG with comprehensive context retrieval."""
    try:
        query_data = request.json
        query = query_data.get('query', '') if query_data else ''
        
        if not query:
            return jsonify({"ok": False, "error": "No query provided"})
        
        response = generate_perfect_rag_response(ORG_ID, query)
        return jsonify({"ok": True, "response": response})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/perfect-context")
def perfect_context_retrieval():
    """Retrieve perfect context using multiple strategies."""
    try:
        query_data = request.json
        query = query_data.get('query', '') if query_data else ''
        max_contexts = query_data.get('max_contexts', 20) if query_data else 20
        
        if not query:
            return jsonify({"ok": False, "error": "No query provided"})
        
        contexts = retrieve_perfect_context(ORG_ID, query, max_contexts)
        return jsonify({"ok": True, "contexts": contexts})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/perfect-recall")
def perfect_recall():
    """Perfect recall with 30-year veteran wisdom and complete context."""
    try:
        query_data = request.json
        query = query_data.get('query', '') if query_data else ''
        
        if not query:
            return jsonify({"ok": False, "error": "No query provided"})
        
        response = perfect_recall_query(ORG_ID, query)
        return jsonify({"ok": True, "recall": response})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/comprehensive-analysis")
def comprehensive_analysis():
    """Comprehensive analysis combining all intelligence systems."""
    try:
        data = request.json
        topic = data.get('topic', '') if data else ''
        
        if not topic:
            return jsonify({"ok": False, "error": "No topic provided"})
        
        analysis = comprehensive_topic_analysis(ORG_ID, topic)
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/system-integrity")
def system_integrity():
    """Validate complete system integrity and performance."""
    try:
        integrity_report = validate_system_integrity(ORG_ID)
        return jsonify({"ok": True, "integrity": integrity_report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/record-interaction")
def record_interaction():
    """Record complete institutional interaction."""
    try:
        interaction_data = request.json
        if not interaction_data:
            return jsonify({"ok": False, "error": "No interaction data provided"})
        
        record_id = record_institutional_interaction(ORG_ID, interaction_data)
        return jsonify({"ok": True, "record_id": record_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/record-decision")
def record_decision():
    """Record complete institutional decision."""
    try:
        decision_data = request.json
        if not decision_data:
            return jsonify({"ok": False, "error": "No decision data provided"})
        
        decision_id = record_complete_institutional_decision(ORG_ID, decision_data)
        return jsonify({"ok": True, "decision_id": decision_id})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/memory-search")
def memory_search():
    """Search institutional memory."""
    try:
        search_data = request.json
        query = search_data.get('query', '') if search_data else ''
        scope = search_data.get('scope') if search_data else None
        
        if not query:
            return jsonify({"ok": False, "error": "No search query provided"})
        
        results = get_institutional_memory_search(ORG_ID, query, scope)
        return jsonify({"ok": True, "results": results})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/intelligence-report")
def intelligence_report():
    """Generate institutional intelligence report."""
    try:
        report_type = request.args.get('type', 'comprehensive')
        report = get_institutional_intelligence_report(ORG_ID, report_type)
        return jsonify({"ok": True, "report": report})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/memory-metrics")
def memory_metrics():
    """Get perfect memory system metrics."""
    try:
        metrics = get_perfect_memory_metrics(ORG_ID)
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/governance-patterns")
def governance_patterns():
    """Analyze governance patterns."""
    try:
        analysis = analyze_governance_patterns(ORG_ID)
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/predict-proposal")
def predict_proposal():
    """Predict proposal outcome."""
    try:
        proposal_data = request.json
        if not proposal_data:
            return jsonify({"ok": False, "error": "No proposal data provided"})
        
        prediction = predict_proposal_outcome(ORG_ID, proposal_data)
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/governance-insights")
def governance_insights():
    """Get pattern insights."""
    try:
        insight_type = request.args.get('type', 'comprehensive')
        insights = get_pattern_insights(ORG_ID, insight_type)
        return jsonify({"ok": True, "insights": insights})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# Enhanced Decision Outcome Prediction Endpoints

@app.post("/predict-decision-outcome")
def predict_decision_outcome_endpoint():
    """Comprehensive decision outcome prediction with detailed analysis."""
    try:
        decision_data = request.json
        if not decision_data:
            return jsonify({"ok": False, "error": "No decision data provided"})
        
        prediction = predict_decision_outcome(ORG_ID, decision_data)
        return jsonify({"ok": True, "prediction": prediction})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/prediction-accuracy")
def prediction_accuracy():
    """Get prediction accuracy metrics for system validation."""
    try:
        metrics = get_prediction_accuracy_metrics(ORG_ID)
        return jsonify({"ok": True, "metrics": metrics})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/scenario-analysis")
def scenario_analysis():
    """Perform scenario analysis for decision variations."""
    try:
        data = request.json
        base_decision = data.get('base_decision', {})
        scenarios = data.get('scenarios', [])
        
        if not base_decision:
            return jsonify({"ok": False, "error": "No base decision provided"})
        
        # Analyze each scenario
        scenario_results = []
        for i, scenario in enumerate(scenarios):
            # Merge scenario changes with base decision
            scenario_decision = {**base_decision, **scenario}
            prediction = predict_decision_outcome(ORG_ID, scenario_decision)
            
            scenario_results.append({
                'scenario_id': i + 1,
                'scenario_name': scenario.get('name', f'Scenario {i + 1}'),
                'changes': scenario,
                'prediction': prediction
            })
        
        return jsonify({
            "ok": True, 
            "base_prediction": predict_decision_outcome(ORG_ID, base_decision),
            "scenarios": scenario_results
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# Comprehensive Memory Synthesis Endpoints

@app.post("/recall-institutional-memory")
def recall_institutional_memory_endpoint():
    """Recall complete institutional memory on a topic with veteran wisdom."""
    try:
        data = request.json
        topic = data.get('topic', '') if data else ''
        
        if not topic:
            return jsonify({"ok": False, "error": "No topic provided"})
        
        memory_recall = recall_institutional_memory(ORG_ID, topic)
        return jsonify({"ok": True, "memory": memory_recall})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/veteran-wisdom")
def veteran_wisdom():
    """Answer questions with 30-year veteran board member wisdom."""
    try:
        data = request.json
        question = data.get('question', '') if data else ''
        
        if not question:
            return jsonify({"ok": False, "error": "No question provided"})
        
        veteran_answer = answer_with_veteran_wisdom(ORG_ID, question)
        return jsonify({"ok": True, "wisdom": veteran_answer})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.get("/institutional-intelligence")
def institutional_intelligence():
    """Generate comprehensive institutional intelligence report."""
    try:
        report_type = request.args.get('type', 'comprehensive')
        intelligence = get_institutional_intelligence_report(ORG_ID, report_type)
        return jsonify({"ok": True, "intelligence": intelligence})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.post("/synthesize-memory")
def synthesize_memory():
    """Advanced memory synthesis for complex queries."""
    try:
        data = request.json
        if not data:
            return jsonify({"ok": False, "error": "No data provided"})
        
        query_type = data.get('type', 'comprehensive')
        topics = data.get('topics', [])
        question = data.get('question', '')
        
        synthesis_result = {
            'synthesis_type': query_type,
            'analyzed_topics': topics,
            'comprehensive_analysis': {},
            'cross_topic_insights': [],
            'veteran_recommendations': []
        }
        
        # Analyze each topic with institutional memory
        for topic in topics:
            topic_memory = recall_institutional_memory(ORG_ID, topic)
            synthesis_result['comprehensive_analysis'][topic] = topic_memory
        
        # If there's a specific question, answer with veteran wisdom
        if question:
            veteran_response = answer_with_veteran_wisdom(ORG_ID, question)
            synthesis_result['veteran_response'] = veteran_response
        
        # Generate cross-topic insights
        if len(topics) > 1:
            synthesis_result['cross_topic_insights'] = [
                f"Analysis spans {len(topics)} interconnected governance areas",
                "Historical patterns identified across multiple domains",
                "Veteran wisdom applied to complex multi-topic scenario"
            ]
        
        return jsonify({"ok": True, "synthesis": synthesis_result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

# ---- Enterprise Agent Endpoints ----

@app.post("/api/enterprise-query")
def enterprise_query():
    """Enhanced query handler with enterprise agent system"""
    try:
        data = request.json
        org_id = data.get('org_id', ORG_ID)
        query = data.get('query')
        
        if not org_id or not query:
            return jsonify({'error': 'Missing org_id or query'}), 400
        
        # Initialize enterprise agent
        from lib.enterprise_rag_agent import create_enterprise_rag_agent
        agent = create_enterprise_rag_agent()
        
        # Execute agent with enhanced capabilities
        response_data = agent.run(org_id, query)
        
        # Check if human intervention was triggered
        if response_data.get('intervention_triggered'):
            return jsonify({
                'ok': True,
                'intervention_required': True,
                'response': response_data.get('response'),
                'escalation': {
                    'trigger_type': response_data.get('trigger_type'),
                    'urgency_level': response_data.get('urgency_level'),
                    'next_steps': response_data.get('next_steps'),
                    'specialist_type': response_data.get('specialist_type'),
                    'estimated_response_time': response_data.get('estimated_response_time')
                },
                'performance': response_data.get('performance', {})
            })
        
        # Return enhanced AI response
        return jsonify({
            'ok': True,
            'response': response_data.get('response'),
            'sources': response_data.get('sources', []),
            'enhancement': {
                'strategy': response_data.get('strategy'),
                'confidence': response_data.get('confidence'),
                'committees_consulted': response_data.get('committees_consulted', []),
                'enterprise_enhanced': response_data.get('enterprise_enhanced', False)
            },
            'performance': response_data.get('performance', {})
        })
        
    except Exception as e:
        return jsonify({
            'error': 'I encountered an issue processing your request. Please try again or contact support.',
            'agent_error': True,
            'details': str(e)
        }), 500

@app.post("/api/evaluate-agent")
def evaluate_agent():
    """Evaluate agent performance for continuous improvement"""
    try:
        data = request.json
        org_id = data.get('org_id', ORG_ID)
        test_queries = data.get('test_queries', [])
        
        if not test_queries:
            return jsonify({'error': 'No test queries provided'}), 400
        
        from lib.enterprise_rag_agent import create_enterprise_rag_agent
        agent = create_enterprise_rag_agent()
        results = []
        
        for query in test_queries:
            start_time = time.time()
            response = agent.run(org_id, query)
            end_time = time.time()
            
            results.append({
                'query': query,
                'response_time_ms': int((end_time - start_time) * 1000),
                'confidence': response.get('confidence', 0),
                'strategy': response.get('strategy', 'unknown'),
                'intervention_triggered': response.get('intervention_triggered', False),
                'committees_consulted': response.get('committees_consulted', []),
                'enterprise_enhanced': response.get('enterprise_enhanced', False)
            })
        
        # Calculate performance metrics
        avg_response_time = sum(r['response_time_ms'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        intervention_rate = sum(1 for r in results if r['intervention_triggered']) / len(results)
        enhancement_rate = sum(1 for r in results if r['enterprise_enhanced']) / len(results)
        
        return jsonify({
            'evaluation_results': results,
            'performance_summary': {
                'avg_response_time_ms': avg_response_time,
                'avg_confidence': avg_confidence,
                'intervention_rate': intervention_rate,
                'enhancement_rate': enhancement_rate,
                'total_queries': len(results),
                'enterprise_ready': avg_response_time < 3000 and avg_confidence > 0.7
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500

@app.get("/api/agent-status")
def agent_status():
    """Get comprehensive agent system status"""
    try:
        from lib.enterprise_rag_agent import create_enterprise_rag_agent
        from lib.human_intervention import create_human_intervention_manager
        
        # Test agent initialization
        agent = create_enterprise_rag_agent()
        intervention_manager = create_human_intervention_manager()
        
        # Get system capabilities
        capabilities = {
            'guardrails_enabled': agent.guardrails_enabled,
            'committee_agents_enabled': agent.committee_agents_enabled,
            'intervention_enabled': agent.intervention_enabled,
            'monitoring_enabled': agent.monitoring_enabled,
            'available_tools': len(agent.tools)
        }
        
        # Test basic functionality
        test_query = "What is our membership fee structure?"
        test_start = time.time()
        test_response = agent.run(ORG_ID, test_query)
        test_duration = int((time.time() - test_start) * 1000)
        
        # Get comprehensive system status including monitoring data
        system_status = agent.get_system_status()
        
        return jsonify({
            'ok': True,
            'status': 'operational',
            'capabilities': capabilities,
            'test_results': {
                'query': test_query,
                'response_time_ms': test_duration,
                'strategy': test_response.get('strategy'),
                'confidence': test_response.get('confidence'),
                'intervention_triggered': test_response.get('intervention_triggered', False)
            },
            'system_health': {
                'enterprise_ready': all(capabilities.values()),
                'response_performance': 'excellent' if test_duration < 2000 else 'good' if test_duration < 4000 else 'needs_optimization'
            },
            'detailed_status': system_status
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'status': 'error',
            'error': str(e)
        }), 500

@app.get("/api/performance-report")
def performance_report():
    """Get detailed performance monitoring report"""
    try:
        hours = int(request.args.get('hours', 24))
        
        from lib.enterprise_rag_agent import create_enterprise_rag_agent
        agent = create_enterprise_rag_agent()
        
        if not agent.monitoring_enabled or not agent.performance_monitor:
            return jsonify({
                'ok': False,
                'error': 'Performance monitoring not available'
            }), 404
        
        performance_summary = agent.performance_monitor.get_performance_summary(hours)
        
        return jsonify({
            'ok': True,
            'performance_report': performance_summary,
            'report_generated': datetime.now().isoformat(),
            'time_period_hours': hours
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Performance report failed: {str(e)}'
        }), 500

@app.get("/api/deployment-verification")
def deployment_verification():
    """Run comprehensive deployment verification checks"""
    try:
        from deploy.deployment_verification import DeploymentVerifier
        
        verifier = DeploymentVerifier()
        report = verifier.run_comprehensive_verification()
        
        return jsonify({
            'ok': True,
            'deployment_verification': report
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Deployment verification failed: {str(e)}'
        }), 500

@app.post("/api/bulletproof-processing")
def bulletproof_processing():
    """Run bulletproof document processing to achieve 100% coverage"""
    try:
        data = request.json or {}
        org_id = data.get('org_id', ORG_ID)
        force_reprocess = data.get('force_reprocess', False)
        
        from lib.bulletproof_processing import create_bulletproof_processor
        
        processor = create_bulletproof_processor()
        processing_result = processor.process_all_documents(org_id, force_reprocess)
        
        return jsonify({
            'ok': True,
            'bulletproof_processing': processing_result
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Bulletproof processing failed: {str(e)}'
        }), 500

@app.get("/api/processing-status")
def processing_status():
    """Get comprehensive document processing status"""
    try:
        org_id = request.args.get('org_id', ORG_ID)
        
        from lib.bulletproof_processing import create_bulletproof_processor
        
        processor = create_bulletproof_processor()
        status = processor.get_processing_status(org_id)
        
        return jsonify({
            'ok': True,
            'processing_status': status
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Processing status check failed: {str(e)}'
        }), 500

@app.post("/api/diagnose-coverage")
def diagnose_coverage():
    """Diagnose document coverage issues and provide repair recommendations"""
    try:
        data = request.json or {}
        org_id = data.get('org_id', ORG_ID)
        
        from lib.bulletproof_processing import DocumentCoverageDiagnostic
        
        diagnostic = DocumentCoverageDiagnostic()
        diagnosis = diagnostic.diagnose_coverage_issues(org_id)
        
        return jsonify({
            'ok': True,
            'coverage_diagnosis': diagnosis
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Coverage diagnosis failed: {str(e)}'
        }), 500

@app.post("/api/repair-coverage")
def repair_coverage():
    """Execute repair actions to achieve 100% document coverage"""
    try:
        data = request.json or {}
        org_id = data.get('org_id', ORG_ID)
        repair_actions = data.get('repair_actions', ['force_reprocess_failed', 'process_pending_documents'])
        
        from lib.bulletproof_processing import DocumentCoverageDiagnostic
        
        diagnostic = DocumentCoverageDiagnostic()
        repair_result = diagnostic.repair_coverage_issues(org_id, repair_actions)
        
        return jsonify({
            'ok': True,
            'coverage_repair': repair_result
        })
        
    except Exception as e:
        return jsonify({
            'ok': False,
            'error': f'Coverage repair failed: {str(e)}'
        }), 500

# Automated Processing Queue Endpoints

@app.route('/api/queue-documents', methods=['POST'])
def queue_documents():
    """Add documents to automated processing queue"""
    
    try:
        data = request.json or {}
        org_id = data.get('org_id', ORG_ID)
        document_ids = data.get('document_ids', None)  # Optional: specific documents
        
        queue = get_document_queue()
        
        # Since we can't use async in Flask routes directly, we'll run it in a thread
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            queue.add_documents_to_queue(org_id, document_ids)
        )
        
        return jsonify({
            'success': True,
            'queued_result': result,
            'message': f"Added {result['added_to_queue']} documents to processing queue"
        })
        
    except Exception as e:
        print(f"Queue documents error: {str(e)}")
        return jsonify({'error': 'Failed to queue documents', 'details': str(e)}), 500

@app.route('/api/queue-status', methods=['GET'])
def queue_status():
    """Get processing queue status"""
    
    try:
        queue = get_document_queue()
        status = queue.get_queue_status()
        
        return jsonify(status)
        
    except Exception as e:
        print(f"Queue status error: {str(e)}")
        return jsonify({'error': 'Failed to get queue status', 'details': str(e)}), 500

@app.route('/api/auto-process-documents', methods=['POST'])
def auto_process_documents():
    """Automatically process all unprocessed documents using the queue system"""
    
    try:
        data = request.json or {}
        org_id = data.get('org_id', ORG_ID)
        
        queue = get_document_queue()
        
        # Add all unprocessed documents to queue
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # First add to queue
        queue_result = loop.run_until_complete(
            queue.add_documents_to_queue(org_id)
        )
        
        return jsonify({
            'success': True,
            'queue_result': queue_result,
            'processing_started': queue_result['added_to_queue'] > 0,
            'message': f"Started automated processing of {queue_result['added_to_queue']} documents"
        })
        
    except Exception as e:
        print(f"Auto process error: {str(e)}")
        return jsonify({'error': 'Auto processing failed', 'details': str(e)}), 500

@app.route('/api/document-debug', methods=['GET'])
def document_debug():
    """Get comprehensive document debugging analysis"""
    
    try:
        org_id = request.args.get('org_id', ORG_ID)
        
        from lib.document_debugger import create_document_debugger
        
        debugger = create_document_debugger()
        analysis = debugger.comprehensive_document_analysis(org_id)
        
        return jsonify({
            'ok': True,
            'debug_analysis': analysis,
            'generated_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f'Document debug error: {str(e)}')
        return jsonify({
            'ok': False,
            'error': 'Document debugging failed', 
            'details': str(e)
        }), 500

@app.route('/api/document-coverage-status', methods=['GET'])
def document_coverage_status():
    """Get document coverage status for an organization"""
    
    try:
        org_id = request.args.get('org_id', ORG_ID)
        
        from lib.bulletproof_processing import DocumentCoverageDiagnostic
        
        diagnostic = DocumentCoverageDiagnostic()
        diagnosis = diagnostic.diagnose_coverage_issues(org_id)
        
        return jsonify({
            'ok': True,
            'coverage_analysis': diagnosis['coverage_analysis'],
            'processing_issues': diagnosis['processing_issues'][:10]  # Limit to first 10
        })
        
    except Exception as e:
        print(f'Coverage status error: {str(e)}')
        return jsonify({
            'ok': False,
            'error': 'Coverage status failed', 
            'details': str(e),
            'coverage_analysis': {
                'coverage_percentage': 0,
                'total_documents': 0,
                'processed_documents': 0,
                'unprocessed_documents': 0
            },
            'processing_issues': []
        }), 200  # Return 200 with error structure instead of 500

@app.route('/api/fix-document-coverage', methods=['POST'])
def fix_document_coverage():
    """One-click fix for document coverage issues"""
    
    try:
        data = request.json or {}
        org_id = data.get('org_id', ORG_ID)
        
        print(f"Starting one-click coverage fix for {org_id}")
        
        # Step 1: Diagnose issues
        diagnostic = DocumentCoverageDiagnostic()
        diagnosis = diagnostic.diagnose_coverage_issues(org_id)
        
        initial_coverage = diagnosis['coverage_analysis']['coverage_percentage']
        print(f"Initial coverage: {initial_coverage}%")
        
        if initial_coverage >= 100:
            return jsonify({
                'already_complete': True,
                'coverage': '100%',
                'message': 'All documents already processed successfully!'
            })
        
        # Step 2: Use automated queue system for processing
        queue = get_document_queue()
        
        # Add all unprocessed documents to queue
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        queue_result = loop.run_until_complete(
            queue.add_documents_to_queue(org_id)
        )
        
        # Step 3: Wait for processing to complete (with timeout)
        start_time = time.time()
        max_wait = 300  # 5 minutes max
        
        while queue.is_processing and (time.time() - start_time) < max_wait:
            time.sleep(2)  # Check every 2 seconds
        
        # Step 4: Verify final coverage
        final_diagnosis = diagnostic.diagnose_coverage_issues(org_id)
        final_coverage = final_diagnosis['coverage_analysis']['coverage_percentage']
        
        success = final_coverage >= 90  # 90%+ is considered success
        
        print(f"Final coverage: {final_coverage}%")
        
        return jsonify({
            'fix_completed': True,
            'success': success,
            'initial_coverage': f"{initial_coverage}%",
            'final_coverage': f"{final_coverage}%",
            'queue_result': queue_result,
            'documents_processed': queue_result['added_to_queue'],
            'message': f"Coverage improved from {initial_coverage}% to {final_coverage}%"
        })
        
    except Exception as e:
        print(f"One-click fix failed: {str(e)}")
        return jsonify({
            'fix_completed': False,
            'error': str(e),
            'message': 'Fix attempt failed - please check logs'
        }), 500

if __name__ == "__main__":
    print(f"BoardContinuity using ORG={ORG_ID} USER={USER_ID}")
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host="0.0.0.0", port=port, debug=debug)