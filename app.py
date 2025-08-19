import os, json
from flask import Flask, render_template, request, jsonify
from lib.ingest import upsert_document
from lib.enhanced_ingest import enhanced_upsert_document, validate_reinstatement_coverage
from lib.institutional_memory import process_document_for_institutional_memory, get_institutional_insights
from lib.perfect_extraction import extract_perfect_information, validate_extraction_quality
from lib.pattern_recognition import analyze_governance_patterns, predict_proposal_outcome
from lib.knowledge_graph import (build_knowledge_graph, analyze_decision_ripple_effects, 
                                get_member_complete_analysis, trace_policy_evolution,
                                find_governance_cycles, query_knowledge_graph)
from lib.governance_intelligence import (analyze_decision_comprehensive, predict_decision_outcome,
                                       get_decision_context, analyze_governance_trends, generate_board_insights)
from lib.memory_synthesis import (recall_topic_history, answer_with_veteran_wisdom,
                                get_institutional_wisdom, explain_club_culture)
from lib.rag import answer_question_md
from lib.supa import supa, signed_url_for, SUPABASE_BUCKET

app = Flask(__name__)

ORG_ID = os.getenv("DEV_ORG_ID")
USER_ID = os.getenv("DEV_USER_ID")

@app.get("/")
def home():
    return render_template("home.html")

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

if __name__ == "__main__":
    print(f"BoardContinuity using ORG={ORG_ID} USER={USER_ID}")
    app.run(host="0.0.0.0", port=8000, debug=True)