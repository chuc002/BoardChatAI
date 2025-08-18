import os, json
from flask import Flask, render_template, request, jsonify
from lib.ingest import upsert_document
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
        md, citations = answer_question_md(ORG_ID, q)
        # persist history
        supa.table("qa_history").insert({
            "org_id": ORG_ID,
            "user_id": USER_ID,
            "question": q,
            "answer_md": md,
            "citations": json.loads(json.dumps(citations))  # ensure pure json
        }).execute()
        return jsonify({"ok": True, "markdown": md, "citations": citations})
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

if __name__ == "__main__":
    print(f"BoardContinuity using ORG={ORG_ID} USER={USER_ID}")
    app.run(host="0.0.0.0", port=8000, debug=True)