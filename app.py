import os
from flask import Flask, render_template, request, jsonify
from lib.ingest import upsert_document
from lib.rag import answer_question_md
from lib.supa import supa, signed_url_for, SUPABASE_BUCKET

app = Flask(__name__)

# Dev identity for now (seeded already)
ORG_ID = os.getenv("DEV_ORG_ID", "")
USER_ID = os.getenv("DEV_USER_ID", "")

@app.get("/")
def home():
    return render_template("home.html")

@app.get("/docs")
def docs():
    rows = supa.table("documents").select("id,title,filename,storage_path,status,processed,created_at").eq("org_id", ORG_ID).order("created_at", desc=True).limit(200).execute().data or []
    for r in rows:
        if r.get("storage_path"):
            r["download"] = signed_url_for(r["storage_path"], expires_in=3600)
    return jsonify({"ok": True, "docs": rows})

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    b = f.read()
    doc, n = upsert_document(ORG_ID, USER_ID, f.filename or "unknown.pdf", b, f.mimetype or "application/pdf")
    return jsonify({"ok": True, "document_id": doc["id"], "chunks": n})

@app.post("/chat")
def chat():
    q = (request.form.get("q") if request.form else None) or (request.json.get("q") if request.is_json else None)
    if not q:
        return jsonify({"ok": False, "error": "missing q"}), 400
    md, cites = answer_question_md(ORG_ID, q)
    return jsonify({"ok": True, "markdown": md, "citations": cites})

@app.get("/snippet")
def snippet():
    doc_id = request.args.get("doc")
    chunk = request.args.get("chunk")
    if not doc_id:
        return jsonify({"ok": False, "error": "missing doc"}), 400
    sel = supa.table("doc_chunks").select("content,chunk_index,document_id").eq("document_id", doc_id)
    if chunk is not None:
        sel = sel.eq("chunk_index", int(chunk))
    row = sel.limit(1).execute().data
    if not row:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "snippet": row[0]["content"]})

if __name__ == "__main__":
    print(f"BoardContinuity using ORG={ORG_ID} USER={USER_ID}")
    app.run(host="0.0.0.0", port=8000, debug=True)