import os
from flask import Flask, render_template, request, jsonify
from lib.ingest import upsert_document
from lib.rag import answer_question

app = Flask(__name__)

# Dev identity for now (seeded already)
ORG_ID = os.getenv("DEV_ORG_ID")
USER_ID = os.getenv("DEV_USER_ID")

@app.get("/")
def home():
    return render_template("home.html")

@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "no file"}), 400
    b = f.read()
    doc, n = upsert_document(ORG_ID, USER_ID, f.filename, b, f.mimetype)
    return jsonify({"ok": True, "document_id": doc["id"], "chunks": n})

@app.post("/chat")
def chat():
    q = request.form.get("q") or request.json.get("q")
    if not q:
        return jsonify({"ok": False, "error": "missing q"}), 400
    ans = answer_question(ORG_ID, q)
    return jsonify({"ok": True, "answer": ans})

if __name__ == "__main__":
    print(f"BoardContinuity using ORG={ORG_ID} USER={USER_ID}")
    app.run(host="0.0.0.0", port=8000, debug=True)