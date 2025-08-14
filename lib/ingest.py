import hashlib, io
from typing import Tuple
from pypdf import PdfReader
from pdfminer.high_level import extract_text
from openai import OpenAI
from lib.supa import supa, SUPABASE_BUCKET
import tiktoken

client = OpenAI()

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

# --- PDF text extraction ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    # Try PyPDF first
    try:
        r = PdfReader(io.BytesIO(file_bytes))
        out = []
        for p in r.pages:
            out.append(p.extract_text() or "")
        text = "\n".join(out)
        if len(text.strip()) > 50:
            return text
    except Exception:
        pass
    # Fallback to pdfminer
    try:
        return extract_text(io.BytesIO(file_bytes)) or ""
    except Exception:
        return ""

# --- Chunking ---

def smart_chunks(text: str, target_tokens: int = 900, overlap: int = 150):
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    if not toks:
        return []
    step = max(1, target_tokens - overlap)
    chunks = []
    for i in range(0, len(toks), step):
        seg = toks[i:i + target_tokens]
        chunks.append(enc.decode(seg))
    return chunks

# --- Embeddings ---

def embed_texts(texts: list[str]) -> list[list[float]]:
    from os import getenv
    model = getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

# --- Ingest pipeline ---

def upsert_document(org_id: str, user_id: str, filename: str, file_bytes: bytes, mime_type: str) -> Tuple[dict, int]:
    sha = _sha256_bytes(file_bytes)
    storage_path = f"{org_id}/{sha}/{filename}"

    # Upload to Storage (idempotent)
    supa.storage.from_(SUPABASE_BUCKET).upload(storage_path, file_bytes, {
        "content-type": mime_type,
        "x-upsert": "true"
    })

    # Create/insert document row (compat fields included)
    doc = supa.table("documents").insert({
        "org_id": org_id,
        "created_by": user_id,
        "title": filename,
        "name": filename,
        "filename": filename,
        "storage_path": storage_path,
        "file_path": storage_path,
        "sha256": sha,
        "mime_type": mime_type,
        "size_bytes": len(file_bytes),
        "status": "processing",
        "processed": False,
        "processing_error": None
    }).execute().data[0]

    # Extract text
    text = extract_text_from_pdf(file_bytes)
    if len(text.strip()) < 50:
        # Likely a scan; mark error but keep row
        supa.table("documents").update({
            "status": "error",
            "processed": False,
            "processing_error": "Scanned PDF not supported in MVP"
        }).eq("id", doc["id"]).execute()
        return doc, 0

    # Chunk + embed
    chunks = smart_chunks(text, target_tokens=900, overlap=150)
    embeddings = embed_texts(chunks)

    rows = []
    for i, (c, e) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "org_id": org_id,
            "document_id": doc["id"],
            "chunk_index": i,
            "content": c,
            "token_count": len(c),
            "embedding": e
        })

    if rows:
        supa.table("doc_chunks").insert(rows).execute()
        supa.table("documents").update({"status": "ready", "processed": True}).eq("id", doc["id"]).execute()
    else:
        supa.table("documents").update({"status": "error", "processed": False, "processing_error": "no chunks"}).eq("id", doc["id"]).execute()

    return doc, len(rows)