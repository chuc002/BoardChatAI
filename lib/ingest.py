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

# --- PDF text extraction WITH PAGES ---
def extract_pages_from_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """
    Returns a list of (page_index, text) using PyPDF; falls back to pdfminer (as one page).
    """
    pages: list[tuple[int, str]] = []
    # Try PyPDF first (page-accurate)
    try:
        r = PdfReader(io.BytesIO(file_bytes))
        for i, p in enumerate(r.pages):
            pages.append((i, (p.extract_text() or "")))
        if any(t.strip() for _, t in pages):
            return pages
    except Exception:
        pages = []

    # Fallback: pdfminer (whole doc as page 0)
    try:
        whole = extract_text(io.BytesIO(file_bytes)) or ""
        if whole.strip():
            return [(0, whole)]
    except Exception:
        pass
    return []

# --- Chunking WITH page tracking ---
def smart_chunks_by_page(pages: list[tuple[int, str]], target_tokens: int = 900, overlap: int = 150):
    enc = tiktoken.get_encoding("cl100k_base")
    out: list[tuple[int, str]] = []
    for page_idx, page_text in pages:
        toks = enc.encode(page_text or "")
        if not toks:
            continue
        step = max(1, target_tokens - overlap)
        for i in range(0, len(toks), step):
            seg = toks[i:i + target_tokens]
            out.append((page_idx, enc.decode(seg)))
    return out

# --- Embeddings ---
def embed_texts(texts: list[str]) -> list[list[float]]:
    from os import getenv
    model = getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim, high quality
    try:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        print(f"[EMBED] Failed with model '{model}': {e}")
        # Try fallback model
        if model != "text-embedding-ada-002":
            print("[EMBED] Trying fallback: text-embedding-ada-002")
            resp = client.embeddings.create(model="text-embedding-ada-002", input=texts)
            return [d.embedding for d in resp.data]
        raise

# --- Ingest pipeline (with pre-summaries) ---
def _summarize_chunk(text: str, doc_id: str, idx: int) -> str:
    """
    Compress a chunk to a compact, factual note ending with its citation.
    We cap length so it stays ~60–100 tokens.
    """
    import os
    from openai import OpenAI
    client = OpenAI()
    prompt = (
        "Condense the following board-document excerpt into 1–3 short bullet sentences, "
        "keeping dates, amounts, named entities, rules, and obligations. "
        "No preamble. End with the citation token exactly once.\n\n"
        f"EXCERPT [{doc_id}#{idx}]:\n{text[:1500]}\n\n"
        f"Finish with [Doc:{doc_id}#Chunk:{idx}]"
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("CHAT_COMPRESS","gpt-4o-mini"),
            temperature=0.0,
            messages=[{"role":"user","content":prompt}],
            max_tokens=140  # ~100 tokens output + safety
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # fall back to truncation if API hiccups
        return (text[:500] + f"... [Doc:{doc_id}#Chunk:{idx}]")

def upsert_document(org_id: str, user_id: str, filename: str, file_bytes: bytes, mime_type: str) -> Tuple[dict, int]:
    sha = _sha256_bytes(file_bytes)
    storage_path = f"{org_id}/{sha}/{filename}"

    # De-dupe by sha256
    existing = supa.table("documents").select("*").eq("org_id", org_id).eq("sha256", sha).limit(1).execute().data
    if existing:
        supa.storage.from_(SUPABASE_BUCKET).upload(storage_path, file_bytes, {
            "content-type": mime_type,
            "x-upsert": "true"
        })
        return existing[0], 0

    # Upload binary
    supa.storage.from_(SUPABASE_BUCKET).upload(storage_path, file_bytes, {
        "content-type": mime_type,
        "x-upsert": "true"
    })

    # Insert document row
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

    # Extract text (paged)
    pages = extract_pages_from_pdf(file_bytes)
    joined = "\n".join(t for _, t in pages)
    if len(joined.strip()) < 50:
        supa.table("documents").update({
            "status": "error",
            "processed": False,
            "processing_error": "Scanned PDF not supported in MVP"
        }).eq("id", doc["id"]).execute()
        return doc, 0

    # Chunk + embed (page-aware)
    chunk_tuples = smart_chunks_by_page(pages, target_tokens=900, overlap=150)  # [(page_idx, text)]
    texts = [c for _, c in chunk_tuples]
    embeddings = embed_texts(texts)

    # Pre-summarize each chunk (you already have _summarize_chunk)
    rows = []
    for i, ((page_idx, c), e) in enumerate(zip(chunk_tuples, embeddings)):
        s = _summarize_chunk(c, doc["id"], i)
        rows.append({
            "org_id": org_id,
            "document_id": doc["id"],
            "chunk_index": i,
            "page_index": page_idx,          # <-- store page index
            "content": c,
            "summary": s,
            "token_count": len(c),
            "embedding": e
        })

    if rows:
        supa.table("doc_chunks").insert(rows).execute()
        supa.table("documents").update({"status": "ready", "processed": True}).eq("id", doc["id"]).execute()
    else:
        supa.table("documents").update({"status": "error", "processed": False, "processing_error": "no chunks"}).eq("id", doc["id"]).execute()

    return doc, len(rows)