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

# --- PDF text extraction with page tracking ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text without page tracking (legacy compatibility)"""
    pages_with_text = extract_pages_from_pdf(file_bytes)
    return "\n".join([text for text, _ in pages_with_text])

def extract_pages_from_pdf(file_bytes: bytes) -> list[tuple[str, int]]:
    """Extract text with page numbers: returns [(text, page_index), ...]"""
    # Try PyPDF first
    try:
        r = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page_idx, p in enumerate(r.pages):
            text = p.extract_text() or ""
            if text.strip():  # Only include pages with text
                pages.append((text, page_idx + 1))  # 1-based page numbers
        if len(pages) > 0:
            return pages
    except Exception:
        pass
    
    # Fallback to pdfminer (no page separation available)
    try:
        full_text = extract_text(io.BytesIO(file_bytes)) or ""
        if full_text.strip():
            return [(full_text, 1)]  # Assume single page for fallback
    except Exception:
        pass
    
    return []

# --- Chunking ---
def smart_chunks(text: str, target_tokens: int = 900, overlap: int = 150):
    """Legacy chunking without page tracking"""
    return smart_chunks_with_pages([(text, 1)], target_tokens, overlap)

def smart_chunks_with_pages(pages: list[tuple[str, int]], target_tokens: int = 900, overlap: int = 150) -> list[tuple[str, int]]:
    """
    Chunk text while preserving page numbers.
    Returns [(chunk_text, page_index), ...] where page_index is from the first token of the chunk.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    all_tokens = []
    token_to_page = []
    
    # Build complete token sequence with page mapping
    for text, page_idx in pages:
        toks = enc.encode(text)
        all_tokens.extend(toks)
        token_to_page.extend([page_idx] * len(toks))
    
    if not all_tokens:
        return []
    
    step = max(1, target_tokens - overlap)
    chunks_with_pages = []
    
    for i in range(0, len(all_tokens), step):
        seg = all_tokens[i:i + target_tokens]
        chunk_text = enc.decode(seg)
        # Page index from first token in chunk
        page_idx = token_to_page[i] if i < len(token_to_page) else 1
        chunks_with_pages.append((chunk_text, page_idx))
    
    return chunks_with_pages

# --- Embeddings ---
def embed_texts(texts: list[str]) -> list[list[float]]:
    from os import getenv
    model = getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536-dim
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

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

    # Extract text with page tracking
    pages = extract_pages_from_pdf(file_bytes)
    if not pages or sum(len(text.strip()) for text, _ in pages) < 50:
        supa.table("documents").update({
            "status": "error",
            "processed": False,
            "processing_error": "Scanned PDF not supported in MVP"
        }).eq("id", doc["id"]).execute()
        return doc, 0

    # Chunk with page preservation + embed
    chunks_with_pages = smart_chunks_with_pages(pages, target_tokens=900, overlap=150)
    chunk_texts = [chunk for chunk, _ in chunks_with_pages]
    embeddings = embed_texts(chunk_texts)

    # Pre-summarize each chunk (budgeted; fast mini model)
    rows = []
    for i, ((chunk_text, page_idx), e) in enumerate(zip(chunks_with_pages, embeddings)):
        s = _summarize_chunk(chunk_text, doc["id"], i)
        rows.append({
            "org_id": org_id,
            "document_id": doc["id"],
            "chunk_index": i,
            "content": chunk_text,
            "summary": s,             # <= we store the pre-summary
            "token_count": len(chunk_text),
            "embedding": e,
            "page_index": page_idx    # <= NEW: page number for deep-linking
        })

    if rows:
        supa.table("doc_chunks").insert(rows).execute()
        supa.table("documents").update({"status": "ready", "processed": True}).eq("id", doc["id"]).execute()
    else:
        supa.table("documents").update({"status": "error", "processed": False, "processing_error": "no chunks"}).eq("id", doc["id"]).execute()

    return doc, len(rows)