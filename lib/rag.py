from openai import OpenAI
from lib.supa import supa, signed_url_for
import os

client = OpenAI()

SYSTEM_PROMPT = (
    "You are Forever Board Member. Answer ONLY from the provided excerpts. "
    "Every claim must include an inline citation like [Doc:{document_id}#Chunk:{chunk_index}]. "
    "Prefer table rows when present. If insufficient, say so and ask for more sources."
)

def _vector_search(org_id: str, query: str, k: int = 40):
    emb_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    emb = client.embeddings.create(model=emb_model, input=query).data[0].embedding
    try:
        rows = supa.rpc("match_chunks", {
            "query_embedding": emb,
            "match_count": k,
            "org": str(org_id)
        }).execute().data or []
    except Exception:
        rows = []
    return rows

def _keyword_fallback(org_id: str, q: str, limit: int = 40):
    terms = [q, "reserved", "Open Times", "Primary Golfers", "Ladies", "Juniors", "dues", "assessment", "bylaws"]
    seen = set(); out = []
    for t in terms:
        resp = supa.table("doc_chunks").select("document_id,chunk_index,content,summary").eq("org_id", org_id).ilike("content", f"%{t}%").limit(limit).execute().data
        for r in resp or []:
            key = (r["document_id"], r["chunk_index"])
            if key in seen: continue
            seen.add(key); out.append(r)
    return out

def _doc_title_and_link(doc_id: str):
    d = supa.table("documents").select("title,storage_path").eq("id", doc_id).limit(1).execute().data
    if not d: return (None, None)
    title = d[0].get("title") or "Document"
    link = signed_url_for(d[0].get("storage_path") or "")
    return (title, link)

def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for GPT models"""
    return len(text) // 4

def answer_question_md(org_id: str, question: str, chat_model: str = "gpt-4o"):
    # Start with fewer chunks to avoid token limits
    rows = _vector_search(org_id, question, k=20)
    if len(rows) < 3:
        rows += _keyword_fallback(org_id, question, limit=20)

    if not rows:
        return ("Insufficient sources found. Please upload meeting minutes, bylaws, or project files.", [])

    excerpts = []
    cite_meta = []
    seen_pairs = set()
    total_tokens = len(SYSTEM_PROMPT) // 4 + len(question) // 4 + 200  # Base overhead
    MAX_TOKENS = 25000  # Leave buffer for response tokens
    MAX_CHUNK_LENGTH = 1500  # Limit individual chunk size

    # Process chunks with token limit awareness - prefer summaries when available
    for r in rows[:40]:  # Reduced from 80
        doc_id = r.get('document_id'); chunk_idx = r.get('chunk_index')
        content = r.get('content'); summary = r.get('summary')
        if not (doc_id and (content or summary)): continue
        key = (doc_id, chunk_idx)
        if key in seen_pairs: continue
        seen_pairs.add(key)

        title, link = _doc_title_and_link(doc_id)
        cite_meta.append({"document_id": doc_id, "chunk_index": chunk_idx, "title": title, "url": link})
        cid = f"[Doc:{doc_id}#Chunk:{chunk_idx}]"
        
        # Use pre-computed summary if available (much more token-efficient)
        if summary and len(summary.strip()) > 20:
            text_content = summary
        else:
            # Fallback to truncated content
            text_content = content[:MAX_CHUNK_LENGTH] + "..." if len(content) > MAX_CHUNK_LENGTH else content
        
        excerpt = f"{cid} {text_content}"
        
        # Check if adding this excerpt would exceed token limit
        excerpt_tokens = _estimate_tokens(excerpt)
        if total_tokens + excerpt_tokens > MAX_TOKENS:
            break
        
        excerpts.append(excerpt)
        total_tokens += excerpt_tokens

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nEXCERPTS:\n" + "\n".join(excerpts)}
    ]
    
    print(f"RAG: Using {len(excerpts)} excerpts, estimated {total_tokens} tokens")
    resp = client.chat.completions.create(model=chat_model, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content

    # Return JUST the markdown answer; frontend will build a deduped Sources section from cite_meta
    return (answer, cite_meta)