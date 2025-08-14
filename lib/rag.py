from openai import OpenAI
from lib.supa import supa, signed_url_for
import os
from typing import List, Tuple, Dict, Any

client = OpenAI()

SYSTEM_PROMPT = (
    "You are Forever Board Member. Answer ONLY from the provided excerpts. "
    "Every claim must include an inline citation like [Doc:{document_id}#Chunk:{chunk_index}]. "
    "Prefer table rows when present. If insufficient, say so and ask for more sources."
)

# Vector search via RPC + hybrid keyword fallback

def _vector_search(org_id: str, query: str, k: int = 15):  # Reduced from 40 to 15
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


def _keyword_fallback(org_id: str, q: str, limit: int = 10):  # Reduced from 40 to 10
    terms = [q, "reserved", "Open Times", "Primary Golfers", "Ladies", "Juniors", "dues", "assessment", "bylaws"]
    seen = set(); out = []
    for t in terms:
        resp = supa.table("doc_chunks").select("document_id,chunk_index,content").eq("org_id", org_id).ilike("content", f"%{t}%").limit(limit).execute().data
        for r in resp or []:
            key = (r["document_id"], r["chunk_index"])
            if key in seen: continue
            seen.add(key); out.append(r)
    return out


def _doc_title_and_link(doc_id: str) -> Tuple[str | None, str | None]:
    d = supa.table("documents").select("title,storage_path").eq("id", doc_id).limit(1).execute().data
    if not d: return (None, None)
    title = d[0].get("title") or "Document"
    link = signed_url_for(d[0].get("storage_path") or "")
    return (title, link)


def answer_question_md(org_id: str, question: str, chat_model: str = "gpt-4o") -> Tuple[str, List[Dict[str, Any]]]:
    rows = _vector_search(org_id, question, k=15)  # Reduced chunks
    if len(rows) < 3:
        fallback_rows = _keyword_fallback(org_id, question, limit=10)
        rows = rows + fallback_rows

    if not rows:
        return ("Insufficient sources found. Please upload meeting minutes, bylaws, or project files.", [])

    # Limit to maximum 15 chunks to avoid token limit
    rows = rows[:15]
    
    excerpts = []
    cite_meta = []
    for r in rows:
        doc_id = r.get('document_id'); chunk_idx = r.get('chunk_index'); content = r.get('content')
        if not (doc_id and content is not None):
            continue
        title, link = _doc_title_and_link(doc_id)
        cite_meta.append({"document_id": doc_id, "chunk_index": chunk_idx, "title": title, "url": link})
        cid = f"[Doc:{doc_id}#Chunk:{chunk_idx}]"
        # Truncate very long chunks to save tokens
        truncated_content = content[:800] + "..." if len(content) > 800 else content
        excerpts.append(f"{cid} {truncated_content}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nEXCERPTS:\n" + "\n".join(excerpts)}
    ]
    resp = client.chat.completions.create(model=chat_model, messages=messages, temperature=0.2)
    answer = resp.choices[0].message.content or "No response generated."

    # Build nice markdown footer of sources
    lines = [answer, "\n\n---\n**Sources**:" ]
    seen = set()
    for c in cite_meta[:10]:
        key = (c["document_id"], c.get("chunk_index"))
        if key in seen: continue
        seen.add(key)
        label = c.get("title") or c["document_id"]
        if c.get("url"):
            lines.append(f"- {label} (chunk {c.get('chunk_index')}) — [open]({c['url']})")
        else:
            lines.append(f"- {label} (chunk {c.get('chunk_index')})")
    md = "\n".join(lines)
    return (md, cite_meta)