from openai import OpenAI
from lib.supa import supa
import os

client = OpenAI()

SYSTEM_PROMPT = (
    "You are Forever Board Member. Answer ONLY from the provided excerpts. "
    "Every claim must include an inline citation like [Doc:{document_id}#Chunk:{chunk_index}]. "
    "Prefer table rows when present. If insufficient, say so and ask for more sources."
)

# Vector search via RPC + hybrid keyword fallback

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
    # Simple ILIKE search for tables/terms
    terms = [q, "reserved", "Open Times", "Primary Golfers", "Ladies", "Juniors", "dues", "assessment", "bylaws"]
    seen = set()
    out = []
    for t in terms:
        resp = supa.table("doc_chunks").select("document_id,chunk_index,content").eq("org_id", org_id).ilike("content", f"%{t}%").limit(limit).execute().data
        for r in resp or []:
            key = (r["document_id"], r["chunk_index"])
            if key in seen: continue
            seen.add(key); out.append(r)
    return out


def answer_question(org_id: str, question: str, chat_model: str = "gpt-4o") -> str:
    rows = _vector_search(org_id, question, k=40)
    if len(rows) < 3:
        rows = rows + _keyword_fallback(org_id, question, limit=40)

    if not rows:
        return "Insufficient sources found. Please upload meeting minutes, bylaws, or project files."

    excerpts = []
    for r in rows[:40]:
        # Support both RPC row shape and table row shape
        doc_id = r.get('document_id')
        chunk_idx = r.get('chunk_index')
        content = r.get('content')
        if not (doc_id and content is not None):
            continue
        cid = f"[Doc:{doc_id}#Chunk:{chunk_idx}]"
        excerpts.append(f"{cid} {content}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nEXCERPTS:\n" + "\n".join(excerpts)}
    ]

    resp = client.chat.completions.create(model=chat_model, messages=messages, temperature=0.2)
    return resp.choices[0].message.content