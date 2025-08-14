from openai import OpenAI
from lib.supa import supa, signed_url_for
import os, time
import tiktoken

client = OpenAI()

# ==== CONFIG (env overrideable) ====
CHAT_PRIMARY   = os.getenv("CHAT_PRIMARY", "gpt-4o")         # final answer
CHAT_COMPRESS  = os.getenv("CHAT_COMPRESS", "gpt-4o-mini")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")

MAX_CANDIDATES         = int(os.getenv("MAX_CANDIDATES", "24"))   # retrieved rows (summary-based)
MAX_SUMMARY_TOKENS     = int(os.getenv("MAX_SUMMARY_TOKENS", "2400")) # notes budget
MAX_FINAL_TOKENS       = int(os.getenv("MAX_FINAL_TOKENS", "4800"))   # user + notes into final
TEMPERATURE            = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
# ===================================

SYSTEM_PROMPT = (
    "You are Forever Board Member. Answer ONLY from the provided source notes. "
    "Every claim must include an inline citation like [Doc:{document_id}#Chunk:{chunk_index}]. "
    "If the notes are insufficient, say so and ask for the missing document."
)

enc = tiktoken.get_encoding("cl100k_base")
def _toks(s: str) -> int:
    try: return len(enc.encode(s or ""))
    except Exception: return len((s or "").split())

def _retry(fn, tries=4, base=0.6):
    last=None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last=e; time.sleep(base*(2**i))
    raise last

def _vector(org_id: str, q: str, k: int):
    emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
    try:
        rows = supa.rpc("match_chunks", {
            "query_embedding": emb,
            "match_count": k,
            "org": str(org_id)
        }).execute().data or []
    except Exception:
        rows=[]
    return rows

def _keyword(org_id: str, q: str, k: int):
    terms=[q,"bylaws","policy","rules","minutes","assessment","dues","board","committee","vote","reserved","Open Times","Primary","Juniors","Ladies"]
    seen,out=set(),[]
    for t in terms:
        resp = supa.table("doc_chunks").select("document_id,chunk_index,summary,content").eq("org_id", org_id).ilike("content", f"%{t}%").limit(k).execute().data
        for r in resp or []:
            key=(r["document_id"], r["chunk_index"])
            if key in seen: continue
            seen.add(key); out.append(r)
            if len(out)>=k: break
        if len(out)>=k: break
    return out

def _doc_title_link(doc_id: str):
    d = supa.table("documents").select("title,storage_path").eq("id", doc_id).limit(1).execute().data
    if not d: return ("Document", None)
    return (d[0].get("title") or "Document", signed_url_for(d[0].get("storage_path") or ""))

def answer_question_md(org_id: str, question: str, chat_model: str | None = None):
    # 1) retrieve
    rows = _vector(org_id, question, k=MAX_CANDIDATES)
    if len(rows) < 8:
        rows += _keyword(org_id, question, k=MAX_CANDIDATES)
    # dedupe
    seen=set(); dedup=[]
    for r in rows:
        key=(r.get("document_id"), r.get("chunk_index"))
        if key in seen: continue
        seen.add(key); dedup.append(r)
    rows = dedup[:MAX_CANDIDATES]

    if not rows:
        return ("Insufficient sources found. Upload minutes, bylaws, policies.", [])

    # 2) build notes from pre-summaries, fallback to raw (trimmed)
    notes=[]; total=0; meta=[]
    for r in rows:
        doc_id=r.get("document_id"); ci=r.get("chunk_index")
        title, link = _doc_title_link(doc_id)
        meta.append({"document_id": doc_id, "chunk_index": ci, "title": title, "url": link})

        s = r.get("summary")
        if not s or len(s)<20:
            content = r.get("content") or ""
            content = (content[:1200] + f" [Doc:{doc_id}#Chunk:{ci}]")
            s = content
        # keep budget
        t=_toks(s)
        if total + t > MAX_SUMMARY_TOKENS: break
        notes.append("- " + s.strip())
        total += t

    if not notes:
        return ("Could not build source notes within limits. Refine the question.", meta)

    # 3) final synthesis with strict budget
    preamble = f"QUESTION: {question}\n\nSOURCE NOTES (each ends with its citation):\n"
    body = "\n".join(notes)
    prompt = preamble + body
    while _toks(prompt) > MAX_FINAL_TOKENS and len(notes) > 4:
        notes.pop()
        body = "\n".join(notes)
        prompt = preamble + body

    model = chat_model or CHAT_PRIMARY
    def run():
        resp = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            max_tokens=700
        )
        return resp.choices[0].message.content
    try:
        answer = _retry(run)
    except Exception:
        # Downshift to mini on any 429/limit/timeouts
        model = "gpt-4o-mini"
        answer = _retry(lambda: client.chat.completions.create(
            model=model, temperature=TEMPERATURE,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            max_tokens=600
        ).choices[0].message.content)

    return (answer, meta)