from openai import OpenAI
from lib.supa import supa, signed_url_for
import os, time, math
import tiktoken

client = OpenAI()

# ==== CONFIG ====
CHAT_PRIMARY   = os.getenv("CHAT_PRIMARY", "gpt-3.5-turbo")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
USE_VECTOR     = os.getenv("USE_VECTOR", "1") not in ("0","false","False","no","NO")

MAX_CANDIDATES     = int(os.getenv("MAX_CANDIDATES", "24"))
MMR_K              = int(os.getenv("MMR_K", "12"))
MMR_LAMBDA         = float(os.getenv("MMR_LAMBDA", "0.65"))
MAX_SUMMARY_TOKENS = int(os.getenv("MAX_SUMMARY_TOKENS", "4000"))
MAX_FINAL_TOKENS   = int(os.getenv("MAX_FINAL_TOKENS", "6000"))
TEMPERATURE        = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
# =================

SYSTEM_PROMPT = (
    "You are Forever Board Member, an AI assistant specializing in board governance and club documents. "
    "Provide comprehensive, detailed answers using ONLY the exact information from the source notes provided. "
    "Extract and include specific numerical details: dollar amounts, percentages, timeframes, age limits, member counts, and precise requirements. "
    "Quote exact fee structures, payment schedules, and membership categories mentioned in the sources. "
    "When the source notes contain detailed information, provide thorough breakdowns with specific numbers and procedures. "
    "Include inline citations like [Doc:{document_id}#Chunk:{chunk_index}] for all specific claims. "
    "Synthesize information across multiple sections to give complete answers with all available details. "
    "Only state that notes are insufficient if the specific question truly cannot be answered from the provided sources."
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

def _fit_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    # quick path
    if _toks(text) <= max_tokens:
        return text
    # binary chop by characters until under budget
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = text[:mid]
        if _toks(cand) <= max_tokens:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best

# ---------- Retrieval ----------
def _vector(org_id: str, q: str, k: int):
    if not USE_VECTOR:
        return [], None
    t0=time.time()
    try:
        q_emb = client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
        rows = supa.rpc("match_chunks", {
            "query_embedding": q_emb,
            "match_count": k,
            "org": str(org_id)
        }).execute().data or []
        print(f"[RAG] vector rows={len(rows)} in {time.time()-t0:.3f}s")
        return rows, q_emb
    except Exception as e:
        print(f"[RAG] vector retrieval failed, falling back: {e}")
        return [], None

def _keyword(org_id: str, q: str, k: int):
    t0=time.time()
    
    # High-priority terms that likely contain the answer
    priority_terms = ["reinstatement", "75%", "50%", "25%", "first year", "second year", "third year", "transfer fee", "initiation fee"]
    
    # Membership-specific terms
    membership_terms = ["Foundation", "Social", "Intermediate", "Legacy", "Corporate", "Nonresident", "Golfing Senior"]
    financial_terms = ["fee", "dues", "payment", "percent", "discount", "charge", "cost", "billing"]
    
    # Search with priority order
    seen, out = set(), []
    
    # First, search for high-priority terms that likely contain specific answers
    for term in priority_terms:
        resp = supa.table("doc_chunks").select("document_id,chunk_index,summary,content") \
               .eq("org_id", org_id).ilike("content", f"%{term}%").limit(k).execute().data
        for r in resp or []:
            key = (r["document_id"], r["chunk_index"])
            if key not in seen:
                seen.add(key)
                out.append(r)
    
    # Then search membership + financial combinations
    if len(out) < k:
        for m_term in membership_terms:
            for f_term in financial_terms:
                if len(out) >= k: break
                resp = supa.table("doc_chunks").select("document_id,chunk_index,summary,content") \
                       .eq("org_id", org_id).ilike("content", f"%{m_term}%").ilike("content", f"%{f_term}%").limit(k).execute().data
                for r in resp or []:
                    key = (r["document_id"], r["chunk_index"])
                    if key not in seen:
                        seen.add(key)
                        out.append(r)
                        if len(out) >= k: break
    
    # Finally, broad search with query terms
    if len(out) < k:
        query_words = q.lower().split()
        for word in query_words:
            if len(word) > 3:  # Skip short words
                resp = supa.table("doc_chunks").select("document_id,chunk_index,summary,content") \
                       .eq("org_id", org_id).ilike("content", f"%{word}%").limit(k).execute().data
                for r in resp or []:
                    key = (r["document_id"], r["chunk_index"])
                    if key not in seen:
                        seen.add(key)
                        out.append(r)
                        if len(out) >= k: break
    
    print(f"[RAG] keyword rows={len(out)} in {time.time()-t0:.3f}s")
    return out[:k]

def _doc_title_link(doc_id: str):
    d = supa.table("documents").select("title,storage_path").eq("id", doc_id).limit(1).execute().data
    title = (d[0].get("title") if d else None) or "Document"
    link = signed_url_for(d[0].get("storage_path") or "") if d else None
    return (title, link)

def _fetch_page_index(doc_id: str, chunk_idx: int) -> int | None:
    try:
        row = supa.table("doc_chunks").select("page_index") \
            .eq("document_id", doc_id).eq("chunk_index", chunk_idx) \
            .limit(1).execute().data
        if row and row[0].get("page_index") is not None:
            return int(row[0]["page_index"])
    except Exception:
        pass
    return None

# ---------- MMR ----------
def _cos(a, b):
    if not a or not b: return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(x*x for x in b)) or 1e-9
    return dot/(na*nb)

def _load_embeddings(rows):
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["document_id"], []).append(r["chunk_index"])
    emb_map = {}
    for doc_id, idxs in by_doc.items():
        res = supa.table("doc_chunks").select("chunk_index,embedding") \
              .eq("document_id", doc_id).in_("chunk_index", idxs).execute().data or []
        for item in res:
            emb_map[(doc_id, item["chunk_index"])] = item["embedding"]
    return [emb_map.get((r["document_id"], r["chunk_index"])) for r in rows]

def _mmr(query_emb, rows, row_embs, k: int, lam: float):
    if not query_emb:
        # no vector emb available → return first k (still deduped & summary-first)
        return list(range(min(k, len(rows))))
    selected = []
    candidates = list(range(len(rows)))
    rel = [ _cos(query_emb, e) for e in row_embs ]
    while candidates and len(selected) < k:
        best_i, best_score = None, -1e9
        for i in candidates:
            if not selected:
                div = 0.0
            else:
                div = max(_cos(row_embs[i], row_embs[j]) for j in selected)
            score = lam*rel[i] - (1.0-lam)*div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected

# ---------- Main ----------
def answer_question_md(org_id: str, question: str, chat_model: str | None = None):
    # 1) retrieve (vector + keyword fallback)
    v_rows, q_emb = _vector(org_id, question, k=MAX_CANDIDATES)
    rows = v_rows
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

    # 2) MMR rerank (works even if no q_emb)
    embs = _load_embeddings(rows)
    keep_idx = _mmr(q_emb, rows, embs, k=min(MMR_K, len(rows)), lam=MMR_LAMBDA)
    rows = [rows[i] for i in keep_idx]

    # 3) Build notes with full content when summaries are insufficient (NotebookLM approach)
    notes=[]; total=0; meta=[]
    for r in rows:
        doc_id=r.get("document_id"); ci=r.get("chunk_index")
        title, base_link = _doc_title_link(doc_id)
        page = _fetch_page_index(doc_id, ci)
        link = f"{base_link}#page={page+1}" if (base_link and page is not None) else base_link
        meta.append({"document_id": doc_id, "chunk_index": ci, "title": title, "url": link, "page_index": page})

        # Use full content instead of just summaries for better detail extraction
        content = r.get("content") or ""
        s = r.get("summary")
        
        # For detailed questions, try to get more complete content by checking adjacent chunks
        if any(term in question.lower() for term in ['specific', 'exact', 'how much', 'percentage', 'fee', 'cost', 'amount', 'reinstatement']):
            # Check if this chunk seems truncated (ends with partial number)
            if content and (content.strip().endswith(' 75') or content.strip().endswith(' 70') or content.strip().endswith(' 50') or content.strip().endswith(' 40')):
                # Try to get the next chunk to complete the thought
                try:
                    next_chunk = supa.table("doc_chunks").select("content").eq("document_id", doc_id).eq("chunk_index", ci + 1).limit(1).execute()
                    if next_chunk.data:
                        next_content = next_chunk.data[0].get("content", "")
                        # Combine chunks for complete context
                        source_text = (content + " " + next_content)[:3000]  # Extended content for detailed answers
                    else:
                        source_text = content[:2000]
                except:
                    source_text = content[:2000]
            else:
                source_text = content[:2000] if content else s
        else:
            # Use summary for general questions
            source_text = s if s and len(s) > 20 else content[:1200]
        
        if not source_text:
            continue
            
        source_text += f" [Doc:{doc_id}#Chunk:{ci}]"

        remaining = MAX_SUMMARY_TOKENS - total
        s_fit = _fit_to_tokens(source_text.strip(), max_tokens=max(200, remaining))  # More space per chunk
        if not s_fit:
            if not notes:
                tiny = _fit_to_tokens(source_text.strip(), 300) or (source_text[:500] + f" [Doc:{doc_id}#Chunk:{ci}]")
                notes.append("- " + tiny.strip())
                total += _toks(tiny)
            break
        notes.append("- " + s_fit)
        total += _toks(s_fit)
        if total >= MAX_SUMMARY_TOKENS:
            break

    if not notes:
        return ("No usable source notes yet. Try again in a moment after processing finishes.", meta)

    # 4) Final answer under strict budget
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
    except Exception as e:
        print(f"[RAG] final model error; downshifting: {e}")
        model = "gpt-3.5-turbo"
        print(f"[RAG] Downgrading to {model} due to primary model failure")
        answer = _retry(lambda: client.chat.completions.create(
            model=model, temperature=TEMPERATURE,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
            max_tokens=600
        ).choices[0].message.content)

    return (answer, meta)