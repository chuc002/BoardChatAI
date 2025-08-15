# scripts/backfill_summaries.py
import os
from openai import OpenAI
from lib.supa import supa

client = OpenAI()
CHAT_COMPRESS = os.getenv("CHAT_COMPRESS","gpt-4o-mini")

def summarize(doc_id, idx, text):
    text = (text or "")[:1500]
    if not text.strip():
        return ""
    prompt = (
        "Condense to 1–3 short factual sentences preserving dates, amounts, names, and obligations. "
        f"End with [Doc:{doc_id}#Chunk:{idx}].\n\nEXCERPT:\n{text}"
    )
    try:
        r = client.chat.completions.create(
            model=CHAT_COMPRESS, temperature=0.0,
            messages=[{"role":"user","content":prompt}], max_tokens=140
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return (text[:500] + f"... [Doc:{doc_id}#Chunk:{idx}]")

def main():
    batch = 50
    total = 0
    while True:
        rows = supa.table("doc_chunks") \
            .select("document_id,chunk_index,content") \
            .is_("summary", "null").limit(batch).execute().data
        if not rows:
            break
        for r in rows:
            s = summarize(r["document_id"], r["chunk_index"], r["content"])
            supa.table("doc_chunks").update({"summary": s}) \
               .eq("document_id", r["document_id"]) \
               .eq("chunk_index", r["chunk_index"]).execute()
            total += 1
        print(f"Updated {total} summaries so far…")
    print("Backfill complete.")

if __name__ == "__main__":
    main()
