# scripts/backfill_summaries.py
import os
import sys
sys.path.append('..')
from openai import OpenAI
from lib.supa import supa

client = OpenAI()
CHAT_COMPRESS = os.getenv("CHAT_COMPRESS","gpt-4o-mini")

def summarize(doc_id, idx, text):
    text = (text or "").strip()
    if not text:
        return ""
    text = text[:1500]
    prompt = (
        "Condense to 1–3 short factual sentences preserving dates, amounts, names, obligations. "
        f"End with [Doc:{doc_id}#Chunk:{idx}].\n\nEXCERPT:\n{text}"
    )
    try:
        r = client.chat.completions.create(
            model=CHAT_COMPRESS, temperature=0.0,
            messages=[{"role":"user","content":prompt}], max_tokens=140
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return (text[:400] + f"... [Doc:{doc_id}#Chunk:{idx}]")

def main():
    total=0
    while True:
        rows = supa.table("doc_chunks") \
            .select("document_id,chunk_index,summary,content") \
            .is_("summary","null").limit(50).execute().data
        if not rows:
            break
        for r in rows:
            s = summarize(r["document_id"], r["chunk_index"], r.get("content") or "")
            supa.table("doc_chunks").update({"summary": s}) \
               .eq("document_id", r["document_id"]) \
               .eq("chunk_index", r["chunk_index"]).execute()
            total += 1
        print(f"Updated {total} summaries…")
    print("Backfill complete.")

if __name__ == "__main__":
    main()