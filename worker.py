import os, time, traceback
from lib.supa import supa, SUPABASE_BUCKET
from lib.ingest import upsert_document  # reuse our full ingest (chunks + summaries)

ORG_ID  = os.getenv("DEV_ORG_ID")
USER_ID = os.getenv("DEV_USER_ID")
POLL_SEC = float(os.getenv("WORKER_POLL_SEC", "1.5"))
BATCH    = int(os.getenv("WORKER_BATCH", "3"))

def _download_bytes(storage_path: str) -> bytes:
    return supa.storage.from_(SUPABASE_BUCKET).download(storage_path)

def _update_job_status(job_id: str):
    # If any queued/running remain -> running; else done/error
    items = supa.table("ingest_items").select("status").eq("job_id", job_id).execute().data or []
    statuses = {r["status"] for r in items}
    if not items:
        supa.table("ingest_jobs").update({"status": "error", "error_message": "no items"}).eq("id", job_id).execute()
        return
    if "queued" in statuses or "running" in statuses:
        supa.table("ingest_jobs").update({"status": "running"}).eq("id", job_id).execute()
    else:
        final = "error" if "error" in statuses else "done"
        supa.table("ingest_jobs").update({"status": final, "finished_at": "now()"}).eq("id", job_id).execute()

def _claim_next_items(limit: int):
    # Simple claim loop: pick oldest queued, flip to running atomically per item
    rows = supa.table("ingest_items").select("id,job_id,org_id,storage_path,filename,mime_type,size_bytes,status") \
           .eq("status","queued").order("created_at").limit(limit).execute().data or []
    claimed = []
    for r in rows:
        upd = supa.table("ingest_items").update({"status":"running","started_at":"now()"}) \
              .eq("id", r["id"]).eq("status","queued").execute().data
        if upd: claimed.append(r)
    return claimed

def _process_item(item):
    item_id = item["id"]; job_id = item["job_id"]
    try:
        # pull file, call our existing ingest
        b = _download_bytes(item["storage_path"])
        doc, n = upsert_document(ORG_ID, USER_ID, item["filename"], b, item.get("mime_type") or "application/pdf")
        # mark item done + link to document_id
        supa.table("ingest_items").update({
            "status":"done","finished_at":"now()","document_id":doc["id"]
        }).eq("id", item_id).execute()
    except Exception as e:
        supa.table("ingest_items").update({
            "status":"error","finished_at":"now()","error_message":str(e)[:1000]
        }).eq("id", item_id).execute()
        print("[worker] item error:", e)
        traceback.print_exc()
    finally:
        _update_job_status(job_id)

def main():
    print(f"[worker] start ORG={ORG_ID} USER={USER_ID} poll={POLL_SEC}s batch={BATCH}")
    while True:
        todo = _claim_next_items(BATCH)
        if not todo:
            time.sleep(POLL_SEC); continue
        for it in todo:
            _update_job_status(it["job_id"])
            _process_item(it)

if __name__ == "__main__":
    main()