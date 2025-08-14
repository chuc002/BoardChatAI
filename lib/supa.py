import os
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "bc_documents")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE")

supa: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

# Helper to create a time-limited download URL for a stored file

def signed_url_for(path: str, expires_in: int = 3600) -> str | None:
    try:
        res = supa.storage.from_(SUPABASE_BUCKET).create_signed_url(path, expires_in)
        return res.get("signedURL") or res.get("signed_url")
    except Exception:
        return None

__all__ = ["supa", "SUPABASE_BUCKET", "signed_url_for"]