import os

def ocr_extract_pdf(bytes_data: bytes) -> str:
    """
    Optional OCR via AWS Textract. Returns "" if OCR is not configured or fails.
    """
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("AWS_REGION")):
        return ""
    try:
        import boto3
        import time
        textract = boto3.client("textract", region_name=os.getenv("AWS_REGION"))
        resp = textract.start_document_text_detection(DocumentLocation={"Bytes": bytes_data})
        job_id = resp["JobId"]
        # poll until done (simple MVP polling)
        while True:
            r = textract.get_document_text_detection(JobId=job_id)
            status = r["JobStatus"]
            if status in ("SUCCEEDED", "FAILED"):
                if status == "FAILED":
                    return ""
                blocks = []
                next_token = None
                # paginate
                while True:
                    page = textract.get_document_text_detection(JobId=job_id, NextToken=next_token) if next_token else r
                    for b in page.get("Blocks", []):
                        if b.get("BlockType") == "LINE":
                            blocks.append(b.get("Text", ""))
                    next_token = page.get("NextToken")
                    if not next_token:
                        break
                return "\n".join(blocks)
            time.sleep(1.0)
    except Exception:
        return ""
