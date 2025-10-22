from fastapi import APIRouter
import httpx
import os
import json

router = APIRouter(tags=["health"])

@router.get("/health/worker")
async def health_worker():
    base = os.getenv("RENDER_API_URL", "")
    if not base:
        return {"ok": False, "error": "RENDER_API_URL not set"}
    url = f"{base}/health"
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(url)
        
        if r.status_code != 200:
            return {"ok": False, "url": url, "status": r.status_code, "error": f"Worker returned status {r.status_code}"}
        
        try:
            body = r.json()
            if body.get("status") == "ok":
                return {"ok": True, "url": url, "status": r.status_code, "worker_response": body}
            else:
                return {"ok": False, "url": url, "status": r.status_code, "error": "Unexpected response format", "body": r.text[:200]}
        except json.JSONDecodeError:
            return {"ok": False, "url": url, "status": r.status_code, "error": "Invalid JSON response", "body": r.text[:200]}
            
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e)}
