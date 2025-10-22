from fastapi import APIRouter
router = APIRouter(prefix="/api", tags=["metrics"])
@router.get("/metrics")
def metrics_api():
    return {"counters": {}}
