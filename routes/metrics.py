from fastapi import APIRouter
from services.metrics import dump

router = APIRouter(prefix="/api", tags=["metrics"])

@router.get("/metrics")
async def get_metrics():
    return dump()
