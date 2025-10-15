import os, sys, traceback
sys.path.append(os.path.dirname(__file__))

try:
    # Try to run the real app (which includes /api/metrics)
    from main import app as app
except Exception:
    # Fallback minimal app so /health still works if main import fails
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/health")
    def _h():
        return {"status":"ok","source":"fallback"}
