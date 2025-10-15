# startup.py  (at repo root)
import os, sys, traceback
sys.path.append(os.path.dirname(__file__))
try:
    from main import app as app
except Exception as _e:
    print("startup.py: main import failed:")
    traceback.print_exception(_e)
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/health")
    def _h(): return {"status":"ok","source":"fallback"}
