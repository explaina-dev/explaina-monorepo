import os, sys, traceback
sys.path.append(os.path.dirname(__file__))

try:
    from main import app as real_app  # your real FastAPI app
    # guarantee /health exists on the real app
    @real_app.get("/health")
    def _health_ok():
        return {"status": "ok", "source": "main"}
    app = real_app

except Exception as _import_err:
    # print full traceback to Azure log stream so we can see the cause
    print("startup.py: import of main.app failed:")
    traceback.print_exception(_import_err)

    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/health")
    def _health_fallback():
        return {"status": "ok", "source": "fallback"}
