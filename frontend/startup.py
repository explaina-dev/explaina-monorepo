import os, sys, traceback
sys.path.append(os.path.dirname(__file__))

def _debug_log():
    try:
        print("startup.py: cwd=", os.getcwd())
        print("startup.py: __file__=", __file__)
        print("startup.py: sys.path[0:3]=", sys.path[:3])
        print("startup.py: wwwroot listing ->", os.listdir(os.path.dirname(__file__)))
    except Exception as e:
        print("startup.py: listing error:", e)

_debug_log()

try:
    # run your real app (must be main.py with `app = FastAPI()`)
    from main import app as app
    print("startup.py: imported main.app successfully")
except Exception as _e:
    print("startup.py: main import failed:")
    traceback.print_exception(_e)
    from fastapi import FastAPI
    app = FastAPI()
    @app.get("/health")
    def _h(): return {"status":"ok","source":"fallback"}
