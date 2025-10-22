from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.health import router as health_router
from routes.answer import router as answer_router
from routes.metrics import router as metrics_router

app = FastAPI(title="Explaina API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(health_router)
app.include_router(answer_router)
app.include_router(metrics_router)

app.mount("/", StaticFiles(directory="www", html=True), name="spa")
