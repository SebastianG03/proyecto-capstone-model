from app.tasks import analyze_router
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.utils.routes import ensure_directories

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\033c", end="")
    print("🚀 API iniciada")
    ensure_directories()
    yield
    print("🛑 API detenida")

def run_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(analyze_router)
    return app

app = run_app()
