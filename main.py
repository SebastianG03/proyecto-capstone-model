import asyncio

from app.api.v1.endpoints import analyze_router
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.retrain.retrain import RetrainingOrchestrator
from app.utils.routes import ensure_directories
from sqlalchemy import event

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\033c", end="")
    ensure_directories()
    loop = asyncio.get_event_loop()
    orchestrator = RetrainingOrchestrator()
    retrain_task = loop.run_in_executor(
        None,
        orchestrator.retrain,
    )
    print("🚀 API iniciada")
    yield
    orchestrator.stop()
    retrain_task.cancel()
    print("🛑 API detenida")


def run_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(analyze_router)
    return app


app = run_app()