from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from app.infraestructure.services import get_db
from app.entities.models import AnalyzeRequest
from app.application.background_task import process_video_async

analyze_router = APIRouter(
    prefix="/analyze",
    tags=["analyze"]
)


@analyze_router.post("/run")
async def analyze_video(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    No espera a que termine el procesamiento:
    Lanza el análisis en segundo plano.
    """
    try:
        background_tasks.add_task(
            process_video_async,
            payload.video_name,
            payload.match_id,
        )

        return {
            "status": "processing",
            "message": (
                "El video está siendo procesado. "
                "Los resultados se subirán automáticamente cuando estén listos."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
