from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.entities.models import AnalyzeRequest
from app.application.background_task import process_video_async

analyze_router = APIRouter(prefix="/analyze", tags=["analyze"])


@analyze_router.post("/run")
async def analyze_video(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    No espera a que termine el procesamiento:
    Lanza el analisis en segundo plano.
    """
    try:
        background_tasks.add_task(
            process_video_async,
            **payload.model_dump(),
        )

        return {
            "status": "processing",
            "message": (
                "El video esta siendo procesado. "
                "Los resultados se subiran automaticamente cuando esten listos."
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
