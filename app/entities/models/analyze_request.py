from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    video_name: str
    match_id: int