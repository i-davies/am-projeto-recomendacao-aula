from pydantic import BaseModel
from typing import Optional


class MusicFeatures(BaseModel):
    energy: float
    loudness: float


class TrackRequest(BaseModel):
    track_id: Optional[str] = "unknown"
    track_name: str
    artist_name: str
    features: MusicFeatures


class TrackResponse(BaseModel):
    track: str
    artist: str
    recommendation: str
    debug_info: dict

class BatchTrackItem(BaseModel):
    """Uma música dentro de um lote (batch)."""
    track_name: str
    artist_name: str
    features: MusicFeatures


class TrackBatchRequest(BaseModel):
    """Requisição com várias músicas de uma vez."""
    tracks: list[BatchTrackItem]


class TrackBatchResponse(BaseModel):
    """Resposta com o resultado de todas as músicas."""
    results: list[TrackResponse]
    total: int
    summary: dict