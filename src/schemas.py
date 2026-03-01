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