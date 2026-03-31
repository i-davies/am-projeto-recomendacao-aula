from fastapi import APIRouter, HTTPException, Depends
from typing import List
import pandas as pd
from pydantic import BaseModel, Field

from src.services.regression_model import MusicPopularityRegressor

router = APIRouter()


class RegressionTrackInput(BaseModel):
    """Dados de uma música para predição de Popularity Score."""
    track_name: str
    danceability: float = Field(ge=0, le=1, description="Quão dançável (0 a 1)")
    energy: float = Field(ge=0, le=1, description="Intensidade e atividade (0 a 1)")
    loudness_norm: float = Field(ge=0, le=1, description="Volume normalizado (0 a 1)")
    speechiness: float = Field(ge=0, le=1, default=0.05, description="Presença de voz falada (0 a 1)")
    acousticness: float = Field(ge=0, le=1, default=0.5, description="Quão acústica (0 a 1)")
    instrumentalness: float = Field(ge=0, le=1, default=0.0, description="Quão instrumental (0 a 1)")
    liveness: float = Field(ge=0, le=1, default=0.2, description="Presença de audiência ao vivo (0 a 1)")
    valence: float = Field(ge=0, le=1, default=0.5, description="Positividade musical (0 a 1)")
    tempo: float = Field(ge=0, le=300, default=120.0, description="BPM da música")


class RegressionPredictRequest(BaseModel):
    tracks: List[RegressionTrackInput]


class RegressionTrackResult(BaseModel):
    track_name: str
    predicted_popularity: float
    interpretation: str
    confidence_range: dict
    debug: dict


class RegressionPredictResponse(BaseModel):
    results: List[RegressionTrackResult]
    total: int
    model_info: dict


def get_regressor() -> MusicPopularityRegressor:
    regressor = MusicPopularityRegressor()
    regressor.load()
    return regressor


@router.post("/model/regression/predict", response_model=RegressionPredictResponse)
def regression_predict(
    request: RegressionPredictRequest,
    regressor: MusicPopularityRegressor = Depends(get_regressor),
):
    try:
        feature_columns = [
            "danceability", "energy", "loudness_norm", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo",
        ]

        rows = []
        for t in request.tracks:
            rows.append({
                "danceability": t.danceability,
                "energy": t.energy,
                "loudness_norm": t.loudness_norm,
                "speechiness": t.speechiness,
                "acousticness": t.acousticness,
                "instrumentalness": t.instrumentalness,
                "liveness": t.liveness,
                "valence": t.valence,
                "tempo": t.tempo,
            })

        df = pd.DataFrame(rows, columns=feature_columns)
        result = regressor.predict(df)

        mae = regressor.metrics.get("mae", 15.0) if regressor.metrics else 15.0
        results = []

        for i, track in enumerate(request.tracks):
            pred = result["predictions"][i]
            interp = result["interpretations"][i]

            results.append(RegressionTrackResult(
                track_name=track.track_name,
                predicted_popularity=pred,
                interpretation=interp,
                confidence_range={
                    "min": round(max(0, pred - mae), 1),
                    "max": round(min(100, pred + mae), 1),
                },
                debug={
                    "features_used": feature_columns,
                    "feature_values": df.iloc[i].to_dict(),
                },
            ))

        model_info = regressor.metrics or {"mae": "N/A", "rmse": "N/A", "r2": "N/A"}

        return RegressionPredictResponse(
            results=results,
            total=len(results),
            model_info=model_info,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo de regressão não encontrado. Rode 'uv run python scripts/train_regression.py' primeiro. Erro: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))