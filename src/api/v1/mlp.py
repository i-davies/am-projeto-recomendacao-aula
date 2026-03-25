# src/api/v1/mlp.py
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import pandas as pd
from pydantic import BaseModel

from src.services.feature_engineer import FeatureEngineer
from src.services.mlp_classifier import MusicMLPClassifier

router = APIRouter()


class MLPTrackInput(BaseModel):
    """Dados de entrada de uma música para predição."""
    track_name: str
    track_genre: str
    tempo: float
    popularity: float
    danceability: float = 0.5
    energy: float = 0.5


class MLPTrackResult(BaseModel):
    """Resultado da predição para uma única música."""
    track_name: str
    prediction: str
    probability: float
    debug: dict


class MLPPredictRequest(BaseModel):
    """Corpo da requisição: uma ou mais músicas."""
    tracks: List[MLPTrackInput]


class MLPPredictResponse(BaseModel):
    """Resposta completa da predição."""
    results: List[MLPTrackResult]
    total: int
    summary: dict

def get_feature_engineer() -> FeatureEngineer:
    fe = FeatureEngineer()
    if not fe.transformer_path.exists():
        dummy = pd.DataFrame([
            {"tempo": 120.0, "popularity": 50.0, "danceability": 0.5,
             "energy": 0.5, "track_genre": "pop"},
            {"tempo": 80.0, "popularity": 10.0, "danceability": 0.3,
             "energy": 0.2, "track_genre": "rock"},
        ])
        fe.fit(dummy, ["tempo", "popularity", "danceability", "energy"], ["track_genre"])
    else:
        fe.load()
    return fe


def get_mlp_model() -> MusicMLPClassifier:
    mlp = MusicMLPClassifier()
    mlp.load()
    return mlp

@router.post("/model/mlp/predict", response_model=MLPPredictResponse)
def mlp_predict(
    request: MLPPredictRequest,
    fe: FeatureEngineer = Depends(get_feature_engineer),
    mlp: MusicMLPClassifier = Depends(get_mlp_model),
):
    """Prevê se uma ou mais músicas serão 'Curtidas' ou 'Não Curtidas'."""
    try:
        df = pd.DataFrame([t.model_dump() for t in request.tracks])
        df_features = fe.transform(df)
        result = mlp.predict(df_features)

        results = []
        liked_count = 0

        for i, track in enumerate(request.tracks):
            pred_val = result["predictions"][i]
            if pred_val == 1:
                liked_count += 1

            results.append(MLPTrackResult(
                track_name=track.track_name,
                prediction=result["labels"][i],
                probability=round(result["probabilities"][i], 4),
                debug={
                    "raw_prediction": pred_val,
                    "features_used": df_features.columns.tolist(),
                },
            ))

        return MLPPredictResponse(
            results=results,
            total=len(results),
            summary={"curtidas": liked_count, "nao_curtidas": len(results) - liked_count},
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo não encontrado. Rode 'train_mlp.py' primeiro. Erro: {e}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))