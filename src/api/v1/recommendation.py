from fastapi import APIRouter, Depends
from functools import lru_cache
from src.schemas import (
    TrackRequest, TrackResponse,
    TrackBatchRequest, TrackBatchResponse,
)
from src.models.perceptron import Perceptron
from src.models.perceptron_numpy import PerceptronNumpy

router = APIRouter()


# Injeção de Dependência — carrega o modelo uma vez só
@lru_cache
def get_perceptron() -> Perceptron:
    return Perceptron(weights={'energy': 0.8, 'loudness': 0.2}, bias=0.1)


@lru_cache
def get_perceptron_numpy() -> PerceptronNumpy:
    return PerceptronNumpy()


# ---- Endpoint Individual (Semana 02) ----

@router.post("/recommend/predict", response_model=TrackResponse)
def predict_track(request: TrackRequest, model: Perceptron = Depends(get_perceptron)):
    """Predição para UMA música (Perceptron Manual)."""
    result = model.predict(request.features.energy, request.features.loudness)
    mood = "Festa/Agitada" if result["prediction"] == 1 else "Relax/Calma"

    return {
        "track": request.track_name,
        "artist": request.artist_name,
        "recommendation": mood,
        "debug_info": result,
    }


# ---- Endpoint Batch (Semana 03 — NumPy) ----

@router.post("/recommend/predict-batch", response_model=TrackBatchResponse)
def predict_batch(request: TrackBatchRequest, model: PerceptronNumpy = Depends(get_perceptron_numpy)):
    """Predição para VÁRIAS músicas de uma vez (Perceptron NumPy)."""

    # 1. Monta a matriz: cada linha é [energy, loudness]
    features_matrix = [
        [t.features.energy, t.features.loudness]
        for t in request.tracks
    ]

    # 2. Predição em lote via NumPy (np.dot, sem loop)
    batch_results = model.predict_batch(features_matrix)

    # 3. Monta a resposta
    results = []
    festa_count = 0


    # Extrai os arrays resultantes do dicionário
    predictions = batch_results["prediction"].tolist()
    activations = batch_results["activation"].tolist()
    loudnesses = batch_results["normalized_loudness"].tolist()

    for i, track in enumerate(request.tracks):
        pred_val = predictions[i]
        mood = "Festa/Agitada" if pred_val == 1 else "Relax/Calma"
        if pred_val == 1:
            festa_count += 1

        debug_info = {
            "prediction": pred_val,
            "activation": activations[i],
            "normalized_loudness": loudnesses[i]
        }

        results.append({
            "track": track.track_name,
            "artist": track.artist_name,
            "recommendation": mood,
            "debug_info": debug_info,
        })

    total = len(results)
    return {
        "results": results,
        "total": total,
        "summary": {"festa": festa_count, "relax": total - festa_count},
    }