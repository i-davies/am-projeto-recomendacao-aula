from fastapi import APIRouter
from src.api.v1 import recommendation

api_router = APIRouter(prefix="/api/v1")

# Inclui as rotas do módulo de recomendação
api_router.include_router(recommendation.router, tags=["recommendation"])
