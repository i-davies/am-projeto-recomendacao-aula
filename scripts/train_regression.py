# -*- coding: utf-8 -*-
"""
Script de treinamento do regressor (Semana 07).
"""
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.services.regression_model import MusicPopularityRegressor


NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "loudness_norm",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def main():
    clean_path = project_root / "data" / "processed" / "dataset_clean.csv"

    print("=" * 60)
    print("  [REGRESSOR] Treinamento do Modelo de Regressão (Semana 07)")
    print("=" * 60)

    if not clean_path.exists():
        print(f"\n[ERRO] Dataset limpo não encontrado em: {clean_path}")
        print("DICA: Rode 'uv run python scripts/clean_dataset.py' primeiro.")
        return

    print("\n[1/5] Carregando dataset limpo...")
    df = pd.read_csv(clean_path)
    print(f"       Shape: {df.shape} ({df.shape[0]} músicas, {df.shape[1]} colunas)")

    print("\n[2/5] Separando features e target...")
    X = df[NUMERIC_FEATURES].copy()
    y = df["popularity"].copy()

    print(f"       Features (X): {list(X.columns)}")
    print("       Target (Y):   popularity")
    print(f"       Amostras:     {len(X)}")

    print("\n[3/5] Dividindo em treino (80%) e teste (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"       Treino: {X_train.shape[0]} amostras")
    print(f"       Teste:  {X_test.shape[0]} amostras")

    print("\n[4/5] Treinando o Regressor MLP...")
    regressor = MusicPopularityRegressor()
    regressor.train(X_train, y_train, hidden_layers=(64, 32), max_iter=500)

    print("\n[5/5] Avaliando no conjunto de teste...")
    results = regressor.evaluate(X_test, y_test)

    print(f"\n{'=' * 60}")
    print("  RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(results["report"])

    mae = results["mae"]
    r2 = results["r2"]
    print("\n  Interpretação:")
    print(f"  -> Erro médio de {mae:.1f} pontos no score de 0-100.")
    print(f"  -> Explica {r2 * 100:.1f}% da variação na popularity.")

    regressor.metrics = {
        "mae": round(mae, 2),
        "rmse": round(results["rmse"], 2),
        "r2": round(r2, 4),
    }
    regressor.save()


if __name__ == "__main__":
    main()