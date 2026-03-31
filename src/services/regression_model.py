import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MusicPopularityRegressor:
    """
    Serviço de regressão usando Rede Neural Multicamadas (MLP).

    Usa o MLPRegressor do scikit-learn para prever o Popularity Score
    (valor contínuo de 0 a 100) de uma música com base nas suas
    características técnicas.
    """

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "regressor_model.joblib"
        self.scaler_path = self.model_dir / "regressor_scaler.joblib"
        self.model: MLPRegressor | None = None
        self.scaler: MinMaxScaler | None = None
        self.feature_names: list[str] | None = None
        self.metrics: dict | None = None

    def train(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        hidden_layers: tuple = (64, 32),
        max_iter: int = 500,
        random_state: int = 42,
    ) -> "MusicPopularityRegressor":
        print(f"[REGRESSOR] Treinando rede neural com arquitetura {hidden_layers}...")
        print(f"[REGRESSOR] Features de entrada: {X_train.shape[1]} | Amostras: {X_train.shape[0]}")

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=random_state,
            activation="relu",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )

        self.model.fit(X_scaled, y_train)
        self.save()
        print(f"[REGRESSOR] Treinamento concluído em {self.model.n_iter_} épocas.")
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> dict:
        if self.model is None:
            self.load()

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        predictions = np.clip(predictions, 0, 100)

        interpretations = []
        for p in predictions:
            if p >= 70:
                interpretations.append("Alta Popularidade")
            elif p >= 40:
                interpretations.append("Média Popularidade")
            else:
                interpretations.append("Baixa Popularidade")

        return {
            "predictions": predictions.round(1).tolist(),
            "interpretations": interpretations,
        }

    def evaluate(self, X_test: np.ndarray | pd.DataFrame, y_test: np.ndarray | pd.Series) -> dict:
        if self.model is None:
            self.load()

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_pred = np.clip(y_pred, 0, 100)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        self.metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

        report = (
            f"  MAE  (Erro Médio Absoluto):      {mae:.2f} pontos\n"
            f"  RMSE (Raiz Erro Quadrático):      {rmse:.2f} pontos\n"
            f"  R2   (Variação Explicada):        {r2:.4f} ({r2 * 100:.1f}%)"
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "report": report,
        }

    def save(self):
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
        }
        joblib.dump(state, self.model_path)
        print(f"[REGRESSOR] Modelo salvo em: {self.model_path}")

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo de regressão não encontrado em: {self.model_path}. "
                "Rode 'uv run python scripts/train_regression.py' primeiro."
            )
        state = joblib.load(self.model_path)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_names = state.get("feature_names")
        self.metrics = state.get("metrics")
        print(f"[REGRESSOR] Modelo carregado de: {self.model_path}")