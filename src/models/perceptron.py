class Perceptron:
    """
    Implementação manual e simplificada de um Perceptron em Python puro.
    Focada no problema de recomendação de músicas (Energy vs Loudness).
    """
    def __init__(self, weights=None, bias=0.1):
        """
        Inicializa o Perceptron.

        Parâmetros:
        -----------
        weights : dict
            Dicionário de pesos. Ex: {'energy': 0.8, 'loudness': 0.2}
        bias : float
            Valor do viés (bias).
        """
        # Pesos padrão baseados na referência (ajustáveis manualmente)
        default_weights = {'energy': 0.8, 'loudness': 0.2}
        self.weights = weights if weights is not None else default_weights
        self.bias = bias

    def predict(self, energy, loudness):
        """
        Realiza a previsão para uma música com base em energia e barulho.
        
        Lógica:
        1. Normaliza loudness: (loudness + 10) / 10
        2. Calcula z = (energy * w_energy) + (loudness_norm * w_loudness) + bias
        3. Aplica degrau: se z >= 0 -> 1 (Recomendada/Festa), senão 0 (Não recomendada/Relax)

        Parâmetros:
        -----------
        energy : float
            Valor de energia (0.0 a 1.0).
        loudness : float
            Valor de loudness (dB, ex: -60.0 a 0.0).

        Retorna:
        --------
        dict
            Dicionário com o resultado da classificação e valor de ativação.
        """
        # 1. Normalização (conforme referência)
        # A ideia é trazer o loudness (negativo) para uma escala comparável
        loudness_norm = (loudness + 10) / 10
        
        # 2. Cálculo da saída linear (z)
        w_energy = self.weights.get('energy', 0.0)
        w_loudness = self.weights.get('loudness', 0.0)
        
        linear_output = (energy * w_energy) + (loudness_norm * w_loudness) + self.bias
        
        # 3. Função de Ativação (Degrau)
        # Se saída linear >= 0.5, consideramos classe 1 (ex: Festa)
        # Ajustado para 0.5 para alinhar com os pesos manuais padrão.
        prediction = 1 if linear_output >= 0.5 else 0 
        
        return {
            "prediction": prediction,
            "activation": linear_output,
            "normalized_loudness": loudness_norm
        }
