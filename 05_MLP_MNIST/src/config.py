from dataclasses import dataclass

@dataclass
class Config:
    # Hiperparámetros para experimentos (Variante 2)
    learning_rates = [0.1, 0.01, 0.001]          # 3 valores de LR
    batch_sizes = [1, 32, 128]                    # Incluye batch=1
    hidden_layers = [0, 1, 2]                     # 0=lineal, 1-2 capas
    hidden_sizes = [64, 128, 256]                 # Neuronas por capa
    optimizers = ["sgd", "sgd_momentum", "adam"]  # Tipos de optimizadores
    
    # Configuraciones fijas para MNIST
    input_size = 784
    output_size = 10
    epochs = 15
    val_ratio = 0.2
