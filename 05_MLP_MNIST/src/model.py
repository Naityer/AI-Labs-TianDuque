import torch.nn as nn
from typing import Union, List

class MLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_layers: Union[int, List[int]] = [128],
        output_size: int = 10,
        loss_fn: str = 'cross_entropy',
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batchnorm: bool = True  # Nuevo parámetro
    ):
        """
        Args:
            input_size: Tamaño de entrada (para MNIST: 28*28=784)
            hidden_layers: Lista de neuronas por capa oculta (o int para 1 capa)
            output_size: Neuronas de salida (clases)
            loss_fn: Función de pérdida ('cross_entropy', 'mse', etc.)
            activation: Función de activación ('relu', 'leaky_relu', etc.)
            dropout: Probabilidad de dropout (0=desactivado)
            use_batchnorm: Si usa Batch Normalization
        """
        super().__init__()
        
        # Validación de parámetros
        if isinstance(hidden_layers, int):
            hidden_layers = [hidden_layers]
        assert all(size > 0 for size in hidden_layers), "Cada capa debe tener >0 neuronas"
        
        # Construcción de capas
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            # Capa lineal
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # BatchNorm (opcional)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activación
            layers.append(self._get_activation(activation))
            
            # Dropout (opcional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = hidden_size
        
        # Capa de salida (sin activación)
        layers.append(nn.Linear(prev_size, output_size))
        
        # Compilar secuencia
        self.net = nn.Sequential(*layers)
        self.loss_fn = self._get_loss_fn(loss_fn)
        self._init_weights()

    def _init_weights(self):
        """Inicialización Xavier/Glorot para mejores resultados"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _get_activation(self, name: str) -> nn.Module:
        """Selector de funciones de activación"""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "linear": nn.Identity()  # Para modelos lineales (0 capas ocultas)
        }
        return activations.get(name.lower(), nn.ReLU())  # Default: ReLU

    def _get_loss_fn(self, name: str) -> nn.Module:
        """Selector de funciones de pérdida"""
        loss_fns = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'bce': nn.BCEWithLogitsLoss(),  # Para clasificación binaria
            'huber': nn.HuberLoss()
        }
        try:
            return loss_fns[name.lower()]
        except KeyError:
            raise ValueError(f"Función de pérdida no soportada: {name}. Opciones: {list(loss_fns.keys())}")

    def forward(self, x):
        # Aplanar entrada si son imágenes (MNIST: [batch, 1, 28, 28] -> [batch, 784])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
