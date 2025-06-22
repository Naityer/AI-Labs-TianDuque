import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from time import time

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer_type: str = "adam",
    lr: float = 0.01,
    epochs: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Entrena el modelo y devuelve historial de métricas
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        optimizer_type: Tipo de optimizador ('sgd', 'sgd_momentum', 'adam')
        lr: Learning rate
        epochs: Número de épocas
        verbose: Mostrar progreso
    Returns:
        Dict con historial de:
        - train_loss: List[float] (pérdida por batch)
        - val_acc: List[float] (accuracy por época)
        - train_acc: List[float] (accuracy por época)
        - training_time: float (segundos)
    """
    # Configurar optimizador (Variante 2)
    optimizers = {
        "sgd": torch.optim.SGD(model.parameters(), lr=lr),
        "sgd_momentum": torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "adam": torch.optim.Adam(model.parameters(), lr=lr)
    }
    optimizer = optimizers[optimizer_type]
    
    criterion = model.loss_fn  # Usa la pérdida definida en el modelo
    history = {
        "train_loss": [],
        "val_acc": [],
        "train_acc": [],
        "training_time": 0
    }
    
    start_time = time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        correct_train, total_train = 0, 0
        
        # Training phase
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X.view(X.shape[0], -1))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            # Calcular accuracy de entrenamiento
            _, predicted = torch.max(outputs.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
        
        # Validation phase
        val_acc = evaluate(model, val_loader)
        train_acc = 100 * correct_train / total_train
        
        # Guardar métricas
        history["train_loss"].extend(epoch_loss)
        history["val_acc"].append(val_acc)
        history["train_acc"].append(train_acc)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {np.mean(epoch_loss):.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Acc: {val_acc:.2f}%")
    
    history["training_time"] = time() - start_time
    return history

def evaluate(model, data_loader) -> float:
    """
    Evalúa el modelo en un DataLoader
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con los datos
    Returns:
        accuracy: Porcentaje de aciertos
    """
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X.view(X.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return 100 * correct / total

def save_checkpoint(model, optimizer, epoch, path):
    """
    Guarda el estado del entrenamiento
    Args:
        model: Modelo a guardar
        optimizer: Optimizador
        epoch: Época actual
        path: Ruta para guardar
    """
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, path)
