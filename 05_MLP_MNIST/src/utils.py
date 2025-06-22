import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path

plt.style.use('seaborn-v0_8') 
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

def evaluate(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: str = 'cpu') -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def save_experiment_results(config: Dict, history: Dict, experiment_dir: str):
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{experiment_dir}/metrics.json", 'w') as f:
        json.dump({
            'config': config,
            'history': {
                'train_loss': history['train_loss'],
                'val_acc': history['val_acc'],
                'train_acc': history.get('train_acc', []),
                'training_time': history.get('training_time', 0)
            }
        }, f, indent=2)
    plot_training_curves(history, save_path=f"{experiment_dir}/training_curves.png")

def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    if 'val_acc' in history:
        epochs = range(1, len(history['val_acc']) + 1)
        ax2.plot(epochs, history['val_acc'], label='Validation')
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], '--', label='Training')
    ax2.set_title("Accuracy Progress")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_optimizers_comparison(results: List[Dict], save_path: Optional[str] = None):
    plt.figure(figsize=(14, 6))
    
    # Validation accuracy over epochs
    plt.subplot(1, 2, 1)
    for res in results:
        plt.plot(res['history']['val_acc'], 
                 label=f"{res['config']['optimizer'].upper()} (Max: {max(res['history']['val_acc']):.1f}%)")
    plt.title("Validation Accuracy by Optimizer")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    # Final accuracy bar chart
    plt.subplot(1, 2, 2)
    final_accs = {res['config']['optimizer']: res['history']['val_acc'][-1] for res in results}
    optimizers = list(final_accs.keys())
    accuracies = list(final_accs.values())
    plt.bar(optimizers, accuracies, color='skyblue')
    plt.title("Final Validation Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': get_model_config(model)
    }, path)

def load_checkpoint(path: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None):
    checkpoint = torch.load(path)
    if model:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return {
        'epoch': checkpoint['epoch'],
        'model': model,
        'optimizer': optimizer,
        'config': checkpoint.get('config', {})
    }

def get_model_config(model: torch.nn.Module) -> Dict:
    return {
        'input_size': model.net[0].in_features if hasattr(model, 'net') else None,
        'hidden_layers': [layer.out_features for layer in model.net 
                         if isinstance(layer, torch.nn.Linear)][:-1],
        'output_size': model.net[-1].out_features if hasattr(model, 'net') else None,
        'activation': type(model.net[2]).__name__ if len(model.net) > 2 else None
    }

def plot_individual_results(history: Dict, save_path: Optional[str] = None):
    plot_training_curves(history, save_path=save_path)

def plot_comparative_results(results: List[Dict], save_path: Optional[str] = None):
    plot_optimizers_comparison(results, save_path=save_path)
