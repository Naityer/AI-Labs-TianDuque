import torch
import random
import numpy as np
import json
import os
from pathlib import Path
from src.model import MLP
from src.trainer import train_model
from src.data_loader import load_data
from src.config import Config
from src.utils import plot_comparative_results, plot_individual_results

def ensure_dirs():
    """Crea todos los directorios necesarios"""
    os.makedirs('experiments/exp_optim', exist_ok=True)
    os.makedirs('report/figures', exist_ok=True)

def set_seed(seed=42):
    """Fija semillas para asegurar reproducibilidad"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_experiment_results(config, history, optim_name, model):
    """Guarda resultados y modelo entrenado"""
    exp_dir = f"experiments/exp_optim/{optim_name}"
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # Guardar métricas numéricas
    with open(f"{exp_dir}/metrics.json", 'w') as f:
        json.dump({
            'config': config,
            'history': {
                'train_loss': history['train_loss'],
                'val_acc': history['val_acc'],
                'train_acc': history.get('train_acc', []),
                'training_time': history.get('training_time', 0)
            }
        }, f, indent=2)
    
    # Guardar modelo entrenado
    torch.save(model.state_dict(), f"{exp_dir}/model.pt")
    
    # Guardar gráfica individual
    plot_individual_results(history, save_path=f"{exp_dir}/training_curves.png")

def main():
    set_seed(42)
    ensure_dirs()
    config = Config()
    all_results = []
    
    for optim in config.optimizers:
        print(f"\n🔍 Experimentando con: {optim.upper()}")
        
        train_loader, val_loader, _ = load_data(batch_size=64)
        model = MLP(
            hidden_layers=[128], 
            loss_fn='cross_entropy',
            use_batchnorm=True
        )
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time: start_time.record()
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_type=optim,
            lr=0.01,
            epochs=config.epochs
        )
        if end_time: end_time.record()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            history['training_time'] = start_time.elapsed_time(end_time) / 1000
        else:
            history['training_time'] = history.get('training_time', 0)
        
        save_experiment_results(
            config={
                'optimizer': optim,
                'lr': 0.01,
                'batch_size': 64,
                'hidden_size': 128,
                'num_layers': 1
            },
            history=history,
            optim_name=optim,
            model=model
        )
        
        all_results.append({
            'optimizer': optim,
            'history': history,
            'config': {
                'optimizer': optim,
                'lr': 0.01,
                'batch_size': 64,
                'hidden_size': 128,
                'num_layers': 1
            }
        })
    
    plot_comparative_results(
        all_results, 
        save_path="report/figures/optimizers_comparison.png"
    )
    print("✅ Todos los resultados guardados en experiments/exp_optim")

if __name__ == "__main__":
    main()
