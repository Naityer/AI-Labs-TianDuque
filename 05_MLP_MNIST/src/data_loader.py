import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_data(batch_size, val_ratio=0.2):
    """Carga MNIST y crea DataLoaders para train/val/test"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Descargar datasets
    train_data = datasets.MNIST(
        root='./data', 
        train=True,
        download=True,
        transform=transform
    )
    
    test_data = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )
    
    # Split train/val
    val_size = int(len(train_data) * val_ratio)
    train_size = len(train_data) - val_size
    train_set, val_set = random_split(train_data, [train_size, val_size])
    
    # DataLoaders (num_workers=0 para evitar problemas en Windows)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader
