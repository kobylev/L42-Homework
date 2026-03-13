import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import DATA_DIR, BATCH_SIZE

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    # CIFAR-10 classes: 0 is 'airplane', 1 is 'automobile'
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    # Filter for cars (class 1)
    car_train_indices = np.where(train_targets == 1)[0].tolist()
    car_test_indices = np.where(test_targets == 1)[0].tolist()
    
    # Filter for planes (class 0)
    plane_train_indices = np.where(train_targets == 0)[0].tolist()
    plane_test_indices = np.where(test_targets == 0)[0].tolist()

    car_train_set = Subset(train_dataset, car_train_indices)
    car_test_set = Subset(test_dataset, car_test_indices)
    
    plane_train_set = Subset(train_dataset, plane_train_indices)
    plane_test_set = Subset(test_dataset, plane_test_indices)

    # DataLoaders
    car_train_loader = DataLoader(car_train_set, batch_size=BATCH_SIZE, shuffle=True)
    car_test_loader = DataLoader(car_test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    plane_train_loader = DataLoader(plane_train_set, batch_size=BATCH_SIZE, shuffle=True)
    plane_test_loader = DataLoader(plane_test_set, batch_size=BATCH_SIZE, shuffle=False)

    return car_train_loader, car_test_loader, plane_train_loader, plane_test_loader
