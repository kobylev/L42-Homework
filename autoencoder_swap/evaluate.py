import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from config import DEVICE, ASSETS_DIR

def plot_loss_curves(loss_cars, loss_planes):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_cars) + 1), loss_cars, label='AE_cars Training Loss', marker='o')
    plt.plot(range(1, len(loss_planes) + 1), loss_planes, label='AE_planes Training Loss', marker='o')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(ASSETS_DIR, "loss_curves.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {save_path}")

def get_n_images(data_loader, n):
    imgs = []
    for batch, _ in data_loader:
        imgs.append(batch)
        if sum(x.size(0) for x in imgs) >= n:
            break
    return torch.cat(imgs, dim=0)[:n].to(DEVICE)

def plot_pair_grid(originals, outputs, title, filename, grid_size=10):
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size))
    plt.suptitle(title, fontsize=16)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            orig = originals[idx].cpu().squeeze().permute(1, 2, 0).numpy().clip(0, 1)
            out = outputs[idx].cpu().squeeze().permute(1, 2, 0).numpy().clip(0, 1)
            
            # Concatenate side-by-side: Original Left, Reconstructed/Translated Right
            pair = np.concatenate([orig, out], axis=1)
            
            axes[i, j].imshow(pair)
            axes[i, j].axis('off')
            
    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Grid saved to {save_path}")

def visualize_reconstruction(model, data_loader, title, filename, num_images=100):
    model.eval()
    test_images = get_n_images(data_loader, num_images)
    
    with torch.no_grad():
        reconstructed = model(test_images)
        
    plot_pair_grid(test_images, reconstructed, title, filename, grid_size=10)

def visualize_cross_domain_grid(ae_cars, ae_planes, car_loader, plane_loader, num_images=100):
    ae_cars.eval()
    ae_planes.eval()
    
    test_cars = get_n_images(car_loader, num_images)
    test_planes = get_n_images(plane_loader, num_images)
    
    with torch.no_grad():
        # Car as Plane
        latent_cars = ae_cars.encoder(test_cars)
        cars_as_planes = ae_planes.decoder(latent_cars)
        
        # Plane as Car
        latent_planes = ae_planes.encoder(test_planes)
        planes_as_cars = ae_cars.decoder(latent_planes)
        
    plot_pair_grid(test_cars, cars_as_planes, "Cars Translated to Planes (Original | Translated)", "cars_as_planes_grid.png", grid_size=10)
    plot_pair_grid(test_planes, planes_as_cars, "Planes Translated to Cars (Original | Translated)", "planes_as_cars_grid.png", grid_size=10)
