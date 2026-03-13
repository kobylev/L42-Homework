import torch
from config import EPOCHS
from datasets import get_dataloaders
from model import Autoencoder
from train import train_model
from evaluate import plot_loss_curves, visualize_reconstruction, visualize_cross_domain_grid

def main():
    print("Project: Autoencoder Latent Space Translation (Cars to Planes)")
    print("Initializing datasets and dataloaders...")
    car_train_loader, car_test_loader, plane_train_loader, plane_test_loader = get_dataloaders()
    
    print("Creating models...")
    ae_cars = Autoencoder()
    ae_planes = Autoencoder()
    
    print("\n==================================")
    print("Training Cars Autoencoder")
    print("==================================")
    ae_cars, loss_cars = train_model(ae_cars, car_train_loader, EPOCHS, "AE_Cars")
    
    print("\n==================================")
    print("Training Planes Autoencoder")
    print("==================================")
    ae_planes, loss_planes = train_model(ae_planes, plane_train_loader, EPOCHS, "AE_Planes")
    
    print("\n==================================")
    print("Evaluating and Generating Images")
    print("==================================")
    
    # 1. Loss curves
    plot_loss_curves(loss_cars, loss_planes)
    
    # 2. Reconstructions
    visualize_reconstruction(ae_cars, car_test_loader, "Reconstruction - Cars (Original | Reconstructed)", "cars_reconstruction.png")
    visualize_reconstruction(ae_planes, plane_test_loader, "Reconstruction - Planes (Original | Reconstructed)", "planes_reconstruction.png")
    
    # 3. Cross-Domain
    visualize_cross_domain_grid(ae_cars, ae_planes, car_test_loader, plane_test_loader)
    
    print("\nProject complete! Result saved in assets/")

if __name__ == "__main__":
    main()
