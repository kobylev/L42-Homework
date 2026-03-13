import torch
import torch.nn as nn
from tqdm import tqdm
from config import DEVICE, LEARNING_RATE

def train_model(model, train_loader, epochs, model_name="Autoencoder"):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    epoch_losses = []
    print(f"--- Training {model_name} ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in pbar:
            images, _ = batch
            images = images.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images) # Autoencoder trying to reconstruct its input
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

    return model, epoch_losses
