import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
DATA_DIR = "./data"
ASSETS_DIR = "./assets"

# Ensure assets dir exists
os.makedirs(ASSETS_DIR, exist_ok=True)
