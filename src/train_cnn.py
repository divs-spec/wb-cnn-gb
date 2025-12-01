# src/train_cnn.py

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from data_utils import WhiteBalanceImageDataset, image_transforms
from model import ResNet18Regressor

# Paths (to be adjusted as needed)
TRAIN_IMG_DIR = 'data/Train'
TRAIN_CSV = 'data/Train/sliders.csv'
MODEL_PATH = 'cnn_model.pth'

# Hyperparameters
batch_size = 16
epochs = 10
learning_rate = 1e-4

# Prepare dataset and loader
train_dataset = WhiteBalanceImageDataset(TRAIN_IMG_DIR, TRAIN_CSV, transform=image_transforms, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18Regressor(pretrained=True).to(device)

# Loss and optimizer
criterion = nn.L1Loss()  # MAE
optimizer = Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        preds = model(images)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * images.size(0)
    avg_loss = epoch_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Training MAE: {avg_loss:.4f}")

# Save trained model weights
torch.save(model.state_dict(), MODEL_PATH)
