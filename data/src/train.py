# File: data/src/train.py
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append('data/src')
from model import FloodCNNLSTM
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load data
print('Loading data...')
X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')
print(f'X: {X.shape}, y: {y.shape}')

# Fix size mismatch
min_len = min(len(X), len(y))
X, y = X[:min_len], y[:min_len]

# Replace NaN and Inf values
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_val   = torch.FloatTensor(X_val)
y_train = torch.FloatTensor(y_train)
y_val   = torch.FloatTensor(y_val)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=512)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = FloodCNNLSTM(n_features=21, seq_length=24).to(device)

# Class weights to handle imbalance (very few flood events)
flood_ratio = (y == 0).sum() / (y == 1).sum()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

os.makedirs('models', exist_ok=True)
best_val_loss = float('inf')

print('Training...')
for epoch in range(15):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item()
            correct += ((pred > 0.5) == yb).sum().item()

    acc = correct / len(y_val) * 100
    print(f'Epoch {epoch+1:2d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Acc: {acc:.2f}%')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/flood_model.pth')
        print(f'           -> Model saved!')

    scheduler.step()

print('Training complete! Model saved to models/flood_model.pth')