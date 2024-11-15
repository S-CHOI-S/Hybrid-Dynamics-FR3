import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from algorithms.MLP import MLP, EarlyStopping
from dataset import load_data

def add_noise_to_tensor(tensor, noise_level=0.02):
    q_indices = slice(0, 7)  # q1 ~ q7
    noise = torch.normal(0, noise_level, size=tensor[:, q_indices].shape)
    tensor[:, q_indices] += noise
    return tensor

def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{save_path}/loss_plot.png')
    print("Loss plot saved as 'loss_plot.png'")
    plt.show()
    
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=1000):
    early_stopping = EarlyStopping(patience=10)
    best_model = None
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)

    return model, train_losses, val_loss

def main(model_dir, train_csv):
    X, Y = load_data(train_csv)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    # X_train = add_noise_to_tensor(X_train, noise_level=0.01)

    train_dataset = TensorDataset(X_train.clone().detach(), Y_train.clone().detach())
    val_dataset = TensorDataset(X_val.clone().detach(), Y_val.clone().detach())

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_size = 64

    model = MLP(X.shape[1], hidden_size, Y.shape[1], device).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)
    
    model_path = model_dir + "/model.pth"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    plot_loss(train_losses, val_losses, model_dir)

if __name__ == "__main__":
    model_dir = "model/mlp_qdq"
    train_csv = "data/data.csv"
    main(model_dir, train_csv)

