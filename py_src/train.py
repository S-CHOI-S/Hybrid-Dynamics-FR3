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

# CSV 파일에서 데이터 로드
def load_data(filename):
    df = pd.read_csv(filename)
    
    df = df[df["done"] != 1]
    
    # 입력 데이터 (X)와 출력 데이터 (Y) 분리
    X = df[["q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
            "des_EE_x", "des_EE_y", "des_EE_z", "des_EE_roll", "des_EE_pitch", "des_EE_yaw"]].values
    Y = df[["tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]].values

    return X, Y

def add_noise_to_tensor(tensor, noise_level=0.02):
    """훈련 데이터에만 노이즈를 추가합니다."""
    q_indices = slice(0, 7)  # q1 ~ q7에 해당하는 인덱스 범위
    noise = torch.normal(0, noise_level, size=tensor[:, q_indices].shape)
    tensor[:, q_indices] += noise
    return tensor

def plot_loss(train_losses, val_losses):
    """훈련 및 검증 손실을 시각화하고 저장합니다."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 그래프를 이미지 파일로 저장
    plt.savefig('loss_plot.png')
    print("Loss plot saved as 'loss_plot.png'")
    plt.show()

# 데이터를 로드하고 준비
X, Y = load_data("data/data.csv")

# 데이터를 텐서로 변환
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# 훈련 및 검증 데이터 분할
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = add_noise_to_tensor(X_train, noise_level=0.01)

train_dataset = TensorDataset(X_train.clone().detach(), Y_train.clone().detach())
val_dataset = TensorDataset(X_val.clone().detach(), Y_val.clone().detach())

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 7 + 7 + 6
hidden_size = 64
output_size = 7

# 모델 초기화
model = MLP(input_size, hidden_size, output_size, device).to(device)

# 손실 함수 및 최적화 기법 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=500):
    early_stopping = EarlyStopping(patience=10)
    best_model = None
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # 훈련 단계
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
        
        # 검증 단계
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

        # Early Stopping 체크
        if early_stopping.step(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

        # 최적의 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # 최적의 모델 로드
    model.load_state_dict(best_model)

    # 학습 과정을 시각화하는 함수 호출
    plot_loss(train_losses, val_losses)

    return model

# 모델 학습
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

# 모델 저장
model_path = "model/mlp_model_0.02_wo_des_torque.pth"
torch.save(trained_model.state_dict(), model_path)
print(f"Model saved to {model_path}")
