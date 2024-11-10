import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from algorithms.MLP import MLP
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_test_data(filename):
    """CSV 파일에서 테스트 데이터를 로드합니다."""
    df = pd.read_csv(filename)
    
    df = df[df["done"] != 1]
    
    # 입력 데이터 (X)와 실제 출력 데이터 (Y)도 함께 로드합니다 (ground truth 포함)
    X = df[["q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
            "ddq1", "ddq2", "ddq3", "ddq4", "ddq5", "ddq6", "ddq7",
            "des_tau1", "des_tau2", "des_tau3", "des_tau4", "des_tau5", "des_tau6", "des_tau7",
            "des_EE_x", "des_EE_y", "des_EE_z", "des_EE_roll", "des_EE_pitch", "des_EE_yaw",
            "EE_x", "EE_y", "EE_z", "EE_roll", "EE_pitch", "EE_yaw"]].values

    Y = df[["tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def predict(model, data_loader, device):
    """모델을 사용하여 예측을 수행합니다."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)

def plot_results(true_values, predictions):
    """예측 결과와 오차를 시각화합니다."""
    time_steps = range(len(true_values))
    num_joints = true_values.shape[1]

    # 예측 값과 실제 값 간의 절대 오차 계산
    errors = np.abs(true_values - predictions)

    plt.figure(figsize=(18, 12))
    
    for i in range(num_joints):
        plt.subplot(4, 2, i + 1)
        
        # 실제 값과 예측 값 그래프
        plt.plot(time_steps, true_values[:, i], label=f"True tau{i+1}", color='blue')
        plt.plot(time_steps, predictions[:, i], label=f"Pred tau{i+1}", color='red', linestyle='dashed')
        
        # 오차 그래프 추가
        plt.plot(time_steps, errors[:, i], label=f"Error tau{i+1}", color='green', linestyle=':')
        
        plt.xlabel("Time Step")
        plt.ylabel(f"Torque tau{i+1}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(true_values, predictions):
    """평가 지표를 계산하고 출력합니다."""
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions) # Coefficient of Determination
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mae, mse, r2

def main(model_path, test_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 테스트 데이터를 로드
    X_test, Y_test = load_test_data(test_csv)
    
    # DataLoader 생성
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 모델 로드
    input_size = 40
    hidden_size = 64
    output_size = 7
    model = MLP(input_size, hidden_size, output_size, device).to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 예측 수행
    predictions = predict(model, test_loader, device)
    
    # 평가 지표 계산 및 출력
    print("\nEvaluation Metrics:")
    calculate_metrics(Y_test.numpy(), predictions)

    # 예측 결과와 실제 값 시각화
    plot_results(Y_test.numpy(), predictions)

if __name__ == "__main__":
    model_path = "model/mlp_model_0.02.pth"  # 학습된 모델 경로
    test_csv = "data/data_test2.csv"  # 테스트 데이터 경로
    main(model_path, test_csv)
