import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from algorithms.MLP import MLP
from dataset import load_data

def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions, axis=0)

def plot_results(true_values, predictions):
    time_steps = range(len(true_values))
    num_joints = true_values.shape[1]

    errors = np.abs(true_values - predictions)

    plt.figure(figsize=(18, 12))
    
    for i in range(num_joints):
        plt.subplot(4, 2, i + 1)
        
        plt.plot(time_steps, true_values[:, i], label=f"True tau{i+1}", color='blue')
        plt.plot(time_steps, predictions[:, i], label=f"Pred tau{i+1}", color='red', linestyle='dashed')
        
        plt.plot(time_steps, errors[:, i], label=f"Error tau{i+1}", color='green', linestyle=':')
        
        plt.xlabel("Time Step")
        plt.ylabel(f"Torque tau{i+1}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(true_values, predictions):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    r2 = r2_score(true_values, predictions) # Coefficient of Determination
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mae, mse, r2

def main(model_path, test_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_test, Y_test = load_data(test_csv)
    
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    hidden_size = 64
    model = MLP(X_test.shape[1], hidden_size, Y_test.shape[1], device).to(device)
    model.load_state_dict(torch.load(model_path + "/model.pth")) # GPU
    # model.load_state_dict(torch.load(model_path + "/model.pth", map_location=torch.device('cpu'))) # CPU
    model.eval()
    print(model)

    predictions = predict(model, test_loader, device)
    
    print("\nEvaluation Metrics:")
    calculate_metrics(Y_test.numpy(), predictions)

    plot_results(Y_test.numpy(), predictions)

if __name__ == "__main__":
    model_path = "model/mlp_qdq"
    for i in range(10):
        test_csv = f"data/data_test{i+1}.csv"
        main(model_path, test_csv)
