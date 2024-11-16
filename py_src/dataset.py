import torch
import pandas as pd
import numpy as np

joint_ranges = {
    "q1": [-1.7837, 1.7837],
    "q2": [-2.9007, 2.9007],
    "q3": [-3.0421, -0.1518],
    "q4": [-2.8065, 2.8065],
    "q5": [0.5445, 4.5169],
    "q6": [-3.0159, 3.0159],
    "q7": [-3.0159, 3.0159]
}

def normalize_position(q, joint_name):
    min_val, max_val = joint_ranges[joint_name]
    return (q - min_val) / (max_val - min_val) * 2 - 1  # [-1, 1] 범위로 변환

def load_data(filename):
    df = pd.read_csv(filename)
    
    df = df[df["done"] != 1]
    
    for i in range(1, 8):
        joint_name = f"q{i}"
        df[joint_name] = df[joint_name].apply(lambda q: normalize_position(q, joint_name))
    
    X = df[["q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
        ]].values
    
    Y = df[["tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)