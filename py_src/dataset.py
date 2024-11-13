import torch
import pandas as pd

def load_data(filename):
    df = pd.read_csv(filename)
    
    df = df[df["done"] != 1]
    
    X = df[["q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
        ]].values
    
    Y = df[["tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)