import torch
import pandas as pd
import numpy as np

joint_ranges = {
    "q1": [-1.7837,  1.7837],
    "q2": [-2.9007,  2.9007],
    "q3": [-3.0421, -0.1518],
    "q4": [-2.8065,  2.8065],
    "q5": [ 0.5445,  4.5169],
    "q6": [-3.0159,  3.0159],
    "q7": [-3.0159,  3.0159]
}

torque_ranges = {
    "tau1": [-87, 87],
    "tau2": [-87, 87],
    "tau3": [-87, 87],
    "tau4": [-87, 87],
    "tau5": [-87, 87],
    "tau6": [-12, 12],
    "tau7": [-12, 12]
}

def normalize_position(q, joint_name):
    min_val, max_val = joint_ranges[joint_name]
    return (q - min_val) / (max_val - min_val) * 2 - 1  # [-1, 1] 범위로 변환

def normalize_velocity(dq):
    return (dq - (-2.1750)) / (2.1750 - (-2.1750)) * 2 - 1

def normalize_torque(tau, joint_name):
    min_val, max_val = torque_ranges[joint_name]
    min_val *= 0.7
    max_val *= 0.7
    return (tau - min_val) / (max_val - min_val) * 2 - 1

def unnormalize_torque(norm_tau, joint_name):
    min_val, max_val = torque_ranges[joint_name]
    min_val /= 0.7
    max_val /= 0.7
    return (norm_tau + 1) / 2 * (max_val - min_val) + min_val

def load_data(filename):
    df = pd.read_csv(filename)
    
    df = df[df["done"] != 1]
    
    for i in range(1, 8):
        joint_name = f"q{i}"
        joint_velocity_name = f"dq{i}"
        torque_name = f"tau{i}"
        
        df[joint_name] = df[joint_name].apply(lambda q: normalize_position(q, joint_name))
        # df[joint_velocity_name] = df[joint_velocity_name].apply(lambda dq: normalize_velocity(dq))
        # df[torque_name] = df[torque_name].apply(lambda tau: normalize_torque(tau, torque_name))
    
    X = df[["q1", "q2", "q3", "q4", "q5", "q6", "q7",
            "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
        ]].values
    
    Y = df[["EE_x", "EE_y", "EE_z", "EE_roll", "EE_pitch", "EE_yaw",
            "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"]].values

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

class DataCollector:
    def __init__(self):
        # 데이터 저장을 위한 리스트
        self.data = []

    def collect(self, values):
        # 데이터를 수집
        self.data.append(values)

    def save_to_csv(self, filename="data.csv", append=False):
        # q, dq, ddq, command_tau, desired_EE, EE, tau
        df = pd.DataFrame(self.data, columns=["done",
                                              "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                                              "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                                              "ddq1", "ddq2", "ddq3", "ddq4", "ddq5", "ddq6", "ddq7",
                                              "des_tau1", "des_tau2", "des_tau3", "des_tau4", "des_tau5", "des_tau6", "des_tau7",
                                              "des_EE_x", "des_EE_y", "des_EE_z", "des_EE_roll", "des_EE_pitch", "des_EE_yaw",
                                              "EE_x", "EE_y", "EE_z", "EE_roll", "EE_pitch", "EE_yaw",
                                              "tau1", "tau2", "tau3", "tau4", "tau5", "tau6", "tau7"])
        if append:
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
        print("Data saved to", filename)

        self.data = []