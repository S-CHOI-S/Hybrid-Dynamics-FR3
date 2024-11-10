#!/usr/bin/env python3
import sys
import time

import pandas as pd

import numpy as np
from numpy.linalg import inv
from build import controller

import mujoco
import gym
from gym import spaces
from random import random, randint, uniform
from scipy.spatial.transform import Rotation as R
from mujoco import viewer
from time import sleep
import tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

## data
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

## env
class sim_env:
    def __init__(self):
        super(sim_env, self).__init__()
        self.k = 7
        self.dof = 9
        
        self.model_path = "../model/fr3.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.controller = controller.CController()
        self.q_init = [0, np.deg2rad(-45), 0, np.deg2rad(-135), 0, np.deg2rad(90), np.deg2rad(45), 0, 0]
        self.qdot_init = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.q_range = self.model.jnt_range[:self.k]
        self.q_soft_range = 0.9 * self.q_range
        self.qdot_range = np.array([[-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750],
                                    [-2.61, 2.61], [-2.61, 2.61], [-2.61, 2.61]])
        self.tau_range = np.array([[-87, 87], [-87, 87], [-87, 87], [-87, 87], [-87, 87], [-12, 12], [-12, 12], [-100, 100]])
        self.random_sampled_EE = [0.30689056659294117, -4.480957754469757e-12, 0.6972820523028392, 3.1415926535848966, -4.8965455623439355e-12, -0.7853981634043731]

        self.viewer = None
        self.rendering = True

        self.episode_num = 0
        self.first_init = True
        self.data_collector = DataCollector()
        
    def reset(self):
        print(f"Episode Num: {self.episode_num}")
        self.start_time = self.data.time
        self.episode_time = 100
        self.done_int = 0

        self.controller.initialize()
        
        # if self.first_init:
        #     self.data.qpos = self.q_init
        #     self.first_init = False
        # else:
        #     self.data.qpos = self.sampling_joint_position()
        # self.controller.write_qpos_init(self.data.qpos)
        
        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init
        
        self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
        self.controller.control_mujoco()
        self._torque = self.controller.write()

        for i in range(self.dof - 1):
            self.data.ctrl[i] = self._torque[i]
        mujoco.mj_step(self.model, self.data)

        self.bound_done = False
        self.time_done = False
        self.torque_done = False

        if self.rendering:
            self.render()

    def step(self, action=None):
        if self.controller.count_plan() > 0 and self.controller.check_trajectory_complete():
            self.random_sampled_EE = self.sampling_EE()
            self.controller.write_random_sampled_EE(self.random_sampled_EE)
        self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
        self.controller.control_mujoco()
        self._torque = self.controller.write()
        
        for i in range(self.dof - 1):
            self.data.ctrl[i] = self._torque[i]
        mujoco.mj_step(self.model, self.data)

        if self.rendering:
            self.render()

        obs = self.observation()
        reward = self.reward(action)
        done = self.done()
        info = self.info()

        if int(self.data.time * 100) % 10 == 0:
            self.collect_data()

        return obs, reward, done, info
    
    def observation(self):
        self.obs_q = self.data.qpos[0:self.k]
        self.obs_dq = self.data.qvel[0:self.k]

        obs = np.concatenate((self.obs_q, self.obs_dq), axis=0)

        return obs
    
    def reward(self, action=None):
        reward = 1

        return reward

    def done(self):
        # bound done
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)
        self.bound_done = -1 in self.q_operation_list

        # time done
        self.time_done = self.data.time - self.start_time >= self.episode_time

        # high torque
        tau_threshold_lower = self.tau_range[:, 0] * 0.7
        tau_threshold_upper = self.tau_range[:, 1] * 0.7

        if np.any((self._torque < tau_threshold_lower) | (self._torque > tau_threshold_upper)):
            self.torque_done = True
        
        if self.bound_done or self.torque_done:
            self.done_int = 1
            self.collect_data()
            self.data_collector.save_to_csv(filename="data_test3.csv", append=True)
            self.episode_num += 1
            return True
        else:
            self.done_int = 0
            return False
        
    def info(self):
        info = 'sim_env'
        return info
    
    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
            self.viewer.cam.lookat = self.data.body('link0').subtree_com
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 130
            # self.viewer.cam.distance = 3
        else:
            self.viewer.sync()

    def sampling_joint_position(self):
        random_joint_values = np.random.uniform(self.q_soft_range[:, 0], self.q_soft_range[:, 1])
        random_joint_values = np.append(random_joint_values, [0, 0])
        return random_joint_values

    def sampling_EE(self):
        current_EE = self.controller.get_EE()
        noise_first_half = np.random.uniform(-0.15, 0.15, 3)
        noise_second_half = np.random.uniform(-0.5, 0.5, 3)
        noise = np.concatenate((noise_first_half, noise_second_half))
        updated_values = [current_EE[i] + noise[i] for i in range(6)]
        return updated_values
    
    def collect_data(self):
        q = self.data.qpos[:self.k]  # 조인트 위치
        dq = self.data.qvel[:self.k]  # 조인트 속도
        ddq = self.data.qacc[:self.k]  # 조인트 가속도
        command_tau = self._torque[:self.k] # 조인트 토크
        desired_EE = self.random_sampled_EE
        EE = self.controller.get_EE()  # End-Effector 위치
        tau = self.data.qfrc_actuator[:self.k]  # 각 조인트의 토크 값(state)

        data_to_collect = [self.done_int] + list(q) + list(dq) + list(ddq) + list(command_tau) + list(desired_EE) + list(EE) + list(tau)

        if np.any(desired_EE) != 0:
            self.data_collector.collect(data_to_collect)
        
        # if self.done_int == 1:
        #     print("done: 1")