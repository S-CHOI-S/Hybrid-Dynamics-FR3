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

## env
class sim_env:
    def __init__(self):
        self.k = 7
        self.dof = 9
        
        self.model_path = "../model/fr3.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.controller = controller.CController()
        self.q_init = [0, np.deg2rad(-45), 0, np.deg2rad(-135), 0, np.deg2rad(90), np.deg2rad(45), 0, 0]
        self.qdot_init = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.viewer = None
        self.rendering = True
        
    def reset(self):
        self.controller.initialize()
        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init
        self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
        self.controller.control_mujoco()
        self._torque = self.controller.write()

        for i in range(self.dof - 1):
            self.data.ctrl[i] = self._torque[i]
        mujoco.mj_step(self.model, self.data)

        if self.rendering:
            self.render()

    def step(self):
        self.controller.read(self.data.time, self.data.qpos, self.data.qvel, self.model.opt.timestep)
        self.controller.control_mujoco()
        self._torque = self.controller.write()

        for i in range(self.dof - 1):
            self.data.ctrl[i] = self._torque[i]
        mujoco.mj_step(self.model, self.data)
        if self.rendering:
            self.render()
    
    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
            self.viewer.cam.lookat = self.data.body('link0').subtree_com
            self.viewer.cam.elevation = -15
            self.viewer.cam.azimuth = 130
            # self.viewer.cam.distance = 3
        else:
            self.viewer.sync()