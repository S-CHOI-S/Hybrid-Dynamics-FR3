import torch
import fr3Env
import numpy as np
import pandas as pd

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = fr3Env.rl_env()
    check_env(env)
    env.rendering = True
    
    model = PPO('MultiInputPolicy', env, verbose=1)
    
    model.learn(total_timesteps=1e6)
    model.save("hybridynamics")
    
    # env.reset()
    # while True:
    #     obs, reward, done, _ = env.step()
    #     if done:
    #         env.reset()
    #         if env.episode_num == 1e6:
    #             break

if __name__ == "__main__":
    main()