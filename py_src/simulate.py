import torch
import fr3Env
import numpy as np
import pandas as pd

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = fr3Env.sim_env()
    env.rendering = False
    
    env.reset()
    while True:
        obs, reward, done, _ = env.step()
        if done:
            env.reset()
            if env.episode_num == 1e6:
                break

    env.data_collector.save_to_csv()

if __name__ == "__main__":
    main()