import torch
import fr3Env
import numpy as np

''' Only for Rendering '''
import mujoco
import mujoco.viewer


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
    
    while True:
        env.reset()
    
    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    #     while viewer.is_running():
    #         # step_start = time.time()
            
    #         viewer.sync()
    

if __name__ == "__main__":
    main()

''' Only for Rendering '''
# env = fr3Env.cabinet_env()
# env.env_rand = True

# # observation = env.reset(RL)

# with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#     while viewer.is_running():
#         # step_start = time.time()
        
#         viewer.sync()
        
        
