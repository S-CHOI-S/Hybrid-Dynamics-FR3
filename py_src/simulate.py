import torch
import fr3Env

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
    
    env.reset()
    while True:
        env.step()

if __name__ == "__main__":
    main()