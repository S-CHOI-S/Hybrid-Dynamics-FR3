import numpy as np
import torch
import gym
import argparse
import json
import os
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import fr3Env
import tools
import torch.nn as nn
import torch.nn.functional as F

''' Only for Rendering '''
import mujoco
import mujoco.viewer

''' for Tensorboard '''
import time
import psutil
import datetime
import subprocess
# import torch
import torchvision
from tensorboard import program
import webbrowser
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib.image as img
import random


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

HEURISTIC = 0
RL = 1

MAX_TIMESTEPS = 1e6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(PATH, TRAIN, OFFLINE, RENDERING):
    env = fr3Env.dynamics_env()
    env.rendering = RENDERING

    if TRAIN:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        writer = SummaryWriter(f"../runs/hybridynamics/{current_time}")
        print(f"{YELLOW}[TENSORBOARD]{RESET} The data will be saved in {YELLOW}../runs/hybridynamics/{current_time}{RESET} directory!")

        # tb = program.TensorBoard()
        # tb.configure(argv=[None, '--logdir', f"../runs/hybridynamics/{current_time}", '--port', '7100'])
        # url = tb.launch()
        # webbrowser.open_new(url)

        while True:
            # print("before reset")

            state = env.reset()

            for t in range(int(MAX_TIMESTEPS)):
                action = [7.2965e-18, 17.0942, -0.788736, 7.66635, 0.316923, 2.61568, -0.0276822]
                env._command_EE()

                # action = [random.uniform(0, 5) for _ in range(7)]

                # action = env.select_action(state)
                state, reward, done, _ = env.step(action)

                if done == True:
                    env.reset()

            # print("after reset")

        # pretrained_model_dir1 = pretrained_model_dir + "/rotation/"
        # pretrained_model_dir2 = pretrained_model_dir + "/force/"
        # trainer1.load(pretrained_model_dir1)
        # trainer2.load(pretrained_model_dir2)
        # continue_train = True

        # actor1.train()
        # actor2.train()
        
        # force_gains = []
        
        # for t in range(int(max_timesteps)):

        #     action_rotation = actor1.select_action(state)
        #     action_force = actor2.select_action(state)
        #     # print(f"{CYAN}c/action_rotation: {RESET}", action_rotation)
        #     # print(f"{CYAN}c/action_force:    {RESET}", action_force)
        #     # print(f"{CYAN}c/time_step:       {RESET}", t)

        #     # action_rotation = np.array([0, 0])
        #     # if t < 80:
        #     #     action_force = 0.1
        #     # else:
        #     #     action_force = 0
                
        #     next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
        #     # print(f"{CYAN}c/env.step-----    {RESET}")
        #     # print(f"{MAGENTA}c/reward_rotation:         {RESET}", reward_rotation)
        #     # print(f"{MAGENTA}c/reward_force:            {RESET}", reward_force)

        #     episode_timesteps += 1

        #     replay_buffer1.add(state, action_rotation, next_state, reward_rotation, done)
        #     replay_buffer2.add(state, action_force, next_state, reward_force, done)

        #     state = next_state
        #     episode_return_rotation += reward_rotation
        #     episode_return_force += reward_force

        #     episode_return_rotation_tb = torch.tensor(episode_return_rotation, dtype=torch.float32)
        #     episode_return_force_tb = torch.tensor(episode_return_force, dtype=torch.float32)
        #     # episode_action_force = torch.tensor(action_force, dtype=torch.float32)
        #     # episode_action_rotation = torch.tensor(action_rotation, dtype=torch.float32)
        #     force_gain_tb = torch.tensor(env.force_gain, dtype=torch.float32)
        #     force_gains.append(force_gain_tb.item())
            
        #     # print(f"{RESET}force_gain:              {RESET}", force_gain_tb)
            
        #     # writer.add_scalar('Action/Rotation', episode_action_rotation, t+1)
        #     # writer.add_scalar('Action/Force', force_gain_tb, t+1)
        #     writer.add_scalar('Action/Force', force_gain_tb, episode_timesteps + 1)

        #     # Train agent after collecting sufficient data
        #     if t >= batch_size:
        #         if IsDemo:
        #             trainer1.train_with_demo(replay_buffer1, replay_buffer1_expert, batch_size)
        #             trainer2.train_with_demo(replay_buffer2, replay_buffer2_expert, batch_size)

        #         else:
        #             if env.data.qpos[9] > 0.5:
        #                 print(f"{YELLOW}cabinet angle: {RESET}", env.data.qpos[9])
        #                 path1 = models_dir + "bestbestbests/rotation/"
        #                 path2 = models_dir + "bestbestbests/force/"
        #                 os.makedirs(path1, exist_ok=True)
        #                 trainer1.save(path1)
        #                 os.makedirs(path2, exist_ok=True)
        #                 trainer2.save(path2)
                        
        #             trainer1.train(replay_buffer1, batch_size)
        #             trainer2.train(replay_buffer2, batch_size)
            
        #     if (t + 1) % save_freq == 0:
        #         save_flag = True
                
        #     if done:
        #         # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        #         if continue_train == True:
        #             print(
        #                 f"Total T: {t + 1} Episode Num: {num_dir*100 + episode_num + 1} Episode T: {episode_timesteps} "
        #                 f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
        #         else:  
        #             print(
        #                 f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
        #                 f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
                
        #         if env.data.qpos[9] > 0.5:
        #             print(f"{YELLOW}cabinet angle: {RESET}", env.data.qpos[9])
        #             path1 = models_dir + "bestbestbest/rotation/"
        #             path2 = models_dir + "bestbestbest/force/"
        #             os.makedirs(path1, exist_ok=True)
        #             trainer1.save(path1)
        #             os.makedirs(path2, exist_ok=True)
        #             trainer2.save(path2)
                
        #         # trainer1.train(replay_buffer1, batch_size)
        #         # trainer2.train(replay_buffer2, batch_size)
                
        #         # save_manual = int(input("save policy? 0: No, 1: Yes"))
        #         # if save_manual == 1:
        #         #     path1 = models_dir + "manual" + "/rotation/"
        #         #     path2 = models_dir + "manual" + "/force/"
        #         #     os.makedirs(path1, exist_ok=True)
        #         #     os.makedirs(path2, exist_ok=True)
        #         #     if not os.path.exists(path1):
        #         #         os.makedirs(path1)
        #         #     if not os.path.exists(path2):
        #         #         os.makedirs(path2)
        #         #     trainer1.save(path1)
        #         #     trainer2.save(path2)
                    
        #         # print(f"{YELLOW}cabinet angle: {RESET}", env.data.qpos[9])
                
                
        #         # TensorBoard에 로깅
        #         writer.add_scalar('Reward/Rotation', episode_return_rotation_tb, episode_num + 1)
        #         writer.add_scalar('Reward/Force', episode_return_force_tb, episode_num + 1)

        #         # if t > save_freq and episode_return_rotation > return_rotation_max and episode_return_force > return_force_max:
        #         #     return_rotation_max = episode_return_rotation
        #         #     return_force_max = episode_return_force
        #         #     path1 = models_dir + "best/rotation/"
        #         #     path2 = models_dir + "best/force/"
        #         #     trainer1.save(path1)
        #         #     trainer2.save(path2)
                
        #         # Save best models
        #         if episode_return_rotation > best_episode_return_rotation:
        #             best_episode_return_rotation = episode_return_rotation
        #             path1 = models_dir + "best/rotation/"
        #             os.makedirs(path1, exist_ok=True)
        #             trainer1.save(path1)
                
        #         if episode_return_force > best_episode_return_force:
        #             best_episode_return_force = episode_return_force
        #             path2 = models_dir + "best/force/"
        #             os.makedirs(path2, exist_ok=True)
        #             trainer2.save(path2)
                    
        #         if (episode_return_rotation > bbest_episode_return_rotation) & (episode_return_force > bbest_episode_return_force):
        #             bbest_episode_return_rotation = episode_return_rotation
        #             bbest_episode_return_force = episode_return_force
        #             path1 = models_dir + "bestbest/rotation/"
        #             path2 = models_dir + "bestbest/force/"
        #             os.makedirs(path1, exist_ok=True)
        #             trainer1.save(path1)
        #             os.makedirs(path2, exist_ok=True)
        #             trainer2.save(path2)
                    
        #         # if env.data.qpos[9] > 0.42:
        #         #     path1 = models_dir + "bestbestbest/rotation/"
        #         #     path2 = models_dir + "bestbestbest/force/"
        #         #     os.makedirs(path1, exist_ok=True)
        #         #     trainer1.save(path1)
        #         #     os.makedirs(path2, exist_ok=True)
        #         #     trainer2.save(path2)
                    
        #         # Reset environment
        #         state = env.reset(PLANNING_MODE)
        #         episode_data.append([episode_num, episode_timesteps, episode_return_rotation, episode_return_force])
                
        #         episode_return_rotation_accum += episode_return_rotation
        #         episode_return_force_accum += episode_return_force
        #         episode_cnt += 1
                
        #         episode_return_rotation = 0
        #         episode_return_force = 0
        #         episode_timesteps = 0
        #         episode_num += 1
                
        #         force_gain_tb = 0
        #         force_gains = []
            
        #     if (t + 1) % 10000 == 0:
        #         timestep_data.append(
        #             [episode_return_rotation_accum / episode_cnt, episode_return_force_accum / episode_cnt])
        #         episode_return_rotation_accum = 0
        #         episode_return_force_accum = 0
        #         episode_cnt = 0
        #         np.save(models_dir + "avg_reward.npy", timestep_data)
            
        #     if save_flag:
        #         if continue_train == True:
        #             path1 = models_dir + str((env.episode_number + 1) // save_freq + num_dir) + "/rotation/"
        #             path2 = models_dir + str((env.episode_number + 1) // save_freq + num_dir) + "/force/"
        #         else:
        #             path1 = models_dir + str((env.episode_number + 1) // save_freq) + "/rotation/"
        #             path2 = models_dir + str((env.episode_number + 1) // save_freq) + "/force/"
        #         os.makedirs(path1, exist_ok=True)
        #         os.makedirs(path2, exist_ok=True)
        #         if not os.path.exists(path1):
        #             os.makedirs(path1)
        #         if not os.path.exists(path2):
        #             os.makedirs(path2)
        #         trainer1.save(path1)
        #         trainer2.save(path2)

        #         np.save(models_dir + "reward.npy", episode_data)
        #         # replay_buffer1.save(models_dir + 'replay_buffer_rotation.pkl')
        #         # replay_buffer2.save(models_dir + 'replay_buffer_force.pkl')
        #         save_flag = False
                
            
        # writer.close() # TensorBoard writer 종료
        
            # if i % 100 == 99:
            #     print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            #     writer.add_scalar('training loss', running_loss / 100, epoch * len(trainloader) + i)
            #     running_loss = 0.0

    # else:
    #     pretrained_model_dir1 = pretrained_model_dir + "/rotation/"
    #     pretrained_model_dir2 = pretrained_model_dir + "/force/"
    #     trainer1.load(pretrained_model_dir1)
    #     actor1.eval()
    #     critic1.eval()
    #     actor1.training = False

    #     trainer2.load(pretrained_model_dir2)
    #     actor2.eval()
    #     critic2.eval()
    #     actor2.training = False

    #     num_ep = 16
    #     force_data = []
        
    #     # plt2.imshow(image)
    #     # plt3.imshow(image)
    #     # plt4.imshow(image)
    #     # plt5.imshow(image)
    #     # env.episode_number = 3
    #     # df = pd.read_csv("/home/kist-robot2/Downloads/obs_real.csv")
    #     # states = df.to_numpy(dtype=np.float32)
        
    #     for _ in range(num_ep):
    #         state = env.reset(PLANNING_MODE)
    #         done = False
    #         step_cnt = 0
    #         episode_return_rotation = 0
    #         episode_return_force = 0
    #         # chkpt = 1
    #         manipulability_data = []

    #         while not done:

    #             step_cnt += 1
    #             action_rotation = actor1.select_action(state)
    #             action_force = actor2.select_action(state)

    #             next_state, reward_rotation, reward_force, done, _ = env.step(action_rotation, action_force)
    #             force_data.append(env.force)

    #             state = next_state
    #             episode_return_rotation += reward_rotation
    #             episode_return_force += reward_force
                
    #             Jac = env.controller.get_jacobian()
    #             manipulability = tools.calc_manipulability(np.array(Jac))
    #             manipulability_data.append([env.data.qpos[9], manipulability])
                
    #             # plt.imshow(image)
                
    #             # if env.cabinet1_angle >= (0.1 * chkpt):
    #             #     print(env.cabinet1_angle)
                    
    #             #     # if chkpt == 1:
    #             #     #     plt.show()
    #             #     # elif chkpt == 2:
    #             #     #     plt.show()
    #             #     # elif chkpt == 3:
    #             #     #     plt.show()
    #             #     # elif chkpt == 4:
    #             #     #     plt.show()
    #             #     # elif chkpt == 5:
    #             #     #     plt.show()
                    
    #             #     chkpt += 1

    #         # np.save("./data/torque_hybrid.npy", env.torque_data)
    #         # print(
    #         #     f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} "
    #         #     f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
    #         print(
    #             f"Reward R: {episode_return_rotation:.3f} Reward F: {episode_return_force:.3f}")
    #         print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
    #               "  goal:", env.goal_done)
    #         print(env.cabinet1_angle)
    #         print(env.obs_omega[0])
            
    #         # np.save("OURS_MANIPULABILITY", manipulability_data)

    #         fig, axs = plt.subplots(3, 2, figsize=(8, 6))
    #         axs[0, 0].plot([sublist[0] for sublist in env.command_data])
    #         axs[0, 0].set_title("droll", pad=20)

    #         axs[0, 1].plot([sublist[1] for sublist in env.command_data])
    #         axs[0, 1].set_title("dpitch", pad=20)

    #         axs[1, 0].plot([sublist[2] for sublist in env.command_data])
    #         axs[1, 0].set_title("roll", pad=20)

    #         axs[1, 1].plot([sublist[3] for sublist in env.command_data])
    #         axs[1, 1].set_title("pitch", pad=20)

    #         axs[2, 0].plot([sublist[4] for sublist in env.command_data])
    #         axs[2, 0].set_title("force gain", pad=20)

    #         axs[2, 1].plot([sublist[5] for sublist in env.command_data])
    #         axs[2, 1].set_title("R force gain", pad=20)
    #         # plt.plot(force_data,  linestyle='-', color='b')
    #         # # plt.title(env.friction)
    #         plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="wo expert demo, first door training, ")
    parser.add_argument("--path", help="data load path", default=" ./log/0729_rerereward1/bestbest")
    parser.add_argument("--train", help="0->test,  1->train", type=int, default=1)
    parser.add_argument("--render", help="0->no rendering,  1->rendering", type=int, default=1)
    parser.add_argument("--offline", help="0->no offline data,  1->with offline data", type=int, default=0)
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict['description'] = parser.description
    if args.train == 1:
        os.makedirs(args.path, exist_ok=True)
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        with open(args.path+'args_and_notes.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")

    main(PATH=args.path, TRAIN=args.train, OFFLINE=args.offline, RENDERING=args.render)


''' Only for Rendering '''
# env = fr3Env.cabinet_env()
# env.env_rand = True

# # observation = env.reset(RL)

# with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#     while viewer.is_running():
#         # step_start = time.time()
        
#         viewer.sync()
        
        
