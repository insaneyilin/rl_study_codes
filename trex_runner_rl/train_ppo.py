import argparse
import logging
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import torch
from ppo_agent import PPOAgent
from tqdm import tqdm
from trex_runner import TRexRunner

logging.basicConfig(filename='./train_ppo.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

MODELS_DIR = "./logs_ppo/models"
RECORD_DIR = "./logs_ppo/record"


def train_ppo(env: TRexRunner, agent, num_episodes, record_prefix,
              models_prefix, resume, id):
    return_list = []
    if resume:
        agent.load(models_prefix)
    steps_done = 0
    with tqdm(total=num_episodes) as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            record = True
            record_path = os.path.join(record_prefix, f'{id}-{i_episode}.gif')
            state = env.begin()
            done = False
            while not done:
                steps_done += 1
                action = agent.take_action(state)
                next_state, reward, done = env.step(action,
                                                    record=record,
                                                    record_path=record_path)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state = next_state
                episode_return += reward
            ratio = agent.update(transition_dict)
            return_list.append((episode_return, ratio))
            pbar.set_postfix({
                'episode': f'{i_episode}',
                'reward': f'{episode_return}',
                'ratio': f'{ratio}'
            })
            pbar.update(1)
            logging.info(
                f"Episode {i_episode}, Total Reward: {episode_return}, Steps: {steps_done}"
            )
            if record:
                agent.save(models_prefix)
    return return_list


if __name__ == '__main__':
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    actor_lr = 1e-4
    critic_lr = 1e-4
    mini_batch = 32
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 4
    eps = 0.2
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp_name',
                           type=str,
                           required=True,
                           help='experiment name')
    argparser.add_argument('--recover',
                           action='store_true',
                           help='recover from the latest model')
    args = argparser.parse_args()

    env = TRexRunner()
    input_shape = env.observation_shape()
    num_actions = env.num_actions()
    agent = PPOAgent(input_shape, hidden_dim, num_actions, actor_lr, critic_lr,
                     mini_batch, lmbda, epochs, eps, gamma, device)

    pathlib.Path(RECORD_DIR).mkdir(exist_ok=True)
    pathlib.Path(MODELS_DIR).mkdir(exist_ok=True)

    return_list = train_ppo(env, agent, num_episodes, RECORD_DIR, MODELS_DIR,
                            args.recover, args.exp_name)
    rewards, ratio = zip(*return_list)

    plt.plot(rewards)
    plt.savefig(f'{RECORD_DIR}/{args.exp_name}-reward.png')
    plt.close()

    plt.plot(ratio)
    plt.savefig(f'{RECORD_DIR}/{args.exp_name}-ratio.png')
    plt.close()

    env.close()
