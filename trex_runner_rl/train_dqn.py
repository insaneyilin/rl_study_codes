import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dqn_model import DuelingDQN
from tqdm import tqdm
from trex_runner import TRexRunner

# 配置日志
logging.basicConfig(filename='./train_dqn.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# 超参数
GAMMA = 0.99
EPSILON_START = float(sys.argv[1]) if len(sys.argv) > 1 else 1
EPSILON_END = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
EPSILON_DECAY = 10000
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
BUFFER_SIZE = 2048
REMEMBER_RATE = 0.2
TARGET_UPDATE = 20
RECORD_INTERVAL = 10
EPISODE = 1000
SAVE_INTERVAL = 10  # 每隔100个episode保存一次模型
RECOVER = (sys.argv[3] == "True") if len(sys.argv) > 3 else False

MODELS_DIR = "./logs_dqn/models"
RECORD_DIR = "./logs_dqn/record"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORD_DIR, exist_ok=True)


# 经验回放缓冲区
class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train(env):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: ", device)
    input_shape = (1, 4, 175, 500)
    num_actions = 3
    hidden_dim = 512

    policy_net = DuelingDQN(input_shape, hidden_dim, num_actions).to(device)
    target_net = DuelingDQN(input_shape, hidden_dim, num_actions).to(device)

    if RECOVER:
        policy_net.load_state_dict(torch.load(f"{MODELS_DIR}/latest.pth"))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    steps_done = 0
    epsilon = EPSILON_START

    for episode in tqdm(range(EPISODE)):
        state = env.begin()

        total_reward = 0

        while True:
            steps_done += 1
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
                -1. * steps_done / EPSILON_DECAY)

            if random.random() > epsilon:
                with torch.no_grad():
                    action = policy_net(
                        torch.tensor(
                            np.expand_dims(state, 0),
                            dtype=torch.float32).to(device)).max(1)[1].view(
                                1, 1).item()
            else:
                action = random.randrange(num_actions)

            next_state, reward, done = env.step(action)
            total_reward += reward

            if done or random.random() < REMEMBER_RATE:
                replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            if done or env.over:
                break

            if len(replay_buffer) > BATCH_SIZE:
                transitions = replay_buffer.sample(BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                    *transitions)

                batch_state = torch.tensor(np.array(batch_state),
                                           dtype=torch.float32).to(device)
                batch_action = torch.tensor(
                    np.array(batch_action),
                    dtype=torch.int64).unsqueeze(1).to(device)
                batch_reward = torch.tensor(np.array(batch_reward),
                                            dtype=torch.float32).to(device)
                batch_next_state = torch.tensor(np.array(batch_next_state),
                                                dtype=torch.float32).to(device)
                batch_done = torch.tensor(np.array(batch_done),
                                          dtype=torch.float32).to(device)

                current_q_values = policy_net(batch_state).gather(
                    1, batch_action)
                next_q_values = target_net(batch_next_state).max(1)[0].detach()
                expected_q_values = batch_reward + (GAMMA * next_q_values *
                                                    (1 - batch_done))

                loss = F.mse_loss(current_q_values,
                                  expected_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        logging.info(
            f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Steps: {steps_done}"
        )

        if episode % SAVE_INTERVAL == 0:
            policy_net.save(f"{MODELS_DIR}/latest.pth")
            test_dqn(env, episode)

        if env.over:
            break


def test_dqn(env, episode_to_test):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    input_shape = (1, 4, 175, 500)
    num_actions = 3
    hidden_dim = 512

    policy_net = DuelingDQN(input_shape, hidden_dim, num_actions).to(device)
    policy_net.load_state_dict(torch.load(f"{MODELS_DIR}/latest.pth"))
    policy_net.eval()

    state = env.begin()
    total_reward = 0

    while True:
        with torch.no_grad():
            action = policy_net(
                torch.tensor(np.expand_dims(state, 0),
                             dtype=torch.float32).to(device)).max(1)[1].view(
                                 1, 1).item()

        next_state, reward, done = env.step(
            action,
            record=True,
            record_path=f"{RECORD_DIR}/test_{episode_to_test}.gif")
        total_reward += reward
        state = next_state

        if done or env.over:
            break

    logging.info(
        f"Test Episode {episode_to_test}, Total Reward: {total_reward}")

    if os.path.exists(f'{MODELS_DIR}/best_reward.log'):
        with open(f'{MODELS_DIR}/best_reward.log', 'r') as f:
            best_reward = float(f.read())
    else:
        best_reward = -1e9

    print("Total reward:", total_reward)
    if total_reward > best_reward:
        with open(f'{MODELS_DIR}/best_reward.log', 'w') as f:
            best_reward = total_reward
            print(total_reward, file=f)
            print("Best reward updated:", total_reward)
        shutil.copy(f"{MODELS_DIR}/latest.pth",
                    f"{MODELS_DIR}/best_reward.pth")
        shutil.copy(f"{RECORD_DIR}/test_{episode_to_test}.gif",
                    f"{MODELS_DIR}/best_reward.gif")


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = TRexRunner()
    train(env)
    env.close()
