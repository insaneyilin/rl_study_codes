import collections
import random

import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms.

    Stores (state, action, reward, next_state, done) transitions and allows
    random sampling to break temporal correlations in training data.

    Attributes:
        buffer (deque): Circular buffer storing experiences
        capacity (int): Maximum number of transitions to store
    """

    def __init__(self, capacity: int):
        """Initialize replay buffer with given capacity."""
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            done: Terminal flag
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self) -> int:
        """Return current number of transitions stored in buffer."""
        return len(self.buffer)


def moving_average(values, window_size):
    """Compute moving average of a sequence with specified window size.

    Provides smoothed version of input sequence for better visualization.
    Handles edges by using progressively smaller windows.

    Args:
        values: Input sequence to smooth
        window_size: Size of averaging window

    Returns:
        Smoothed sequence with same length as input
    """
    cumulative_sum = np.cumsum(np.insert(values, 0, 0))
    middle = (cumulative_sum[window_size:] -
              cumulative_sum[:-window_size]) / window_size

    # Handle edges with progressively smaller windows
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(values[:window_size - 1])[::2] / r
    end = (np.cumsum(values[:-window_size:-1])[::2] / r)[::-1]

    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    """Train an on-policy agent (e.g., Policy Gradient, PPO).

    On-policy algorithms require fresh samples from current policy for each update.

    Args:
        env: Training environment
        agent: On-policy agent implementing take_action() and update() methods
        num_episodes: Total number of training episodes

    Returns:
        List of returns for each episode
    """
    return_list = []

    for i in range(10):  # 10 progress bars
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for _ in range(int(num_episodes / 10)):
                # Initialize episode tracking
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }

                # Run episode
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # Store transition
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward

                # Update agent and track performance
                return_list.append(episode_return)
                agent.update(transition_dict
                             )  # On-policy update uses current episode data

                if len(return_list) % 10 == 0:
                    pbar.set_postfix({
                        'episode': len(return_list),
                        'return': np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size):
    """Train an off-policy agent (e.g., DQN, SAC).

    Off-policy algorithms can learn from historical data stored in replay buffer.

    Args:
        env: Training environment
        agent: Off-policy agent implementing take_action() and update() methods
        num_episodes: Total number of training episodes
        replay_buffer: Experience replay buffer
        minimal_size: Minimum buffer size before starting updates
        batch_size: Number of transitions per update

    Returns:
        List of returns for each episode
    """
    return_list = []

    for i in range(10):  # 10 progress bars
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for _ in range(int(num_episodes / 10)):
                # Run episode
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # Store transition in replay buffer
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    # Update agent if enough samples available
                    if replay_buffer.size() > minimal_size:
                        # Sample random batch from replay buffer
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': batch_states,
                            'actions': batch_actions,
                            'next_states': batch_next_states,
                            'rewards': batch_rewards,
                            'dones': batch_dones
                        }
                        agent.update(
                            transition_dict
                        )  # Off-policy update uses historical data

                # Track performance
                return_list.append(episode_return)
                if len(return_list) % 10 == 0:
                    pbar.set_postfix({
                        'episode': len(return_list),
                        'return': np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list


def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor):
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        gamma: Discount factor
        lmbda: GAE parameter (0=TD, 1=MC)
        td_delta: Tensor of TD errors (δ_t = r_t + γV(s_{t+1}) - V(s_t))

    Returns:
        Tensor of advantage estimates with same shape as td_delta
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0

    # Compute advantages by backward recursion
    for delta in reversed(td_delta):
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)

    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
