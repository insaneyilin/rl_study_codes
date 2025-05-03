import collections
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rl_utils import ReplayBuffer
from tqdm import tqdm


class QNetwork(torch.nn.Module):
    """Neural Network for approximating the Q-function.

    Attributes:
        fc1 (nn.Linear): First fully connected layer
        fc2 (nn.Linear): Output layer (one Q-value per action)
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """Initialize Q-network with specified dimensions.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layer
            action_dim: Number of possible actions
        """
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # Input layer
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # Output layer

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input state tensor

        Returns:
            Tensor of Q-values for each action
        """
        x = F.relu(self.fc1(x))  # ReLU activation for hidden layer
        return self.fc2(x)  # Linear output for Q-values


class DQNAgent:
    """Deep Q-Network (DQN) implementation with experience replay and target network.

    Key components:
        1. Q-network for action-value estimation
        2. Target network for stable learning
        3. Experience replay buffer
        4. ε-greedy exploration strategy

    Attributes:
        q_net (QNetwork): Online Q-network (continuously updated)
        target_q_net (QNetwork): Target Q-network (periodically synchronized)
        optimizer (torch.optim): Optimizer for Q-network
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate for ε-greedy policy
        target_update (int): Frequency for updating target network
        count (int): Counter for target network updates
        device (torch.device): CPU or GPU for computation
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        """Initialize DQN agent with specified parameters."""
        self.action_dim = action_dim
        # Initialize both online and target Q-networks
        self.q_net = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, hidden_dim,
                                     action_dim).to(device)
        self.target_q_net.load_state_dict(
            self.q_net.state_dict())  # Synchronize initially

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # Discount factor (0 ≤ γ ≤ 1)
        self.epsilon = epsilon  # Exploration rate (ε)
        self.target_update = target_update  # Target network update frequency
        self.count = 0  # Counter for target updates
        self.device = device

    def take_action(self, state):
        """Select action using ε-greedy policy.

        Args:
            state: Current environment state

        Returns:
            action: Selected action (int)
        """
        if isinstance(state, list):
            state = np.array(state)  # Convert list to numpy array first
        state_tensor = torch.as_tensor(state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0)

        if torch.rand(1,
                      device=self.device).item() < self.epsilon:  # Exploration
            return torch.randint(0, self.action_dim, (1, ),
                                 device=self.device).item()
        else:  # Exploitation
            return self.q_net(state_tensor).argmax().item()  # Greedy action

    def update(self, transition_dict):
        """Perform DQN learning update with experience replay.

        Implements key DQN components:
            1. Experience replay (handled by ReplayBuffer)
            2. Target network for stable Q-targets
            3. TD-error minimization with MSE loss

        Args:
            transition_dict: Dictionary containing batch of transitions
        """
        # Convert batch data to tensors
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # Compute current Q-values for taken actions
        q_values = self.q_net(states).gather(1, actions)  # Q(s_t, a_t)

        # Compute target Q-values using target network
        # TD target: r_t + γ * max_a Q(s_{t+1}, a)
        # where max_a Q(s_{t+1}, a) is the maximum Q-value for next state
        # NOTE: target network is not updated during training (torch.no_grad())
        with torch.no_grad():  # No gradient for target computation
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
                -1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (
                1 - dones)  # TD target

        # Compute loss and update Q-network
        loss = F.mse_loss(q_values, q_targets)  # Mean squared TD error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def train_dqn_agent(env, agent, replay_buffer, num_episodes, minimal_size,
                    batch_size):
    """Training loop for DQN agent.

    Args:
        env: Training environment
        agent: DQN agent instance
        replay_buffer: Experience replay buffer
        num_episodes: Total training episodes
        minimal_size: Minimum replay buffer size before learning starts
        batch_size: Size of minibatches for training

    Returns:
        return_list: List of returns for each episode
    """
    return_list = []

    for i in range(10):  # 10 progress bars
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for _ in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False

                while not done:
                    # Agent interacts with environment
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # Store transition in replay buffer
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    # Train when enough samples are available
                    if replay_buffer.size() > minimal_size:
                        # Sample random batch from replay buffer
                        batch = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': batch[0],
                            'actions': batch[1],
                            'rewards': batch[2],
                            'next_states': batch[3],
                            'dones': batch[4]
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if len(return_list) % 10 == 0:
                    pbar.set_postfix({
                        'episode': len(return_list),
                        'return': np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list


if __name__ == "__main__":
    # Hyperparameters
    lr = 2e-3  # Learning rate
    num_episodes = 500  # Total training episodes
    hidden_dim = 128  # Hidden layer dimension
    gamma = 0.98  # Discount factor
    epsilon = 0.01  # Exploration rate
    target_update = 10  # Target network update frequency
    buffer_size = 10000  # Replay buffer capacity
    minimal_size = 500  # Minimum replay buffer size before training
    batch_size = 64  # Training batch size
    device = torch.device("cpu")
    # FIXME: Uncomment the following line if using MPS (Apple Silicon), but it works much slower
    # than CPU on my MacBook Pro.
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(42)
    np.random.seed(42)
    env.seed(42)
    torch.manual_seed(42)

    # Initialize components
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                     target_update, device)

    # Training
    returns = train_dqn_agent(env, agent, replay_buffer, num_episodes,
                              minimal_size, batch_size)

    # Plot results
    plt.plot(returns)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN Performance on {}'.format(env_name))
    plt.show()
