import os
import random
import time
from typing import Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from rl_utils import ReplayBuffer
from tqdm import tqdm


# We use Pendulum-v1 as an example, which has continuous action space.
# DQN is designed for discrete action spaces, so we will discretize the
# continuous action space into a finite number of discrete actions.
def discrete_to_continuos(discrete_action, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)


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


class VAnet(torch.nn.Module):
    """Dueling DQN Network Architecture that separately estimates state value and action advantages.

    Key components:
    1. Shared feature extraction layers
    2. Value stream (V) estimating state value
    3. Advantage stream (A) estimating action advantages
    4. Special aggregation layer combining V and A

    Paper reference: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        """Initialize the dueling network architecture.

        Args:
            state_dim: Dimension of the input state space
            hidden_dim: Number of neurons in the hidden layer
            action_dim: Number of possible actions
        """
        super(VAnet, self).__init__()

        # Shared feature extraction layer (common to both streams)
        self.shared_fc = torch.nn.Linear(state_dim, hidden_dim)

        # Advantage stream head (estimates advantage for each action)
        self.advantage_stream = torch.nn.Linear(hidden_dim, action_dim)

        # Value stream head (estimates state value)
        self.value_stream = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network.

        Args:
            state: Input state tensor (batch_size × state_dim)

        Returns:
            Q-values tensor (batch_size × action_dim)
        """
        # Shared feature extraction
        shared_features = F.relu(self.shared_fc(state))

        # Calculate advantage values for each action
        advantages = self.advantage_stream(shared_features)

        # Calculate state value
        state_value = self.value_stream(shared_features)

        # Combine streams using dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        # This helps with identifiability and stable learning
        q_values = state_value + (advantages -
                                  advantages.mean(dim=1, keepdim=True))

        return q_values


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
        dqn_type (str): Type of DQN (VanillaDQN or DoubleDQN or DuelingDQN)
    """

    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 device,
                 dqn_type='VanillaDQN'):
        """Initialize DQN agent with specified parameters."""
        self.action_dim = action_dim
        # Initialize both online and target Q-networks
        if dqn_type == 'DuelingDQN':
            self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim,
                                      action_dim).to(device)
        else:
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
        self.dqn_type = dqn_type
        print(f"Using {self.dqn_type}")

    def take_action(self, state, deterministic: bool = False):
        """Select action using ε-greedy policy or deterministically.

        Args:
            state: Current environment state
            deterministic: If True, always takes the best action (for testing)

        Returns:
            action: Selected action (int)
        """
        if isinstance(state, list):
            state = np.array(state)  # Convert list to numpy array first
        state_tensor = torch.as_tensor(state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0)

        if not deterministic and torch.rand(
                1, device=self.device).item() < self.epsilon:  # Exploration
            return torch.randint(0, self.action_dim, (1, ),
                                 device=self.device).item()
        else:  # Exploitation
            return self.q_net(state_tensor).argmax().item()  # Greedy action

    def get_max_q_value(self, state):
        if isinstance(state, list):
            state = np.array(state)  # Convert list to numpy array first
        state_tensor = torch.as_tensor(state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
        return self.q_net(state_tensor).max().item()

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
            if self.dqn_type == 'VanillaDQN' or self.dqn_type == 'DuelingDQN':
                max_next_q_values = self.target_q_net(next_states).max(
                    1)[0].view(-1, 1)
            elif self.dqn_type == 'DoubleDQN':
                # Action selection using online Q-network
                max_q_action = self.q_net(next_states).max(1)[1].view(-1, 1)
                # Action evaluation using target Q-network
                max_next_q_values = self.target_q_net(next_states).gather(
                    1, max_q_action)
            else:
                raise ValueError(f"Unknown DQN type: {self.dqn_type}")
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

    def save(self, path: str):
        """Save the agent's Q-network to a file.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                'q_net_state_dict': self.q_net.state_dict(),
                'target_q_net_state_dict': self.target_q_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'count': self.count,
                'epsilon': self.epsilon,
            }, path)

    def load(self, path: str, device: torch.device):
        """Load the agent's Q-network from a file.

        Args:
            path: Path to load the model from
            device: Device to load the model onto
        """
        checkpoint = torch.load(path, map_location=device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_q_net.load_state_dict(
            checkpoint['target_q_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.count = checkpoint['count']
        self.epsilon = checkpoint['epsilon']
        self.q_net.to(device)
        self.target_q_net.to(device)


def train_dqn_agent(env,
                    agent,
                    replay_buffer,
                    num_episodes,
                    minimal_size,
                    batch_size,
                    checkpoint_dir: Optional[str] = None,
                    checkpoint_freq: int = 100):
    """Training loop for DQN agent.

    Args:
        env: Training environment
        agent: DQN agent instance
        replay_buffer: Experience replay buffer
        num_episodes: Total training episodes
        minimal_size: Minimum replay buffer size before learning starts
        batch_size: Size of minibatches for training
        checkpoint_dir: Directory to save checkpoints (None to disable)
        checkpoint_freq: Frequency (in episodes) to save checkpoints

    Returns:
        return_list: List of returns for each episode
    """
    return_list = []
    max_q_value_list = []
    max_q_value = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for i in range(10):  # 10 progress bars
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False

                while not done:
                    # Agent interacts with environment
                    action = agent.take_action(state)
                    max_q_value = agent.get_max_q_value(state)
                    max_q_value_list.append(max_q_value)
                    action_continuous = discrete_to_continuos(
                        action, env, agent.action_dim)
                    # print(action)
                    # print(action_continuous)
                    next_state, reward, done, _ = env.step([action_continuous])

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
                    avg_return = np.mean(return_list[-10:])
                    pbar.set_postfix({
                        'episode': len(return_list),
                        'return': avg_return
                    })
                    # Save checkpoint.
                    if checkpoint_dir is not None and len(
                            return_list) % checkpoint_freq == 0:
                        checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f'episode_{len(return_list)}_avg_return_{avg_return}.pth'
                        )
                        agent.save(checkpoint_path)
                        print(f"\nSaved checkpoint to {checkpoint_path}")
                pbar.update(1)

    return return_list, max_q_value_list


def test_agent(env,
               agent,
               num_episodes: int = 10,
               render: bool = True,
               max_steps: int = 500,
               pause: float = 0.01):
    """Test the trained agent.

    Args:
        env: Environment to test in
        agent: Trained agent
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        max_steps: Maximum steps per episode
        pause: Pause time between rendered frames (for visualization)
    """
    returns = []
    lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            if render:
                env.render()
                time.sleep(pause)

            action = agent.take_action(state, deterministic=True)
            action_continuous = discrete_to_continuos(action, env,
                                                      agent.action_dim)
            next_state, reward, done, _ = env.step([action_continuous])
            state = next_state
            episode_return += reward
            episode_length += 1

        returns.append(episode_return)
        lengths.append(episode_length)
        print(
            f"Episode {episode + 1}: Return = {episode_return}, Length = {episode_length}"
        )

    env.close()
    print(f"\nAverage return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(
        f"Average episode length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}"
    )

    return returns, lengths


def plot_returns(returns, window_size=10, title="DQN Performance"):
    """Plot training returns with moving average.

    Args:
        returns: List of returns from training
        window_size: Size of moving average window
        title: Plot title
    """
    # Calculate moving average
    moving_avg = np.convolve(returns,
                             np.ones(window_size) / window_size,
                             mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(returns, label='Episode Returns', alpha=0.3)
    plt.plot(moving_avg,
             label=f'Moving Average (window={window_size})',
             color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_max_q_values(max_q_value_list, title="DQN Max Q-value"):
    """Plot maximum Q-values

    Args:
        max_q_value_list: List of maximum Q-values from training
        title: Plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(max_q_value_list, label='Max Q-value', alpha=0.3)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Max Q-value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """Main function with training and testing modes."""
    import argparse

    parser = argparse.ArgumentParser(description='DQN for Pendulum-v1')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test'],
                        help='Mode: "train" to train, "test" to load and test')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file for testing or resuming training')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='dqn_checkpoints',
                        help='Directory to save checkpoints during training')
    parser.add_argument('--num_episodes',
                        type=int,
                        default=200,
                        help='Number of training episodes')
    parser.add_argument('--test_episodes',
                        type=int,
                        default=10,
                        help='Number of test episodes')
    parser.add_argument('--render',
                        action='store_true',
                        help='Render environment during testing')
    parser.add_argument(
        '--dqn_type',
        type=str,
        default='DoubleDQN',
        help='Type of DQN to use (VanillaDQN or DoubleDQN or DuelingDQN)')
    args = parser.parse_args()
    assert args.dqn_type in ['VanillaDQN', 'DoubleDQN', 'DuelingDQN']

    # Hyperparameters
    lr = 1e-2  # Learning rate
    hidden_dim = 128  # Hidden layer dimension
    gamma = 0.98  # Discount factor
    epsilon = 0.01  # Exploration rate
    target_update = 50  # Target network update frequency
    buffer_size = 5000  # Replay buffer capacity
    minimal_size = 1000  # Minimum replay buffer size before training
    batch_size = 64  # Training batch size
    device = torch.device("cpu")
    # FIXME: Uncomment the following line if using MPS (Apple Silicon), but it works much slower
    # than CPU on my MacBook Pro.
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    # NOTE: Pendulum-v1 is a continuous action space environment, but we will
    # discretize it into 11 discrete actions for DQN.
    state_dim = env.observation_space.shape[0]
    action_dim = 11

    random.seed(42)
    np.random.seed(42)
    env.seed(42)
    torch.manual_seed(42)

    # Initialize components
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    # Pendulum-v1 has continuous action space, but we will discretize it
    # action_dim = env.action_space.n
    agent = DQNAgent(state_dim,
                     hidden_dim,
                     action_dim,
                     lr,
                     gamma,
                     epsilon,
                     target_update,
                     device,
                     dqn_type=args.dqn_type)

    if args.mode == 'train':
        # Training mode
        if args.checkpoint:  # Resume training from checkpoint
            agent.load(args.checkpoint, device)
            print(f"Resumed training from checkpoint: {args.checkpoint}")

        returns, max_q_value_list = train_dqn_agent(
            env,
            agent,
            replay_buffer,
            args.num_episodes,
            minimal_size,
            batch_size,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_freq=50)

        # Save final model
        final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
        agent.save(final_path)
        print(f"\nSaved final model to {final_path}")

        # Plot training results
        plot_returns(returns, title=f'{args.dqn_type} Training on {env_name}')

        plot_max_q_values(max_q_value_list,
                          title=f'{args.dqn_type} Max Q-value on {env_name}')

        # Quick test after training
        print("\nTesting trained agent...")
        test_agent(env,
                   agent,
                   num_episodes=args.test_episodes,
                   render=args.render)

    elif args.mode == 'test':
        # Testing mode
        if args.checkpoint is None:
            # Try to load the final model if no checkpoint specified
            args.checkpoint = os.path.join(args.checkpoint_dir,
                                           'final_model.pth')

        agent.load(args.checkpoint, device)
        print(f"Loaded model from {args.checkpoint}")

        # Test the agent
        test_agent(env,
                   agent,
                   num_episodes=args.test_episodes,
                   render=args.render)


if __name__ == "__main__":
    main()
