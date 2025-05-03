import gym
import matplotlib.pyplot as plt
import numpy as np
import rl_utils
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    """Policy network (actor) that outputs action probabilities π(a|s)"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,
                                   hidden_dim)  # First fully connected layer
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # Output layer

    def forward(self, x):
        """Forward pass that returns action probabilities"""
        x = F.relu(self.fc1(x))  # ReLU activation for hidden layer
        return F.softmax(self.fc2(x),
                         dim=1)  # Softmax for probability distribution


class ValueNet(torch.nn.Module):
    """Value network (critic) that estimates state value V(s)"""

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,
                                   hidden_dim)  # First fully connected layer
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # Outputs scalar value

    def forward(self, x):
        """Forward pass that returns state value estimate"""
        x = F.relu(self.fc1(x))  # ReLU activation for hidden layer
        return self.fc2(x)  # Linear output for state value


class PPO:
    """Proximal Policy Optimization (PPO) implementation with clipped objective

    Key components:
    1. Actor (PolicyNet): Selects actions and learns to improve policy
    2. Critic (ValueNet): Estimates state values and provides baseline
    3. Clipped objective: Prevents excessively large policy updates
    4. Generalized Advantage Estimation (GAE): Computes advantage estimates

    PPO Theory:
    - Maximizes surrogate objective L(θ) = E[min(r(θ)A, clip(r(θ), 1±ε)A]
    - Uses multiple epochs of minibatch updates from sampled data
    - Maintains stability through clipped probability ratios
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, clip_eps, gamma, device):
        """
        Initialize PPO agent

        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layers
            action_dim: Number of possible actions
            actor_lr: Learning rate for policy network
            critic_lr: Learning rate for value network
            lmbda: GAE parameter (0 ≤ λ ≤ 1)
            epochs: Number of optimization epochs per update
            clip_eps: Clipping parameter ε for PPO objective
            gamma: Discount factor for future rewards
            device: Computation device (CPU/GPU)
        """
        # Initialize networks
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.lmbda = lmbda  # GAE parameter
        self.epochs = epochs  # Number of optimization epochs
        self.clip_eps = clip_eps  # Clipping parameter ε
        self.device = device  # Computation device

    def take_action(self, state):
        """Sample action from current policy π(a|s)

        Args:
            state: Current environment state

        Returns:
            action: Sampled action from policy distribution
        """
        if isinstance(state, list):
            state = np.array(state)  # Convert list to numpy array first
        state_tensor = torch.as_tensor(state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
        action_probs = self.actor(state_tensor)  # Get action probabilities
        action_dist = torch.distributions.Categorical(
            action_probs)  # Create distribution
        return action_dist.sample().item()  # Sample and return action

    def update(self, transition_dict):
        """Perform PPO update using collected transitions

        Key steps:
        1. Compute TD targets and advantages using GAE
        2. Perform multiple epochs of updates on both networks
        3. Use clipped surrogate objective for policy updates
        4. Update value function with MSE loss
        """
        # Convert transitions to tensors
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

        # 1. Compute TD targets and advantages
        with torch.no_grad():
            # TD target: r + γV(s') if not done
            td_target = rewards + self.gamma * self.critic(next_states) * (
                1 - dones)
            # Advantage: δ = r + γV(s') - V(s)
            td_delta = td_target - self.critic(states)
            # Compute GAE advantages
            advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                                   td_delta.cpu()).to(
                                                       self.device)
            # Save old log probabilities for importance sampling
            old_log_probs = torch.log(self.actor(states).gather(
                1, actions)).detach()

        # 2. Perform multiple epochs of updates
        for _ in range(self.epochs):
            # Compute current log probabilities and ratios
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)  # Importance weight

            # Compute surrogate objectives
            surr1 = ratio * advantage  # Unclipped objective
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 +
                                self.clip_eps) * advantage  # Clipped objective

            # PPO's clipped objective
            actor_loss = -torch.min(
                surr1, surr2).mean()  # Negative for gradient ascent

            # Value function loss (MSE)
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


# Hyperparameters
actor_lr = 1e-3  # Learning rate for policy network
critic_lr = 1e-2  # Learning rate for value network (typically higher)
num_episodes = 500  # Total training episodes
hidden_dim = 128  # Hidden layer dimension
gamma = 0.98  # Discount factor
lmbda = 0.95  # GAE parameter
epochs = 10  # Number of optimization epochs per update
clip_eps = 0.2  # Clipping parameter ε
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# Environment setup
env_name = 'CartPole-v0'
env = gym.make(env_name)
env.seed(0)  # For reproducibility
torch.manual_seed(0)  # For reproducibility
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize agent
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, clip_eps, gamma, device)

# Training loop
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

# Plot results
episodes_list = list(range(len(return_list)))
plt.figure(figsize=(10, 5))
plt.plot(episodes_list, return_list, label='Episode Returns', alpha=0.3)
plt.plot(rl_utils.moving_average(return_list, 9),
         label='Moving Average (window=9)',
         color='red')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(f'PPO Performance on {env_name}')
plt.legend()
plt.grid(True)
plt.show()
