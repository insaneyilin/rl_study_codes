import gym
import matplotlib.pyplot as plt
import numpy as np
import rl_utils
import torch
import torch.nn.functional as F


class PolicyNetContinuous(torch.nn.Module):
    """Policy network for continuous action spaces that outputs Gaussian distribution parameters"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layer
            action_dim: Dimension of action space
        """
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,
                                   hidden_dim)  # Shared hidden layer
        self.fc_mu = torch.nn.Linear(hidden_dim,
                                     action_dim)  # Mean output layer
        self.fc_std = torch.nn.Linear(hidden_dim,
                                      action_dim)  # Std output layer

    def forward(self, x):
        """Forward pass that returns parameters of action distribution

        Returns:
            mu: Mean of Gaussian action distribution (tanh scaled to [-2, 2])
            std: Standard deviation (softplus to ensure positivity)
        """
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))  # Scale mean to [-2, 2] range
        std = F.softplus(self.fc_std(x))  # Ensure positive standard deviation
        return mu, std


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


class PPOContinuous:
    """Proximal Policy Optimization for continuous action spaces

    Key features:
    1. Gaussian policy with learnable mean and standard deviation
    2. Clipped objective function for stable updates
    3. Generalized Advantage Estimation (GAE)
    4. Multiple epochs of optimization on sampled data
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, clip_eps, gamma, device):
        """
        Initialize PPOContinuous agent

        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layers
            action_dim: Dimension of action space
            actor_lr: Learning rate for policy network
            critic_lr: Learning rate for value network
            lmbda: GAE parameter (0 ≤ λ ≤ 1)
            epochs: Number of optimization epochs per update
            clip_eps: Clipping parameter ε for PPO objective
            gamma: Discount factor
            device: Computation device
        """
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        # Hyperparameters
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # Number of optimization passes
        self.clip_eps = clip_eps  # PPO clipping parameter
        self.device = device

    def take_action(self, state):
        """Sample action from current policy distribution

        Args:
            state: Current environment state

        Returns:
            action: Sampled action from Gaussian policy
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        if isinstance(state, list):
            state = np.array(state)  # Convert list to numpy array first
        state_tensor = torch.as_tensor(state,
                                       dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
        mu, sigma = self.actor(state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.detach().cpu().numpy().flatten()  # Return as numpy array

    def update(self, transition_dict):
        """Perform PPO update using collected transitions

        Key steps:
        1. Compute advantages using GAE
        2. Normalize rewards (environment-specific)
        3. Perform multiple epochs of updates:
           - Policy updates with clipped objective
           - Value function updates with MSE
        """
        # Convert transitions to tensors
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # Environment-specific reward scaling (for Pendulum-v1)
        pendulum_max_reward = 0.0
        pendulum_min_reward = -16.2736044
        half_range = (pendulum_max_reward - pendulum_min_reward) / 2.0
        # Scale to approximately [-1, 1] range
        rewards = (rewards + half_range) / half_range

        # Compute TD targets and advantages
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (
                1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                                   td_delta.cpu()).to(
                                                       self.device)

            # Get old action probabilities for importance sampling
            mu_old, std_old = self.actor(states)
            old_action_dist = torch.distributions.Normal(
                mu_old.detach(), std_old.detach())
            old_log_probs = old_action_dist.log_prob(actions)

        # Multiple epochs of optimization
        for _ in range(self.epochs):
            # Get current action distribution and probabilities
            mu, std = self.actor(states)
            action_dist = torch.distributions.Normal(mu, std)
            log_probs = action_dist.log_prob(actions)

            # Compute probability ratios (importance weights)
            ratio = torch.exp(log_probs - old_log_probs)

            # Compute surrogate objectives
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                1 + self.clip_eps) * advantage

            # Policy loss (negative for gradient ascent)
            actor_loss = -torch.min(surr1, surr2).mean()

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
actor_lr = 1e-4  # Lower learning rate for continuous actions
critic_lr = 5e-3  # Higher learning rate for value function
num_episodes = 2000  # More episodes for continuous control
hidden_dim = 128
gamma = 0.9  # Lower discount factor for continuous tasks
lmbda = 0.9  # GAE parameter
epochs = 10  # Optimization epochs per update
clip_eps = 0.1  # PPO clipping parameter
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# Environment setup
env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.seed(42)
torch.manual_seed(42)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # Continuous action space

# Initialize agent
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, clip_eps, gamma, device)

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
plt.title(f'PPO Continuous Performance on {env_name}')
plt.legend()
plt.grid(True)
plt.show()
