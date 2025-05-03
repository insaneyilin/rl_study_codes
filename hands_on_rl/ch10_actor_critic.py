import gym
import matplotlib.pyplot as plt
import numpy as np
import rl_utils
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    """Policy network (Actor) that outputs action probabilities π(a|s)"""

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
    """Value network (Critic) that estimates state value V(s)"""

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,
                                   hidden_dim)  # First fully connected layer
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # Outputs scalar value

    def forward(self, x):
        """Forward pass that returns state value estimate"""
        x = F.relu(self.fc1(x))  # ReLU activation for hidden layer
        return self.fc2(x)  # Linear output for state value


class ActorCritic:
    """Implementation of the Actor-Critic algorithm

    Key components:
    1. Actor (PolicyNet): Selects actions and learns to improve policy
    2. Critic (ValueNet): Evaluates state values and provides feedback
    3. TD Error: Combines immediate reward with discounted next state value

    The Actor is updated using policy gradient with the Critic's TD error as advantage estimate.
    The Critic is updated using TD learning to better estimate state values.
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        """
        Initialize Actor-Critic agent

        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layers
            action_dim: Number of possible actions
            actor_lr: Learning rate for policy network
            critic_lr: Learning rate for value network
            gamma: Discount factor for future rewards
            device: Computation device (CPU/GPU)
        """
        # Initialize networks
        self.actor = PolicyNet(state_dim, hidden_dim,
                               action_dim).to(device)  # Policy network
        self.critic = ValueNet(state_dim,
                               hidden_dim).to(device)  # Value network

        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)

        self.gamma = gamma  # Discount factor (0 ≤ γ ≤ 1)
        self.device = device  # Computation device

    def take_action(self, state):
        """
        Sample an action from the current policy π(a|s)

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
        action_probs = self.actor(
            state_tensor)  # Get action probabilities from policy network
        action_dist = torch.distributions.Categorical(
            action_probs)  # Create categorical distribution
        return action_dist.sample().item()  # Sample and return action

    def update(self, transition_dict):
        """
        Perform one update step for both Actor and Critic

        Key steps:
        1. Compute TD targets: r + γ*V(s')
        2. Compute TD errors (advantage estimates): δ = TD_target - V(s)
        3. Update Critic to minimize MSE between V(s) and TD_target
        4. Update Actor using policy gradient with TD error as advantage

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

        # 1. Compute TD targets: r + γ*V(s')*(1 - done)
        # If episode terminates (done=1), next state value is 0
        with torch.no_grad():  # No gradient for target computation
            td_targets = rewards + self.gamma * self.critic(next_states) * (
                1 - dones)

        # 2. Compute TD errors (advantage estimates), use TD error to approximate advantage.
        # NOTE: This is different from REINFORCE with baseline, TD error use bootstrapping, while
        # REINFORCE use Monte Carlo estimate advantage (A value net that approximate V(s)).
        td_errors = td_targets - self.critic(states)

        # 3. Update Actor (policy network)
        # Policy gradient: ∇J(θ) ≈ E[∇logπ(a|s) * A(s,a)] where A(s,a) ≈ δ
        log_probs = torch.log(self.actor(states).gather(1,
                                                        actions))  # logπ(a|s)
        actor_loss = -torch.mean(
            log_probs * td_errors.detach())  # Negative for gradient ascent

        # 4. Update Critic (value network)
        # MSE loss between current value estimates and TD targets
        critic_loss = F.mse_loss(self.critic(states), td_targets.detach())

        # Perform gradient updates
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # Compute policy gradients
        critic_loss.backward()  # Compute value gradients
        self.actor_optimizer.step()  # Update policy parameters
        self.critic_optimizer.step()  # Update value parameters


# Hyperparameters
actor_lr = 1e-3  # Learning rate for policy network
critic_lr = 1e-2  # Learning rate for value network (typically higher)
num_episodes = 1000  # Total training episodes
hidden_dim = 128  # Hidden layer dimension
gamma = 0.98  # Discount factor
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
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

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
plt.title(f'Actor-Critic Performance on {env_name}')
plt.legend()
plt.grid(True)
plt.show()
