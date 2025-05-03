import gym
import matplotlib.pyplot as plt
import numpy as np
import rl_utils
import torch
import torch.nn.functional as F
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    """Neural network to parameterize the policy π(a|s)"""

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
    """Neural network to estimate state value function V(s) (the baseline)"""

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class REINFORCE:
    """Implementation of the REINFORCE policy gradient algorithm"""

    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 learning_rate,
                 gamma,
                 device,
                 reward_to_go=False,
                 use_baseline=False):
        """
        Initialize REINFORCE agent

        Args:
            state_dim: Dimension of state space
            hidden_dim: Number of neurons in hidden layer
            action_dim: Number of possible actions
            learning_rate: Learning rate for policy gradient updates
            gamma: Discount factor for future rewards
            device: Computation device (CPU/GPU)
            reward_to_go: Whether to use reward-to-go or not
            use_baseline: Whether to use a value function as a baseline
        """
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # Discount factor for future rewards
        self.device = device
        self.reward_to_go = reward_to_go  # Use reward-to-go or not
        self.use_baseline = use_baseline  # Use a value function as a baseline
        if self.use_baseline:
            self.value_net = ValueNet(state_dim, hidden_dim).to(device)
            self.optimizer_value = torch.optim.Adam(
                self.value_net.parameters(), lr=learning_rate)

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
        # NOTE: note the difference between PolicyNet in Policy Gradient and QNet in DQN:
        # Policy-based method outputs action probabilities, take action from the distribution.
        # Value-based method outputs Q-values, take action with max Q-value.
        action_probs = self.policy_net(
            state_tensor)  # Get action probabilities
        action_dist = torch.distributions.Categorical(
            action_probs)  # Create categorical distribution
        action = action_dist.sample()  # Sample an action
        return action.item()  # Return action as Python scalar

    def update(self, episode_transitions):
        """
        Update policy network using the REINFORCE algorithm

        REINFORCE Update Rule:
        Δθ = α * γ^t * G_t * ∇_θ log π(a_t|s_t)

        Where:
        - α is learning rate
        - γ is discount factor
        - G_t is return from time t
        - π(a_t|s_t) is policy probability for action a_t in state s_t

        Args:
            episode_transitions: Dictionary containing one complete episode's transitions
        """
        rewards = episode_transitions['rewards']
        states = episode_transitions['states']
        actions = episode_transitions['actions']

        # Initialize return and zero gradients
        self.optimizer.zero_grad()

        if self.reward_to_go:
            # Reward-to-Go version: each action is weighted by its future discounted rewards
            discounted_return = 0  # G_t
            # Calculate gradients for each step in reverse order (from last to first)
            for t in reversed(range(len(rewards))):
                reward = rewards[t]
                state = torch.tensor([states[t]],
                                     dtype=torch.float).to(self.device)
                action = torch.tensor([actions[t]]).view(-1, 1).to(self.device)

                # Calculate log probability of the taken action
                action_probs = self.policy_net(state)
                log_prob = torch.log(action_probs.gather(1, action))

                # Update discounted return (G_t = r_t + γ*G_{t+1})
                discounted_return = self.gamma * discounted_return + reward
                weight = discounted_return  # Weight for current log_prob
                if self.use_baseline:
                    # Use value function as a baseline
                    state_value = self.value_net(state)
                    # Calculate advantage (A_t = G_t - V(s_t))
                    # NOTE: detach() is used to prevent backpropagation through the value network
                    # during the policy update.
                    advantage = discounted_return - state_value.detach().item()
                    weight = advantage  # Use advantage as weight

                    # Update value network (MSE loss)
                    target_value = torch.tensor(
                        discounted_return,
                        dtype=state_value.dtype,
                        device=state_value.device).unsqueeze(0).unsqueeze(0)
                    value_loss = F.mse_loss(state_value, target_value)
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    self.optimizer_value.step()

                # Calculate loss (negative because we do gradient ascent)
                policy_loss = -log_prob * weight

                # Backpropagate to accumulate gradients
                policy_loss.backward()
        else:
            # Vanilla REINFORCE: all actions weighted by full episode return
            # Calculate full episode discounted return
            full_return = 0
            for t in range(len(rewards)):
                full_return += (self.gamma**t) * rewards[t]
            # Calculate loss for all actions using the same full_return
            for t in range(len(rewards)):
                state = torch.tensor([states[t]],
                                     dtype=torch.float).to(self.device)
                action = torch.tensor([actions[t]]).view(-1, 1).to(self.device)
                # Calculate log probability of the taken action
                action_probs = self.policy_net(state)
                log_prob = torch.log(action_probs.gather(1, action))
                # Calculate loss (negative because we do gradient ascent)
                # NOTE: full_return is the same for all actions in this case
                policy_loss = -log_prob * full_return
                policy_loss.backward()

        # Perform parameter update after processing entire episode
        self.optimizer.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='REINFORCE for CartPole-v0')
    parser.add_argument('--reward_to_go',
                        action='store_true',
                        help='Whether to use reward-to-go or not')
    parser.add_argument('--use_baseline',
                        action='store_true',
                        help='Use value function baseline')
    args = parser.parse_args()
    if args.use_baseline:
        assert args.reward_to_go, "Baseline can only be used with reward-to-go"

    # Hyperparameters
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Environment setup
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    env.seed(0)  # For reproducibility
    torch.manual_seed(0)  # For reproducibility
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent
    agent = REINFORCE(state_dim,
                      hidden_dim,
                      action_dim,
                      learning_rate,
                      gamma,
                      device,
                      reward_to_go=args.reward_to_go,
                      use_baseline=args.use_baseline)

    # Training loop
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc=f'Training Iteration {i+1}') as pbar:
            for episode in range(int(num_episodes / 10)):
                # Initialize episode variables
                episode_return = 0
                episode_transitions = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'dones': []
                }

                # Run one episode
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)

                    # Store transition
                    episode_transitions['states'].append(state)
                    episode_transitions['actions'].append(action)
                    episode_transitions['rewards'].append(reward)
                    episode_transitions['dones'].append(done)

                    state = next_state
                    episode_return += reward

                # Store episode return and update policy
                return_list.append(episode_return)
                agent.update(episode_transitions)

                # Update progress bar
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        f'{int(num_episodes/10)*i + episode + 1}',
                        'avg_return':
                        f'{np.mean(return_list[-10:]):.1f}'
                    })
                pbar.update(1)

    # Plot results
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_list, return_list, label='Episode Returns', alpha=0.3)
    plt.plot(rl_utils.moving_average(return_list, 9),
             label='Moving Average (window=9)',
             color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'REINFORCE Performance on {env_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
