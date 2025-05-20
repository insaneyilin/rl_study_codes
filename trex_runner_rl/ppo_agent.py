import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ppo_model import ActorNet, CriticNet


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
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


class PPOAgent:

    def __init__(self, input_shape: tuple, hidden_dim: int, action_dim: int,
                 actor_lr: float, critic_lr: float, mini_batch: int,
                 lmbda: float, epochs: int, eps: float, gamma: float,
                 device: torch.device):
        self.actor = ActorNet(input_shape, hidden_dim, action_dim).to(device)
        self.critic = CriticNet(input_shape, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)
        self.mini_batch = mini_batch
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # number of epochs for training on one episode
        self.eps = eps  # PPO clipping range parameter
        self.device = device

    def take_action(self, state):
        """Sample action from current policy π(a|s)

        Args:
            state: Current environment state

        Returns:
            action: Sampled action from policy distribution
        """
        state = torch.tensor(np.expand_dims(state, 0),
                             dtype=torch.float).to(self.device)
        # NOTE: here we use logits instead of probs, check ActorNet, we don't use softmax.
        logits = self.actor(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        """Perform PPO update using collected transitions

        Key steps:
        1. Compute TD targets and advantages using GAE
        2. Perform multiple epochs of updates on both networks
        3. Use clipped surrogate objective for policy updates
        4. Update value function with MSE loss
        """
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']),
                               dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']),
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']),
                             dtype=torch.float).view(-1, 1).to(self.device)

        batch = states.size()[0]
        td_targets, td_deltas, old_log_probs = [], [], []
        with torch.no_grad():
            # Use mini-batch to compute targets and deltas
            for i in range(0, batch, self.mini_batch):
                j = i + self.mini_batch
                state = states[i:j]
                action = actions[i:j]
                reward = rewards[i:j]
                next_state = next_states[i:j]
                done = dones[i:j]

                # r + γV(s') if not done
                td_target = reward + self.gamma * self.critic(next_state) * (
                    1 - done)

                # Advantage: δ = r + γV(s') - V(s)
                td_delta = td_target - self.critic(state)

                # Save old log probabilities for importance sampling
                old_log_prob = F.log_softmax(self.actor(state),
                                             dim=-1).gather(1, action)

                # Add to mini-batch lists
                td_targets.append(td_target)
                td_deltas.append(td_delta)
                old_log_probs.append(old_log_prob)

        # Concatenate mini-batch lists to tensors
        td_targets = torch.cat(td_targets).detach()
        td_deltas = torch.cat(td_deltas).cpu()  # cpu for `compute_advantage`
        old_log_probs = torch.cat(old_log_probs).detach()

        advantages = compute_advantage(self.gamma, self.lmbda,
                                       td_deltas).to(self.device)
        # Normalize the advantage.
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                                                         1e-6)

        # Shuffle the original data.
        indices = torch.randperm(batch)
        states = states[indices]
        actions = actions[indices]
        td_targets = td_targets[indices]
        old_log_probs = old_log_probs[indices]
        advantages = advantages[indices]

        # Average importance weight ratio across mini-batches.
        ratios = 0
        # Dynamic clipping range based on batch size. eps is the initial clipping range.
        # The clipping range is scaled by the square root of (256 / batch), 256 is the default batch size.
        # When batch size < 256, the clipping range is larger than eps, more exploration.
        # When batch size > 256, the clipping range is smaller than eps, less exploration.
        clip_range = self.eps * ((256 / batch)**.5)

        for _ in range(self.epochs):
            for i in range(0, batch, self.mini_batch):
                j = i + self.mini_batch
                state = states[i:j]
                action = actions[i:j]
                td_target = td_targets[i:j]
                old_log_prob = old_log_probs[i:j]
                advantage = advantages[i:j]

                # Compute current log probabilities and ratios
                log_prob = F.log_softmax(self.actor(state),
                                         dim=-1).gather(1, action)
                # Importance weight
                ratio = torch.exp(log_prob - old_log_prob)

                # Compute surrogate objectives
                surr1 = ratio * advantage  # Unclipped objective
                # Clipped objective
                surr2 = torch.clamp(ratio, 1 - clip_range,
                                    1 + clip_range) * advantage

                # PPO's clipped objective.
                # Negative for gradient ascent.
                actor_loss = torch.mean(-torch.min(surr1, surr2))
                # Value function loss (MSE)
                critic_loss = torch.mean(
                    F.mse_loss(self.critic(state), td_target))

                # Accumulate importance weight.
                ratios += torch.mean(ratio).item()

                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                               max_norm=0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        num_minibatches = (batch - 1) // self.mini_batch + 1
        num_iterations = num_minibatches * self.epochs
        ratios = ratios / num_iterations
        # The avg ratio can be used to track the performance of the agent.
        # If the ratio is near 1.0, the agent is learning well.
        # Otherwise if the ratio too high or too low, it means the policy is updated with less stability.
        return ratios

    def save(self, path):
        actor_path = os.path.join(path, 'actor.pth')
        critic_path = os.path.join(path, 'critic.pth')
        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)

    def load(self, path):
        actor_path = os.path.join(path, 'actor.pth')
        critic_path = os.path.join(path, 'critic.pth')
        self.actor = torch.load(actor_path, weights_only=False)
        self.critic = torch.load(critic_path, weights_only=False)
