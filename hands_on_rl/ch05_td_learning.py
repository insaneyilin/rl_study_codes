import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class CliffWalkingEnv:

    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        # (0, 0) is the top left corner, (ncol-1, nrow-1) is the bottom right corner
        # start state is (0, nrow-1), goal state is (ncol-1, nrow-1)
        self.x = 0  # x for column
        self.y = self.nrow - 1  # y for row

    # Agent calls this function to take an action
    def step(self, action):
        # (x, y); 0 up, 1 down, 2 left, 3 right
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        # Check if the agent is on the cliff or at the goal
        # On the cliff: x in [1, 2, ..., ncol-2], y == nrow - 1
        # At the goal: x == ncol-1, y == nrow - 1
        # => x in [1, 2, ..., ncol-1], y == nrow - 1
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    # Reset the environment to the initial state and return the initial state.
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

    def goal_state(self):
        goal_x = self.ncol - 1
        goal_y = self.nrow - 1
        return goal_y * self.ncol + goal_x

    def cliff_states(self):
        return [
            self._row_col_to_1d_idx(self.nrow - 1, col)
            for col in range(1, self.ncol - 1)
        ]

    # NOTE: x for column, y for row
    def _row_col_to_1d_idx(self, row, col):
        return row * self.ncol + col


class SarsaAgent:
    """SARSA (on-policy TD control) algorithm implementation.

    Attributes:
        Q_table (np.array): Action-value function Q(s,a)
        n_actions (int): Number of possible actions
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate for ε-greedy policy
    """

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])  # Initialize Q(s,a)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        """Select action using ε-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(
                self.n_actions)  # Random action (exploration)
        return np.argmax(self.Q_table[state])  # Greedy action (exploitation)

    def get_greedy_action_distribution(self, state):
        """Get action distribution for visualization (greedy with ties)."""
        max_q = np.max(self.Q_table[state])
        return [1 if q == max_q else 0 for q in self.Q_table[state]]

    def update(self, state, action, reward, next_state, next_action):
        """Update Q-value using SARSA TD update rule:
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        td_target = reward + self.gamma * self.Q_table[next_state, next_action]
        td_error = td_target - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error


class NStepSarsaAgent:
    """n-step SARSA algorithm implementation.

    Attributes:
        n_steps (int): Number of steps to look ahead
        state_buffer (list): Stores last n states
        action_buffer (list): Stores last n actions
        reward_buffer (list): Stores last n rewards
    """

    def __init__(self,
                 n_steps,
                 ncol,
                 nrow,
                 epsilon,
                 alpha,
                 gamma,
                 n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def take_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q_table[state])

    def get_greedy_action_distribution(self, state):
        """Get action distribution for visualization."""
        max_q = np.max(self.Q_table[state])
        return [1 if q == max_q else 0 for q in self.Q_table[state]]

    def update(self, state, action, reward, next_state, next_action, done):
        """n-step SARSA update:
        G = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n Q(s_{t+n}, a_{t+n})
        """
        # Store current transition
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        if len(self.state_buffer) == self.n_steps:
            # Calculate n-step return
            G = self.Q_table[next_state,
                             next_action]  # Bootstrap with Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n_steps)):
                G = self.gamma * G + self.reward_buffer[i]

                # Update for all steps if episode terminated early
                if done and i > 0:
                    s = self.state_buffer[i]
                    a = self.action_buffer[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])

            # Delete the oldest transition from buffers
            s = self.state_buffer.pop(0)
            a = self.action_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])

        if done:  # Clear buffers at episode end
            self.state_buffer = []
            self.action_buffer = []
            self.reward_buffer = []


class QLearningAgent:
    """Q-learning (off-policy TD control) algorithm implementation."""

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q_table[state])

    def get_greedy_action_distribution(self, state):
        """Get action distribution for visualization."""
        max_q = np.max(self.Q_table[state])
        return [1 if q == max_q else 0 for q in self.Q_table[state]]

    def update(self, state, action, reward, next_state):
        """Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        td_target = reward + self.gamma * np.max(self.Q_table[next_state])
        td_error = td_target - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error


def run_training(agent, env, num_episodes=500):
    """Train an agent and return learning curve."""
    return_list = []
    for i in range(10):  # 10 progress bars
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
            for _ in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()

                # Special handling for SARSA which needs next_action
                if isinstance(agent, (SarsaAgent, NStepSarsaAgent)):
                    action = agent.take_action(state)

                done = False
                while not done:
                    if isinstance(agent, QLearningAgent):
                        action = agent.take_action(state)
                        next_state, reward, done = env.step(action)
                        agent.update(state, action, reward, next_state)
                    elif isinstance(agent, SarsaAgent):
                        next_state, reward, done = env.step(action)
                        next_action = agent.take_action(next_state)
                        agent.update(state, action, reward, next_state,
                                     next_action)
                        action = next_action
                    elif isinstance(agent, NStepSarsaAgent):
                        next_state, reward, done = env.step(action)
                        next_action = agent.take_action(next_state)
                        agent.update(state, action, reward, next_state,
                                     next_action, done)
                        action = next_action

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                if len(return_list) % 10 == 0:
                    pbar.set_postfix({
                        "episode": len(return_list),
                        "return": np.mean(return_list[-10:]),
                    })
                pbar.update(1)
    return return_list


def plot_learning_curve(return_list, algorithm_name):
    """Plot the learning curve."""
    plt.plot(range(len(return_list)), return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"{algorithm_name} on Cliff Walking")
    plt.show()


def print_policy(agent, env, action_symbols, cliff_states, goal_state):
    """Visualize the learned policy.

    Args:
        agent: The trained RL agent
        env: The environment
        action_symbols: List of symbols representing each action
        cliff_states: List of states representing cliff positions
        goal_state: List containing the goal state
    """
    for row in range(env.nrow):
        for col in range(env.ncol):
            state = row * env.ncol + col
            if state in cliff_states:
                print("XXXX", end=" ")  # Mark cliff states
            elif state in goal_state:
                print("GGGG", end=" ")  # Mark goal state
            else:
                action_dist = agent.get_greedy_action_distribution(state)
                policy_str = "".join(action_symbols[i] if prob > 0 else "o"
                                     for i, prob in enumerate(action_dist))
                print(policy_str, end=" ")
        print()  # New line at end of each row


# Environment setup
ncol, nrow = 12, 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
action_symbols = ["^", "v", "<", ">"]
cliff_states = env.cliff_states()
goal_state = [env.goal_state()]

# Hyperparameters
epsilon = 0.1
alpha = 0.1
gamma = 0.9
n_steps = 3
num_episodes = 500

# SARSA Training
print("\nTraining SARSA...")
sarsa_agent = SarsaAgent(ncol, nrow, epsilon, alpha, gamma)
sarsa_returns = run_training(sarsa_agent, env, num_episodes)
plot_learning_curve(sarsa_returns, "SARSA")
print("SARSA Final Policy:")
print_policy(sarsa_agent, env, action_symbols, cliff_states, goal_state)

# n-step SARSA Training
print("\nTraining n-step SARSA...")
nstep_sarsa_agent = NStepSarsaAgent(n_steps, ncol, nrow, epsilon, alpha, gamma)
nstep_sarsa_returns = run_training(nstep_sarsa_agent, env, num_episodes)
plot_learning_curve(nstep_sarsa_returns, f"{n_steps}-step SARSA")
print(f"{n_steps}-step SARSA Final Policy:")
print_policy(nstep_sarsa_agent, env, action_symbols, cliff_states, goal_state)

# Q-learning Training
print("\nTraining Q-learning...")
q_learning_agent = QLearningAgent(ncol, nrow, epsilon, alpha, gamma)
q_learning_returns = run_training(q_learning_agent, env, num_episodes)
plot_learning_curve(q_learning_returns, "Q-learning")
print("Q-learning Final Policy:")
print_policy(q_learning_agent, env, action_symbols, cliff_states, goal_state)
