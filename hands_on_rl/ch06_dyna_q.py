import random

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


class DynaQAgent:
    """Implementation of Dyna-Q algorithm.

    Combines:
    1. Direct RL (Q-learning) from real experience
    2. Planning (model-based updates) from simulated experience

    Attributes:
        Q_table (np.array): Action-value function Q(s,a)
        n_actions (int): Number of possible actions
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate for ε-greedy policy
        n_planning (int): Number of planning steps per real step
        model (dict): Learned environment model (s,a) -> (r,s')
    """

    def __init__(self,
                 ncol,
                 nrow,
                 epsilon,
                 alpha,
                 gamma,
                 n_planning,
                 n_actions=4):
        self.Q_table = np.zeros([nrow * ncol, n_actions])  # Initialize Q(s,a)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning  # Planning steps per real step
        self.model = {}  # Environment model: (s,a) -> (r, s')

    def take_action(self, state):
        """Select action using ε-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Exploration
        return np.argmax(self.Q_table[state])  # Exploitation

    def q_learning_update(self, state, action, reward, next_state):
        """Perform Q-learning update:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        td_target = reward + self.gamma * np.max(self.Q_table[next_state])
        td_error = td_target - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error

    def update(self, state, action, reward, next_state):
        """Dyna-Q update combining:
        1. Direct RL update from real experience
        2. Model learning
        3. Planning updates from model
        """
        # 1. Direct RL update
        self.q_learning_update(state, action, reward, next_state)

        # 2. Model learning - store transition in model
        self.model[(state, action)] = (reward, next_state)

        # 3. Planning - perform n_planning updates from model
        for _ in range(self.n_planning):
            # Randomly sample previously experienced (s,a) pair
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            # Update Q-table using simulated experience
            self.q_learning_update(s, a, r, s_)

    def get_greedy_action_distribution(self, state):
        """Get action distribution for visualization (greedy with ties)."""
        max_q = np.max(self.Q_table[state])
        return [1 if q == max_q else 0 for q in self.Q_table[state]]


def run_dyna_q_training(env, n_planning_steps, num_episodes=300):
    """Train Dyna-Q agent and return learning curve.

    Args:
        n_planning_steps: Number of planning steps per real step
        num_episodes: Total number of training episodes

    Returns:
        List of returns for each episode
    """
    ncol, nrow = env.ncol, env.nrow
    agent = DynaQAgent(ncol,
                       nrow,
                       epsilon=0.01,
                       alpha=0.1,
                       gamma=0.9,
                       n_planning=n_planning_steps)

    return_list = []
    for i in range(10):  # 10 progress bars
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
            for _ in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state)
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                if len(return_list) % 10 == 0:
                    pbar.set_postfix({
                        "episode": len(return_list),
                        "return": np.mean(return_list[-10:]),
                    })
                pbar.update(1)
    return return_list, agent


def plot_results(n_planning_steps_list, returns_list):
    """Plot learning curves for different planning steps."""
    plt.figure(figsize=(10, 6))

    for n_planning, returns in zip(n_planning_steps_list, returns_list):
        plt.plot(range(len(returns)),
                 returns,
                 label=f"{n_planning} planning steps")

    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title("Dyna-Q Performance on Cliff Walking")
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


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    ncol, nrow = 12, 4
    env = CliffWalkingEnv(ncol, nrow)
    action_symbols = ["^", "v", "<", ">"]
    cliff_states = env.cliff_states()
    goal_state = [env.goal_state()]

    planning_steps_to_test = [0, 2, 20]  # 0 = Q-learning only
    returns_list = []
    for n_planning in planning_steps_to_test:
        print(f"Planning steps: {n_planning}")
        returns, agent = run_dyna_q_training(env, n_planning)
        returns_list.append(returns)
        print(f"============= n_planning: {n_planning} ")
        print_policy(agent, env, action_symbols, cliff_states, goal_state)

    plot_results(planning_steps_to_test, returns_list)
