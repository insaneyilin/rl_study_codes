import copy

import matplotlib.pyplot as plt
import numpy as np


class CliffWalkingEnv:
    """A cliff walking environment simulation.

    This is a grid world environment where an agent needs to navigate from the
    start position (bottom-left corner) to the goal position (bottom-right corner)
    while avoiding falling off the cliff (middle cells at the bottom row).

    Attributes:
        ncol (int): Number of columns in the grid
        nrow (int): Number of rows in the grid
        start_state (int): Starting state index (bottom-left corner)
        goal_state (int): Goal state index (bottom-right corner)
        cliff_states (list): List of cliff state indices
        trans_prob_mat (list): State transition probability matrix
    """

    def __init__(self, ncol=12, nrow=4):
        """Initialize the cliff walking environment.

        Args:
            ncol (int, optional): Number of columns in the grid. Defaults to 12.
            nrow (int, optional): Number of rows in the grid. Defaults to 4.
        """
        self.ncol = ncol  # Number of columns
        self.nrow = nrow  # Number of rows

        # Bottom left.
        self.start_loc = (self.nrow - 1, 0)  # (row_idx, col_idx)
        # Bottom right
        self.goal_loc = (self.nrow - 1, self.ncol - 1)  # (row_idx, col_idx)

        # State transition probability matrix
        # trans_prob_mat[state][action] = [(prob, next_state, reward, done)]
        self.trans_prob_mat = self._create_trans_prob_mat()

    def _2d_loc_to_1d_idx(self, row, col):
        return row * self.ncol + col

    def _is_in_cliff(self, row, col):
        """Check if the given row and column are in the cliff."""
        return (row == self.nrow - 1) and (col > 0) and (col < self.ncol - 1)

    def _is_in_goal(self, row, col):
        """Check if the given row and column are in the goal."""
        return (row, col) == self.goal_loc

    def goal_state(self):
        return self._2d_loc_to_1d_idx(self.goal_loc[0], self.goal_loc[1])

    def cliff_states(self):
        return [
            self._2d_loc_to_1d_idx(self.nrow - 1, col)
            for col in range(1, self.ncol - 1)
        ]

    def _create_trans_prob_mat(self):
        """Create the state transition probability matrix.

        The matrix defines for each state and action:
        - The probability of transitioning to next state
        - The reward received
        - Whether the episode terminates

        Returns:
            list: A 3D list where trans_prob_mat[state][action] =
                  [(probability, next_state, reward, done)]
        """
        # Initialize transition probability matrix: [nrow*ncol][4 actions]
        trans_prob_mat = [[[] for _ in range(4)]
                          for _ in range(self.nrow * self.ncol)]

        # Action mappings (delta_r, delta_c): [up, down, left, right]
        action_deltas = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        for r in range(self.nrow):
            for c in range(self.ncol):
                current_state = self._2d_loc_to_1d_idx(r, c)
                for action in range(4):
                    # If current state is cliff or goal, stay with 0 reward
                    if self._is_in_cliff(r, c) or self._is_in_goal(r, c):
                        prob = 1.0
                        next_state = current_state
                        reward = 0.0
                        done = True
                        trans_prob_mat[current_state][action] = [
                            (prob, next_state, reward, done)
                        ]
                        continue

                    # Calculate next position with boundary checks
                    next_r = min(self.nrow - 1,
                                 max(0, r + action_deltas[action][0]))
                    next_c = min(self.ncol - 1,
                                 max(0, c + action_deltas[action][1]))
                    next_state = self._2d_loc_to_1d_idx(next_r, next_c)

                    # Default reward is -1 (per-step cost)
                    reward = -1.0
                    done = False

                    # Check if next state is cliff or goal
                    if self._is_in_cliff(next_r, next_c):
                        done = True
                        reward = -100.0
                    elif self._is_in_goal(next_r, next_c):
                        done = True

                    # Transition probability is deterministic (1.0)
                    prob = 1.0
                    trans_prob_mat[current_state][action] = [(prob, next_state,
                                                              reward, done)]
        return trans_prob_mat

    def print_env(self):
        """Print a text representation of the environment.

        The representation shows:
        - S: Start position (bottom-left)
        - G: Goal position (bottom-right)
        - C: Cliff positions
        - .: Safe positions
        """
        print("Cliff Walking Environment Visualization:")
        print("S: Start, G: Goal, C: Cliff, .: Safe Position")
        print(f"Grid Layout: {self.nrow}x{self.ncol} (row x col)")
        grid = [["." for _ in range(self.ncol)] for _ in range(self.nrow)]
        grid[self.start_loc[0]][self.start_loc[1]] = "S"
        grid[self.goal_loc[0]][self.goal_loc[1]] = "G"
        for c in range(1, self.ncol - 1):
            grid[self.nrow - 1][c] = "C"
        for row in grid:
            print(" ".join(row))
        print()


class PolicyIterationCliffWalking:
    """Policy Iteration algorithm for Cliff Walking environment.

    Implements the policy iteration algorithm which alternates between:
    1. Policy Evaluation: Computes the state-value function for current policy
    2. Policy Improvement: Updates the policy to be greedy with respect to the computed value function

    Attributes:
        env (CliffWalkingEnv): The environment
        state_values (list): Current estimate of state values (V(s))
        policy (list): Current policy (π(a|s)) as probability distribution over actions
        theta (float): Convergence threshold for policy evaluation
        gamma (float): Discount factor for future rewards
    """

    def __init__(self, env: CliffWalkingEnv, theta: float, gamma: float):
        """Initialize the policy iteration algorithm.

        Args:
            env: CliffWalking environment instance
            theta: Convergence threshold for policy evaluation
            gamma: Discount factor for future rewards
        """
        self.env = env
        self.state_values = [0] * (self.env.ncol * self.env.nrow
                                   )  # Initialize V(s) to 0 for all states
        # Initialize uniform random policy (equal probability for all actions)
        self.policy = [[0.25, 0.25, 0.25, 0.25]
                       for _ in range(self.env.ncol * self.env.nrow)]
        self.theta = theta  # Threshold for policy evaluation convergence
        self.gamma = gamma  # Discount factor

    def policy_evaluation(self) -> None:
        """Evaluate the current policy until convergence.

        Iteratively computes the state-value function for the current policy
        using Bellman expectation equations until the maximum change in values
        is below the threshold theta.
        """
        iteration_count = 1
        while True:
            max_value_change = 0
            new_state_values = [0] * (self.env.ncol * self.env.nrow)

            for state in range(self.env.ncol * self.env.nrow):
                # Calculate expected value for each action under current policy
                action_values = []
                for action in range(
                        4):  # 4 possible actions (up, down, left, right)
                    # Compute Q-value for this state-action pair
                    q_value = 0
                    for transition in self.env.trans_prob_mat[state][action]:
                        prob, next_state, reward, done = transition
                        # Bellman equation: Q(s,a) = Σ p(s',r|s,a)[r + γV(s')]
                        # `1- done` ensures we only consider V(s') if not terminal
                        q_value += prob * (reward + self.gamma *
                                           self.state_values[next_state] *
                                           (1 - done))
                    # Weight Q-value by policy probability π(a|s); V(s) = Σ π(a|s)Q(s,a)
                    action_values.append(self.policy[state][action] * q_value)

                # New state value is expected value over all actions
                new_state_values[state] = sum(action_values)
                max_value_change = max(
                    max_value_change,
                    abs(new_state_values[state] - self.state_values[state]),
                )

            self.state_values = new_state_values
            if max_value_change < self.theta:
                break  # Convergence reached
            iteration_count += 1

        print(
            f"Policy evaluation completed after {iteration_count} iterations")

    def policy_improvement(self) -> list:
        """Improve the policy by making it greedy with respect to current value function.

        Returns:
            The new improved policy
        """
        for state in range(self.env.nrow * self.env.ncol):
            # Compute Q-values for all actions in current state
            action_q_values = []
            for action in range(4):
                q_value = 0
                for transition in self.env.trans_prob_mat[state][action]:
                    prob, next_state, reward, done = transition
                    # Bellman equation: Q(s,a) = Σ p(s',r|s,a)[r + γV(s')]
                    # `1- done` ensures we only consider V(s') if not terminal
                    q_value += prob * (
                        reward + self.gamma * self.state_values[next_state] *
                        (1 - done))
                action_q_values.append(q_value)

            # Find the best action(s) with maximum Q-value
            max_q = max(action_q_values)
            best_action_count = action_q_values.count(max_q)

            # Update policy to be greedy (equal probability for all best actions, and 0 for others)
            self.policy[state] = [
                1 / best_action_count if q == max_q else 0
                for q in action_q_values
            ]

        print("Policy improvement completed")
        return self.policy

    def policy_iteration(self) -> None:
        """Perform policy iteration until policy converges.

        Alternates between policy evaluation and policy improvement until
        the policy no longer changes between iterations.
        """
        while True:
            self.policy_evaluation()
            old_policy = copy.deepcopy(self.policy)
            new_policy = self.policy_improvement()
            if old_policy == new_policy:
                break  # Policy has converged


class ValueIterationCliffWalking:
    """Value Iteration algorithm for Cliff Walking environment.

    Implements the value iteration algorithm which:
    1. Iteratively updates state values using Bellman optimality equations
    2. Extracts the optimal policy once value iteration converges

    Attributes:
        env (CliffWalkingEnv): The environment
        state_values (list): Current estimate of state values (V(s))
        theta (float): Convergence threshold for value iteration
        gamma (float): Discount factor for future rewards
        policy (list): Optimal policy derived from converged value function
    """

    def __init__(self, env, theta: float, gamma: float):
        """Initialize the value iteration algorithm.

        Args:
            env: CliffWalking environment instance
            theta: Convergence threshold for value iteration
            gamma: Discount factor for future rewards
        """
        self.env = env
        self.state_values = [0] * (self.env.ncol * self.env.nrow
                                   )  # Initialize V(s) to 0
        self.theta = theta  # Threshold for value convergence
        self.gamma = gamma  # Discount factor
        self.policy = [None] * (self.env.ncol * self.env.nrow
                                )  # Will store optimal policy

    def value_iteration(self) -> None:
        """Perform value iteration until convergence.

        Updates state values using Bellman optimality equation:
        V(s) = max_a [Σ p(s',r|s,a)(r + γV(s'))]

        Continues until maximum change in values is below theta.
        """
        iteration_count = 0
        while True:
            max_value_change = 0
            new_state_values = [0] * (self.env.ncol * self.env.nrow)

            for state in range(self.env.ncol * self.env.nrow):
                # Calculate Q-values for all actions in current state
                action_q_values = []
                for action in range(
                        4):  # 4 possible actions (up, down, left, right)
                    q_value = 0
                    for transition in self.env.trans_prob_mat[state][action]:
                        prob, next_state, reward, done = transition
                        # Bellman optimality equation for Q-value:
                        # Q(s,a) = Σ p(s',r|s,a)[r + γV(s')]
                        q_value += prob * (reward + self.gamma *
                                           self.state_values[next_state] *
                                           (1 - done))
                    action_q_values.append(q_value)

                # Value iteration update rule: V(s) = max_a Q(s,a)
                new_state_values[state] = max(action_q_values)
                max_value_change = max(
                    max_value_change,
                    abs(new_state_values[state] - self.state_values[state]),
                )

            self.state_values = new_state_values
            if max_value_change < self.theta:
                break  # Convergence reached
            iteration_count += 1

        print(f"Value iteration completed after {iteration_count} iterations")
        self._derive_policy()

    def _derive_policy(self) -> None:
        """Derive the optimal policy from the converged value function.

        The policy is greedy with respect to the optimal value function:
        π(s) = argmax_a Q(s,a)
        """
        for state in range(self.env.nrow * self.env.ncol):
            # Compute Q-values for all actions
            action_q_values = []
            for action in range(4):
                q_value = 0
                for transition in self.env.trans_prob_mat[state][action]:
                    prob, next_state, reward, done = transition
                    q_value += prob * (
                        reward + self.gamma * self.state_values[next_state] *
                        (1 - done))
                action_q_values.append(q_value)

            # Find the best action(s) with maximum Q-value
            max_q = max(action_q_values)
            best_action_count = action_q_values.count(max_q)

            # Create deterministic policy (equal probability for all best actions)
            self.policy[state] = [
                1 / best_action_count if q == max_q else 0
                for q in action_q_values
            ]


def print_agent(agent, action_meaning, disaster_states=[], goal_states=[]):
    """Print the agent's state values and policy in a grid format.

    Args:
        agent: The policy iteration agent
        action_meaning: Symbols representing each action
        disaster_states: List of states to mark as dangerous (e.g., cliff)
        goal_states: List of states to mark as goal
    """
    print("State Values:")
    for row in range(agent.env.nrow):
        for col in range(agent.env.ncol):
            state = row * agent.env.ncol + col
            print("%6.6s" % ("%.3f" % agent.state_values[state]), end=" ")
        print()

    print("\nPolicy:")
    for row in range(agent.env.nrow):
        for col in range(agent.env.ncol):
            state = row * agent.env.ncol + col
            if state in disaster_states:
                print("XXXX", end=" ")  # Mark disaster states
            elif state in goal_states:
                print("GGGG", end=" ")  # Mark goal states
            else:
                # Show action probabilities (symbols for actions with >0 probability)
                policy_str = ""
                for action_idx, prob in enumerate(agent.policy[state]):
                    policy_str += action_meaning[
                        action_idx] if prob > 0 else "o"
                print(policy_str, end=" ")
        print()


# Example usage
if __name__ == "__main__":
    env = CliffWalkingEnv(nrow=4, ncol=12)
    action_meaning = ["^", "v", "<", ">"]  # Up, Down, Left, Right
    theta = 0.001  # Convergence threshold
    gamma = 0.9  # Discount factor

    print("===================================")
    print("Policy Iteration Method")
    policy_iter_agent = PolicyIterationCliffWalking(env, theta, gamma)
    policy_iter_agent.policy_iteration()

    cliff_states = env.cliff_states()
    goal_state = [env.goal_state()]  # Bottom-right corner
    print_agent(policy_iter_agent, action_meaning, cliff_states, goal_state)

    print("===================================")
    print("Value Iteration Method")
    value_iter_agent = ValueIterationCliffWalking(env, theta, gamma)
    value_iter_agent.value_iteration()
    print_agent(value_iter_agent, action_meaning, cliff_states, goal_state)

    # Check if the policies are the same
    assert policy_iter_agent.policy == value_iter_agent.policy, "Policies do not match!"
    print("Policies match!")
    # Check if the state values are the same
    assert np.allclose(
        policy_iter_agent.state_values,
        value_iter_agent.state_values), "State values do not match!"
    print("State values match!")
