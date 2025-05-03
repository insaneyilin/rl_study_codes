import matplotlib.pyplot as plt
import numpy as np


class BernoulliMultiArmBandit:
    """
    A Bernoulli multi-armed bandit environment.
    Each arm has a fixed probability of giving a reward (1) or not (0).

    Args:
        num_arms (int): Number of arms (actions) in the bandit problem
    """

    def __init__(self, num_arms):
        # Randomly initialize the win probability for each arm (uniform between 0 and 1)
        self.arm_probs = np.random.uniform(size=num_arms)
        # Identify the best arm (for regret calculation)
        self.best_arm_idx = np.argmax(self.arm_probs)
        self.best_arm_prob = self.arm_probs[self.best_arm_idx]
        self.num_arms = num_arms

    def step(self, k):
        """
        Pull arm k and return the reward.

        Args:
            k (int): Index of the arm to pull

        Returns:
            int: 1 if win, 0 if lose
        """
        assert k < self.num_arms, "Arm index out of bounds"
        # Bernoulli trial - return 1 with probability arm_probs[k], else 0
        return int(np.random.rand() < self.arm_probs[k])


class MBASolver:
    """
    Base class for multi-armed bandit solvers.
    Implements common tracking functionality (regret, rewards, etc.)

    Args:
        bandit (BernoulliMultiArmBandit): The bandit environment to solve
    """

    def __init__(self, bandit: BernoulliMultiArmBandit):
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        # Track number of pulls for each arm
        self.num_pulls_each_arm = np.zeros(self.num_arms)
        # History of actions taken
        self.actions = []
        # Cumulative regret (difference between optimal and actual rewards)
        self.regret = 0.0
        # History of regret at each step
        self.regrets = []
        # Cumulative rewards obtained
        self.cumulative_rewards = 0.0
        # History of rewards at each step
        self.rewards = []
        # History of cumulative rewards at each step
        self.cumulative_rewards_history = []

    def update_regret(self, k):
        """
        Update the regret metrics after pulling arm k.

        Args:
            k (int): Index of the arm that was pulled
        """
        # Regret is the difference between the best possible reward and what we got
        self.regret += self.bandit.best_arm_prob - self.bandit.arm_probs[k]
        self.regrets.append(self.regret)

    def update_rewards(self, r):
        """
        Update the reward metrics after pulling an arm.

        Args:
            r (int): Reward received (0 or 1)
        """
        self.cumulative_rewards += r
        self.rewards.append(r)
        self.cumulative_rewards_history.append(self.cumulative_rewards)

    def run_one_step(self):
        """Execute one step of the bandit algorithm (to be implemented by subclasses)."""
        raise NotImplementedError

    def run(self, num_steps):
        """
        Run the bandit algorithm for a given number of steps.

        Args:
            num_steps (int): Number of steps to run the algorithm
        """
        for _ in range(num_steps):
            # Get the chosen arm and reward from the specific algorithm
            k, r = self.run_one_step()
            # Update tracking metrics
            self.num_pulls_each_arm[k] += 1
            self.actions.append(k)
            self.update_regret(k)
            self.update_rewards(r)


class EpsilonGreedy(MBASolver):
    """
    Epsilon-Greedy algorithm for multi-armed bandits.

    With probability ε, explores randomly (tries any arm).
    With probability 1-ε, exploits the current best known arm.

    Args:
        bandit: The multi-armed bandit environment
        epsilon (float): Probability of exploration (0-1)
        init_prob (float): Initial estimated reward for all arms
    """

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # Initialize estimates for each arm's reward probability
        self.estimates = np.array([init_prob] * self.bandit.num_arms)

    def run_one_step(self):
        """Execute one step of the epsilon-greedy algorithm."""
        # Exploration: choose a random arm
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.num_arms)
        # Exploitation: choose the arm with highest estimated reward
        else:
            k = np.argmax(self.estimates)

        # Get reward from environment
        r = self.bandit.step(k)

        # Update the estimate using incremental averaging:
        # NewEstimate = OldEstimate + (1/N) * (Reward - OldEstimate)
        self.estimates[k] += (1.0 / (self.num_pulls_each_arm[k] + 1) *
                              (r - self.estimates[k]))

        return k, r


class DecayEpsilonGreedy(MBASolver):
    """
    Decaying ε-Greedy algorithm for multi-armed bandits.

    The exploration probability ε decreases over time according to 1/t.
    This provides more exploration early on and more exploitation later.

    Args:
        bandit: The multi-armed bandit environment
        init_prob (float): Initial estimated reward for all arms (default: 0.1)
        min_epsilon (float): Minimum exploration probability (default: 0.002)
    """

    def __init__(self, bandit, init_prob=1.0, min_epsilon=0.002):
        super(DecayEpsilonGreedy, self).__init__(bandit)
        self.total_steps = 0
        # Initialize estimates conservatively to encourage exploration
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.min_epsilon = min_epsilon  # Floor for exploration probability

    def run_one_step(self):
        """Execute one step of the decaying epsilon-greedy algorithm."""
        self.total_steps += 1
        # Calculate decaying epsilon (1/t but not below min_epsilon)
        epsilon = max(1.0 / self.total_steps, self.min_epsilon)

        # Exploration vs Exploitation
        if np.random.random() < epsilon:
            # Explore: choose random arm
            k = np.random.randint(0, self.bandit.num_arms)
        else:
            # Exploit: choose best current estimate
            k = np.argmax(self.estimates)

        # Get reward from environment
        r = self.bandit.step(k)

        # Update estimate incrementally
        learning_rate = 1.0 / (self.num_pulls_each_arm[k] + 1)
        self.estimates[k] += learning_rate * (r - self.estimates[k])

        return k, r


class UCB(MBASolver):
    """
    Upper Confidence Bound (UCB) algorithm for multi-armed bandits.

    Selects arms based on an upper confidence bound that balances:
    - Current reward estimate (exploitation)
    - Uncertainty in the estimate (exploration)

    Args:
        bandit: The multi-armed bandit environment
        coef (float): Exploration coefficient (default: 0.5)
        init_prob (float): Initial estimated reward for all arms (default: 1.0)
    """

    def __init__(self, bandit, coef=0.5, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0  # Total number of pulls across all arms
        # Optimistic initialization encourages exploration
        self.estimates = np.array([init_prob] * self.bandit.num_arms)
        self.coef = coef  # Controls exploration-exploitation tradeoff

    def run_one_step(self):
        """Execute one step of the UCB algorithm."""
        self.total_count += 1

        # Calculate UCB: Q(a) + c * sqrt(ln(t)/(2*N(a)))
        # Where:
        # - Q(a) is current reward estimate
        # - c is exploration coefficient
        # - t is total steps
        # - N(a) is pulls for this arm
        exploration_bonus = self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.num_pulls_each_arm + 1)))
        ucb = self.estimates + exploration_bonus

        # Select arm with highest UCB
        k = np.argmax(ucb)

        # Get reward from environment
        r = self.bandit.step(k)

        # Update estimate incrementally
        learning_rate = 1.0 / (self.num_pulls_each_arm[k] + 1)
        self.estimates[k] += learning_rate * (r - self.estimates[k])

        return k, r


class ThompsonSampling(MBASolver):
    """
    Thompson Sampling algorithm for multi-armed bandits.

    Uses Bayesian approach - maintains a distribution over possible reward probabilities
    for each arm, and samples from these distributions to select actions.
    """

    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        # Beta distribution parameters (α, β) for each arm
        # Start with uniform prior (α=1, β=1)
        self._a = np.ones(self.bandit.num_arms)  # Success counts + 1
        self._b = np.ones(self.bandit.num_arms)  # Failure counts + 1

    def run_one_step(self):
        """Execute one step of the Thompson Sampling algorithm."""
        # Sample reward probability for each arm from its Beta distribution
        samples = np.random.beta(self._a, self._b)
        # Select arm with highest sampled value
        k = np.argmax(samples)
        # Get reward from environment
        r = self.bandit.step(k)

        # Update Beta distribution parameters based on observed reward
        self._a[k] += r  # Increment success count
        self._b[k] += 1 - r  # Increment failure count

        return k, r


def plot_results(solvers, solver_names):
    """
    Plot the cumulative regrets and rewards of different bandit algorithms.

    Args:
        solvers (list): List of solver objects
        solver_names (list): List of solver names for legend
    """
    plt.figure(figsize=(14, 6))

    # Plot cumulative regrets
    plt.subplot(1, 2, 1)
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Cumulative regret", fontsize=12)
    plt.title(f"{solvers[0].bandit.num_arms}-armed bandit: Regret Comparison",
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot cumulative rewards
    plt.subplot(1, 2, 2)
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.cumulative_rewards_history))
        plt.plot(time_list,
                 solver.cumulative_rewards_history,
                 label=solver_names[idx])

    plt.xlabel("Time steps", fontsize=12)
    plt.ylabel("Cumulative rewards", fontsize=12)
    plt.title(f"{solvers[0].bandit.num_arms}-armed bandit: Reward Comparison",
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def compare_algorithms(num_arms=10, num_steps=5000):
    """
    Compare different bandit algorithms on the same problem.

    Args:
        num_arms (int): Number of arms in the bandit problem
        num_steps (int): Number of steps to run each algorithm
    """
    # For reproducible results
    np.random.seed(42)
    # seed 2 produces strange results, DecayEpsilonGreedy is not working as expected,
    # it collapses to linear increasing regret
    # np.random.seed(2)

    # Create bandit environment
    bandit = BernoulliMultiArmBandit(num_arms)

    # Initialize solvers
    epsilon_greedy = EpsilonGreedy(bandit, epsilon=0.1)
    decay_epsilon = DecayEpsilonGreedy(bandit)
    ucb = UCB(bandit, coef=1.0)
    thompson = ThompsonSampling(bandit)

    # Run all algorithms
    solvers = [epsilon_greedy, decay_epsilon, ucb, thompson]
    names = [
        "Epsilon-Greedy (ε=0.1)",
        "Decaying ε-Greedy",
        "UCB (c=1)",
        "Thompson Sampling",
    ]

    for solver in solvers:
        solver.run(num_steps)

    # Print final regrets and rewards
    print("Final cumulative regrets and rewards:")
    for name, solver in zip(names, solvers):
        print(f"{name}:")
        print(f"  Regret: {solver.regret:.1f}")
        print(
            f"  Total rewards: {solver.cumulative_rewards} ({solver.cumulative_rewards/num_steps*100:.1f}% success rate)"
        )

    # Plot results
    plot_results(solvers, names)


if __name__ == "__main__":
    compare_algorithms(num_arms=10, num_steps=5000)
