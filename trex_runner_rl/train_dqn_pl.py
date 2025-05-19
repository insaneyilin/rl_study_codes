import os
from collections import OrderedDict, deque, namedtuple
from typing import Iterator, List, Tuple

import numpy as np
# import pandas as pd
# import seaborn as sn
import torch
from dqn_model import DuelingDQN
# from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from trex_runner import TRexRunner

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:

    def __init__(self, env: TRexRunner, replay_buffer: ReplayBuffer) -> None:
        """Base Agent class handling the interaction with the environment.

        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences

        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.begin()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.begin()

    def get_action(self,
                   net: nn.Module,
                   epsilon: float,
                   device: str,
                   deterministic: bool = False) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device
            deterministic: if True, always take the action with the highest Q-value

        Returns:
            action

        """
        if not deterministic and np.random.random() < epsilon:
            action = self.env.get_random_action()
        else:
            state = torch.tensor([self.state])

            if device not in ["cpu"]:
                state = state.to(device)

            q_values = net(state.float())
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done

        """
        action = self.get_action(net, epsilon, device, deterministic)

        new_state, reward, done = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state

        if done:
            self.reset()

        return reward, done


class DuelingDQNLightning(LightningModule):

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 1e-4,
        hidden_dim: int = 512,
        gamma: float = 0.99,
        sync_rate: int = 100,
        replay_size: int = 2048,
        eps_last_frame: int = 10000,
        eps_start: float = 1.0,
        eps_end: float = 0.03,
        replay_buffer_sample_size: int = 200,
        warm_start_steps: int = 1000,
    ) -> None:
        """Basic DQN Model.

        Args:
            batch_size: size of the batches
            lr: learning rate
            hidden_dim: size of the hidden layer
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            replay_buffer_sample_size: how many samples to take from the replay buffer
            warm_start_steps: number of random steps to take before training

        """
        super().__init__()
        self.save_hyperparameters()
        self.env = TRexRunner()
        obs_shape = self.env.observation_shape()
        n_actions = self.env.num_actions()

        self.hidden_dim = self.hparams.hidden_dim
        self.net = DuelingDQN(obs_shape, self.hidden_dim, n_actions)
        self.target_net = DuelingDQN(obs_shape, self.hidden_dim, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        # self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with

        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values

        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss

        """
        states, actions, rewards, dones, next_states = batch

        # Compute current Q-values for taken actions; Q(s_t, a_t)
        state_action_values = self.net(states.float()).gather(
            1,
            actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states.float()).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        # TD target
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss based on
        the minibatch received.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics

        """
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.hparams.eps_start,
                                   self.hparams.eps_end,
                                   self.hparams.eps_last_frame)
        self.log("epsilon", epsilon)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({
            "reward": reward,
            "train_loss": loss,
        })
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer,
                            self.hparams.replay_buffer_sample_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    model = DuelingDQNLightning()

    accelerator = "cpu"
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "cuda"
    print(f"Using accelerator: {accelerator}")

    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=50000,
        val_check_interval=50,
        logger=CSVLogger(save_dir="logs/"),
    )

    trainer.fit(model)
