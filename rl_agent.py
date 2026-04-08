from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_features
from neural_network import N_FEATURES
from player import Player
from snake import Action

ACTIONS = list(Action) 
N_ACTIONS = len(ACTIONS)


class Basic_RL_Agent(nn.Module):

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, hidden)
        #self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        return self.fc3(x)  


class ReplayBuffer:

    def __init__(self, capacity: int = 50_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(self, state: list[float], action: int, reward: float, next_state: list[float], done: bool) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:

    def __init__(
        self,
        *,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str | None = None,
    ) -> None:
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._steps = 0
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online = Basic_RL_Agent(hidden).to(self.device)
        self.target = Basic_RL_Agent(hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: list[float], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.online(t).argmax(dim=1).item())

    def train_step(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = zip(*self.buffer.sample(self.batch_size))

        states_t      = torch.tensor(states,      dtype=torch.float32).to(self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32).to(self.device)

        q_values = self.online(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            # Double DQN: online selects action, target evaluates it
            next_actions = self.online(next_states_t).argmax(dim=1, keepdim=True)
            next_q   = self.target(next_states_t).gather(1, next_actions).squeeze(1)
            targets  = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # Huber loss — doesn't square large errors, much more stable than MSE
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        torch.save({"online": self.online.state_dict(), "steps": self._steps}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
        self._steps = ckpt.get("steps", 0)


class RLPlayer(Player):
    """Greedy inference player — uses trained DQNAgent at epsilon=0."""

    def __init__(self, player_id: int, agent: DQNAgent) -> None:
        self.player_id = player_id
        self._agent = agent

    def get_action(self, game_state: dict[str, Any]) -> Action:
        features = get_features(game_state, self.player_id)
        if not features:
            return Action.STRAIGHT
        return ACTIONS[self._agent.select_action(features, epsilon=0.0)]
