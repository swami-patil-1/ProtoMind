import torch
import torch.nn as nn


class NewbornBrain(nn.Module):
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=32):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # World model
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Proto-hormones
        self.pleasure = 0.0
        self.stress = 0.0
        self.curiosity = 1.0

        # Salient memory (very limited)
        self.salient_memories = []  # list of (state, action, signal)

        # Action preference (habit seeds)
        self.action_bias = torch.zeros(action_dim)

        # Thresholds
        self.pleasure_threshold = 0.8
        self.stress_threshold = 0.8

    def forward(self, state, action):
        action_onehot = torch.zeros(self.action_dim, device=state.device)
        action_onehot[action] = 1.0
        x = torch.cat([state, action_onehot])
        return self.model(x)

    def update_internal_state(self, reward, pain, prediction_error, state, action):
        # Update hormones
        self.pleasure = 0.95 * self.pleasure + reward
        self.stress = 0.95 * self.stress + pain
        self.curiosity = 0.9 * self.curiosity + prediction_error.item()

        # Store salient memory
        if self.pleasure > self.pleasure_threshold:
            self._store_memory(state, action, signal=+1)
            self.action_bias[action] += 0.1

        if self.stress > self.stress_threshold:
            self._store_memory(state, action, signal=-1)
            self.action_bias[action] -= 0.1

        # decay bias slowly
        self.action_bias *= 0.999

    def _store_memory(self, state, action, signal):
        if len(self.salient_memories) > 50:
            self.salient_memories.pop(0)
        self.salient_memories.append(
            (state.detach().clone(), action, signal)
        )
