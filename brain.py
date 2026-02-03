import torch
import torch.nn as nn


class NewbornBrain(nn.Module):
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=32):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        action_onehot = torch.zeros(self.action_dim, device=state.device, dtype=state.dtype)
        action_onehot[action] = 1.0
        x = torch.cat([state, action_onehot])
        return self.model(x)
