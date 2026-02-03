import torch
import torch.nn as nn

class NewbornBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, state, action):
        action_onehot = torch.zeros(5)
        action_onehot[action] = 1.0

        x = torch.cat([state, action_onehot])
        return self.model(x)
