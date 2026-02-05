import torch
import torch.nn as nn
from memory import SleepMemory


class NewbornBrain(nn.Module):
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=32):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # world model
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # internal states
        self.pleasure = 0.0
        self.stress = 0.0
        self.curiosity = 1.0

        # memory systems
        self.working_memory = []      # temporary (awake)
        self.long_term_memory = SleepMemory(capacity=50)

        # sleep control
        self.sleep_interval = 500
        self.last_sleep_step = 0

    def forward(self, state, action):
        action_onehot = torch.zeros(self.action_dim, device=state.device)
        action_onehot[action] = 1.0
        return self.model(torch.cat([state, action_onehot]))

    def update_internal_state(
        self,
        pleasure,
        stress,
        prediction_error,
        state,
        action,
        step
    ):
        # hormone updates
        self.pleasure = 0.95 * self.pleasure + pleasure
        self.stress = 0.95 * self.stress + stress
        self.curiosity = 0.9 * self.curiosity + prediction_error.item()

        # store in working memory (awake experience)
        self.working_memory.append(
            (state.detach().clone(), action, pleasure, stress, step)
        )

        # sleep trigger
        if step - self.last_sleep_step >= self.sleep_interval:
            self.sleep(step)

    def sleep(self, step):
        # consolidate memories
        for state, action, pleasure, stress, t in self.working_memory:
            if pleasure > 1.0 or stress > 1.0:
                self.long_term_memory.store_experience(
                    state, action, pleasure, stress, t
                )

        self.long_term_memory.sleep_and_consolidate(step)

        # clear working memory
        self.working_memory.clear()
        self.last_sleep_step = step

        # sleep reduces stress & curiosity
        self.stress *= 0.5
        self.curiosity *= 0.7
