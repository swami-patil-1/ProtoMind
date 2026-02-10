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

        # anticipation (state-dependent)
        self.expected_pleasure = 0.0
        self.expected_stress = 0.0

        # memory
        self.working_memory = []
        self.long_term_memory = SleepMemory(capacity=1000)

        # sleep control
        self.sleep_interval = 500
        self.last_sleep_step = 0

        # peak tracking
        self.peak_pleasure = 0.0
        self.peak_stress = 0.0

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
        # update physiology
        self.pleasure = 0.95 * self.pleasure + pleasure
        self.stress = 0.95 * self.stress + stress

        # state-dependent anticipation
        ep, es = self.long_term_memory.get_state_expectation(state)
        self.expected_pleasure += 0.1 * (ep - self.expected_pleasure)
        self.expected_stress += 0.1 * (es - self.expected_stress)

        # anticipation affects baseline emotion
        self.pleasure += 0.1 * self.expected_pleasure
        self.stress += 0.1 * self.expected_stress

        # curiosity from surprise + mismatch
        self.curiosity = (
            0.85 * self.curiosity
            + prediction_error.item()
            + abs(es - stress)
        )

        # track peaks
        self.peak_pleasure = max(self.peak_pleasure, pleasure)
        self.peak_stress = max(self.peak_stress, stress)

        # store *everything* (low intensity allowed)
        self.working_memory.append(
            (state.detach().clone(), pleasure, stress, step)
        )

        if step - self.last_sleep_step >= self.sleep_interval:
            self.sleep(step)

    def sleep(self, step):
        # consolidate ALL experiences, intensity handled by scoring
        for state, pleasure, stress, t in self.working_memory:
            self.long_term_memory.store_experience(
                state, pleasure, stress, t
            )

        self.long_term_memory.sleep_and_consolidate(step)

        # reset awake buffers
        self.working_memory.clear()
        self.peak_pleasure = 0.0
        self.peak_stress = 0.0

        # sleep regulation
        self.stress *= 0.5
        self.curiosity *= 0.7

        self.last_sleep_step = step
