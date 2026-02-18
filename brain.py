import torch
import torch.nn as nn
from memory import SleepMemory


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

        # Internal states
        self.pleasure = 0.0
        self.stress = 0.0
        self.curiosity = 1.0

        # Anticipation
        self.expected_pleasure = 0.0
        self.expected_stress = 0.0

        # Memory
        self.working_memory = []
        self.long_term_memory = SleepMemory(capacity=1000)

        # Sleep control
        self.sleep_interval = 500
        self.last_sleep_step = 0

    def forward(self, state, action):
        action_onehot = torch.zeros(self.action_dim, device=state.device)
        action_onehot[action] = 1.0
        return self.model(torch.cat([state, action_onehot]))

    # ----------- Weak Emotion Reflex -----------
    def get_reflex_action(self):
        probs = torch.ones(self.action_dim)

        # Stress increases probability of staying (freeze tendency)
        probs[4] += 2.0 * self.expected_stress

        # Pleasure increases movement tendency
        movement_boost = 0.5 * self.expected_pleasure
        for i in range(self.action_dim - 1):
            probs[i] += movement_boost

        # Curiosity increases overall randomness
        probs += 0.3 * self.curiosity

        probs = probs / probs.sum()

        return torch.multinomial(probs, 1).item()

    # ----------- Internal Update -----------
    def update_internal_state(
        self,
        pleasure,
        stress,
        source,
        prediction_error,
        state,
        action,
        step
    ):
        # Hormone updates
        self.pleasure = 0.95 * self.pleasure + pleasure
        self.stress = 0.95 * self.stress + stress

        # State-based anticipation
        ep_state, es_state = self.long_term_memory.get_state_expectation(state)

        # Source-based anticipation
        ep_src, es_src = self.long_term_memory.get_source_expectation(source)

        ep = 0.5 * ep_state + 0.5 * ep_src
        es = 0.5 * es_state + 0.5 * es_src

        self.expected_pleasure += 0.1 * (ep - self.expected_pleasure)
        self.expected_stress += 0.1 * (es - self.expected_stress)

        # Anticipation influences baseline
        self.pleasure += 0.1 * self.expected_pleasure
        self.stress += 0.1 * self.expected_stress

        # Curiosity
        self.curiosity = (
            0.85 * self.curiosity
            + prediction_error.item()
            + abs(es - stress)
        )

        # Store in working memory
        self.working_memory.append(
            (state.detach().clone(), pleasure, stress, source, step)
        )

        if step - self.last_sleep_step >= self.sleep_interval:
            self.sleep(step)

    # ----------- Sleep -----------
    def sleep(self, step):
        for state, pleasure, stress, source, t in self.working_memory:
            self.long_term_memory.store_experience(
                state, pleasure, stress, source, t
            )

        self.long_term_memory.sleep_and_consolidate(step)
        self.working_memory.clear()

        # Emotional regulation
        self.stress *= 0.5
        self.curiosity *= 0.7

        self.last_sleep_step = step
