import torch
import torch.nn as nn
import numpy as np
from memory import SleepMemory


class NewbornBrain(nn.Module):
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=32, map_size=10):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.map_size = map_size

        # ---------- World Model ----------
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # ---------- Emotional State ----------
        self.pleasure = 0.0
        self.stress = 0.0
        self.curiosity = 1.0

        # ---------- Expected Values ----------
        self.expected_pleasure = 0.0
        self.expected_stress = 0.0

        # ---------- Drives ----------
        self.curiosity_drive = 1.0
        self.safety_drive = 1.0
        self.attachment_drive = 1.0

        # ---------- Attention ----------
        self.attention = 1.0
        self.salience = 0.0

        # ---------- Memory ----------
        self.working_memory = []
        self.long_term_memory = SleepMemory(capacity=1000)

        # ---------- Spatial Emotional Map ----------
        self.pleasure_map = np.zeros((map_size, map_size))
        self.stress_map = np.zeros((map_size, map_size))
        self.map_counts = np.zeros((map_size, map_size))

        # ---------- Sleep ----------
        self.sleep_interval = 500
        self.last_sleep_step = 0

    # --------------------------------------------------

    def forward(self, state, action):

        action_onehot = torch.zeros(self.action_dim, device=state.device)
        action_onehot[action] = 1.0

        return self.model(torch.cat([state, action_onehot]))

    # --------------------------------------------------
    # Weak reflex behaviour

    def get_reflex_action(self):

        probs = torch.ones(self.action_dim)

        probs[4] += 2.0 * self.expected_stress

        movement_boost = 0.5 * self.expected_pleasure

        for i in range(self.action_dim - 1):
            probs[i] += movement_boost

        probs += 0.3 * self.curiosity

        probs = probs / probs.sum()

        return torch.multinomial(probs, 1).item()

    # --------------------------------------------------
    # Convert state → map index

    def state_to_index(self, state):

        x = state[0].item()
        y = state[1].item()

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        ix = int(x * (self.map_size - 1))
        iy = int(y * (self.map_size - 1))

        return ix, iy

    # --------------------------------------------------
    # Update spatial map

    def update_spatial_map(self, state, pleasure, stress):

        x, y = self.state_to_index(state)

        self.map_counts[x, y] += 1

        alpha = 0.2

        self.pleasure_map[x, y] = (
            (1 - alpha) * self.pleasure_map[x, y] + alpha * pleasure
        )

        self.stress_map[x, y] = (
            (1 - alpha) * self.stress_map[x, y] + alpha * stress
        )

    # --------------------------------------------------
    # Query spatial map

    def get_map_expectation(self, state):

        x, y = self.state_to_index(state)

        ep = self.pleasure_map[x, y]
        es = self.stress_map[x, y]

        return ep, es

    # --------------------------------------------------
    # Salience computation

    def compute_salience(self, prediction_error, pleasure, stress):

        novelty = prediction_error.item()

        emotion_strength = abs(pleasure) + abs(stress)

        self.salience = 0.6 * novelty + 0.4 * emotion_strength

        # attention update
        self.attention = 0.8 * self.attention + 0.2 * self.salience

    # --------------------------------------------------
    # Goal-directed decision

    def choose_action_with_prediction(self, state):

        best_action = 0
        best_value = -1e9

        for action in range(self.action_dim):

            pred_state = self.forward(state, action)

            ep_mem, es_mem = self.long_term_memory.get_state_expectation(pred_state)
            ep_map, es_map = self.get_map_expectation(pred_state)

            ep = 0.5 * ep_mem + 0.5 * ep_map
            es = 0.5 * es_mem + 0.5 * es_map

            curiosity_bonus = self.curiosity_drive * self.curiosity
            safety_penalty = self.safety_drive * es
            attachment_bonus = self.attachment_drive * ep

            # Attention amplifies important states
            attention_boost = self.attention * (ep + es)

            value = (
                attachment_bonus
                - safety_penalty
                + curiosity_bonus
                + attention_boost
            )

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    # --------------------------------------------------
    # Internal state update

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

        self.pleasure = 0.95 * self.pleasure + pleasure
        self.stress = 0.95 * self.stress + stress

        self.update_spatial_map(state, pleasure, stress)

        ep_state, es_state = self.long_term_memory.get_state_expectation(state)
        ep_src, es_src = self.long_term_memory.get_source_expectation(source)

        ep = 0.5 * ep_state + 0.5 * ep_src
        es = 0.5 * es_state + 0.5 * es_src

        self.expected_pleasure += 0.1 * (ep - self.expected_pleasure)
        self.expected_stress += 0.1 * (es - self.expected_stress)

        self.pleasure += 0.1 * self.expected_pleasure
        self.stress += 0.1 * self.expected_stress

        # curiosity update
        self.curiosity = (
            0.85 * self.curiosity
            + prediction_error.item()
            + abs(es - stress)
        )

        # drives
        self.curiosity_drive = 1.0 + self.curiosity
        self.safety_drive = 1.0 + self.stress
        self.attachment_drive = 1.0 + self.pleasure

        # salience / attention
        self.compute_salience(prediction_error, pleasure, stress)

        self.working_memory.append(
            (state.detach().clone(), pleasure, stress, source, step)
        )

        if step - self.last_sleep_step >= self.sleep_interval:
            self.sleep(step)

    # --------------------------------------------------
    # Sleep consolidation

    def sleep(self, step):

        for state, pleasure, stress, source, t in self.working_memory:
            self.long_term_memory.store_experience(
                state, pleasure, stress, source, t
            )

        self.long_term_memory.sleep_and_consolidate(step)

        self.working_memory.clear()

        self.stress *= 0.5
        self.curiosity *= 0.7
        self.attention *= 0.7

        self.last_sleep_step = step