import torch
import torch.nn as nn
import numpy as np
from memory import SleepMemory


class NewbornBrain(nn.Module):
    def __init__(self, state_dim=2, action_dim=6, hidden_dim=32, map_size=14):
        super().__init__()

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.map_size   = map_size

        # ---------- World Model ----------
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # ---------- Emotional State ----------
        self.pleasure  = 0.0
        self.stress    = 0.0
        self.curiosity = 1.0

        # ---------- Expected Values ----------
        self.expected_pleasure = 0.0
        self.expected_stress   = 0.0

        # ---------- Drives ----------
        self.curiosity_drive  = 1.0
        self.safety_drive     = 1.0
        self.attachment_drive = 1.0

        # ---------- Attention ----------
        self.attention = 1.0
        self.salience  = 0.0

        # ---------- Memory ----------
        self.working_memory    = []
        self.long_term_memory  = SleepMemory(capacity=1000)

        # ---------- Spatial Emotional Map (14×14 for 7×7 world) ----------
        self.pleasure_map = np.zeros((map_size, map_size))
        self.stress_map   = np.zeros((map_size, map_size))
        self.map_counts   = np.zeros((map_size, map_size))

        # ---------- Sleep ----------
        self.sleep_interval   = 500
        self.last_sleep_step  = 0

        # ---------- Reward Prediction Error ----------
        self.rpe_pleasure = 0.0
        self.rpe_stress   = 0.0

        # ---------- Planning ----------
        self.lookahead_depth = 3
        self.lookahead_gamma = 0.9

        # ---------- Habituation ----------
        self.source_fatigue = {
            "mother":      0.0,
            "father":      0.0,
            "grandmother": 0.0,
            "stranger":    0.0,
            "danger":      0.0,
            "loud_sound":  0.0,
            "music":       0.0,
            "new_toy":     0.0,
            "rain":        0.0,
            "animal":      0.0,
            "none":        0.0,
        }
        self.FATIGUE_ALPHA = 0.05
        self.FATIGUE_DECAY = 0.005

        # ---------- Day-13: Separation Anxiety ----------
        self.last_seen = {
            "mother": 0, "father": 0, "grandmother": 0, "stranger": 0
        }
        self.separation_anxiety = 0.0

        # ---------- Day-15: Energy ----------
        self.energy = 1.0

        # ---------- Day-15: Mood ----------
        self.mood = 0.0

        # ---------- Day-18: Trust / Reputation ----------
        self.entity_trust = {
            "mother": 0.5, "father": 0.5, "grandmother": 0.5, "stranger": 0.5
        }
        self.entity_encounter_count = {
            "mother": 0, "father": 0, "grandmother": 0, "stranger": 0
        }

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
        for i in range(4):
            probs[i] += movement_boost

        probs += 0.3 * self.curiosity

        # Day-15: movement less likely when tired
        if self.energy < 0.3:
            for i in range(4):
                probs[i] *= (self.energy / 0.3)

        # Day-16: signal more likely when stressed
        probs[5] = self.stress * 0.5

        probs = probs / probs.sum()
        return torch.multinomial(probs, 1).item()

    # --------------------------------------------------

    def state_to_index(self, state):
        x = max(0.0, min(1.0, state[0].item()))
        y = max(0.0, min(1.0, state[1].item()))
        ix = int(x * (self.map_size - 1))
        iy = int(y * (self.map_size - 1))
        return ix, iy

    # --------------------------------------------------

    def update_spatial_map(self, state, pleasure, stress):
        x, y = self.state_to_index(state)
        self.map_counts[x, y] += 1
        alpha = 0.2
        self.pleasure_map[x, y] = (1 - alpha) * self.pleasure_map[x, y] + alpha * pleasure
        self.stress_map[x, y]   = (1 - alpha) * self.stress_map[x, y]   + alpha * stress

    def get_map_expectation(self, state):
        x, y = self.state_to_index(state)
        return self.pleasure_map[x, y], self.stress_map[x, y]

    # --------------------------------------------------
    # Habituation

    def get_habituation_factor(self, source):
        return 1.0 - self.source_fatigue.get(source, 0.0)

    def update_fatigue(self, source):
        for key in self.source_fatigue:
            self.source_fatigue[key] *= (1.0 - self.FATIGUE_DECAY)
        if source in self.source_fatigue:
            f = self.source_fatigue[source]
            self.source_fatigue[source] = f + self.FATIGUE_ALPHA * (1.0 - f)

    # --------------------------------------------------
    # Salience / Attention

    def compute_salience(self, prediction_error, pleasure, stress):
        novelty          = prediction_error.item()
        emotion_strength = abs(pleasure) + abs(stress)
        rpe_signal       = abs(self.rpe_pleasure) + abs(self.rpe_stress)
        self.salience  = 0.4 * novelty + 0.3 * emotion_strength + 0.3 * rpe_signal
        self.attention = 0.8 * self.attention + 0.2 * self.salience

    # --------------------------------------------------
    # Day-13: Separation Anxiety

    def update_separation_anxiety(self, present_entities, step):
        for name in ["mother", "father", "grandmother"]:
            if name in present_entities:
                prev_seen        = self.last_seen[name]
                absence_duration = step - prev_seen
                # Reunion pleasure spike after significant absence
                if prev_seen > 0 and absence_duration > 50:
                    self.pleasure += 0.3 * min(1.0, absence_duration / 200.0)
                self.last_seen[name] = step

        steps_since_mother      = step - self.last_seen["mother"]
        self.separation_anxiety = min(1.0, steps_since_mother / 300.0)
        self.stress += 0.15 * self.separation_anxiety

    # --------------------------------------------------
    # Day-14: Secure Base

    def compute_secure_base_factor(self, proximity):
        return proximity.get("mother", 0.0)

    # --------------------------------------------------
    # Day-18: Trust Update

    def update_trust(self, source, pleasure, stress):
        if source in self.entity_trust:
            self.entity_encounter_count[source] += 1
            n = self.entity_encounter_count[source]
            net_reward = pleasure - stress
            self.entity_trust[source] += (net_reward - self.entity_trust[source]) / n

    # --------------------------------------------------
    # Planning helpers

    def _compute_state_value(self, ep, es):
        return (
            self.attachment_drive * ep
            - self.safety_drive   * es
            + self.curiosity_drive * self.curiosity
            + self.attention      * (ep - es)
        )

    def _rollout(self, state, depth, gamma):
        if depth == 0:
            return 0.0
        best = -1e9
        for action in range(self.action_dim):
            pred = self.forward(state, action)
            ep, es = self.get_map_expectation(pred)
            v = self._compute_state_value(ep, es) + gamma * self._rollout(pred, depth - 1, gamma)
            if v > best:
                best = v
        return best

    # --------------------------------------------------
    # Goal-directed decision

    def choose_action_with_prediction(self, state, proximity=None):
        if proximity is None:
            proximity = {}

        secure_base    = self.compute_secure_base_factor(proximity)
        energy_penalty = max(0.0, 0.3 - self.energy) * 3.0  # ramps from 0 → ~0.9 as energy → 0

        best_action = 0
        best_value  = -1e9

        with torch.no_grad():
            for action in range(self.action_dim):
                pred_state = self.forward(state, action)

                # Depth-1: rich lookup (LTM + spatial map)
                ep_mem, es_mem = self.long_term_memory.get_state_expectation(pred_state)
                ep_map, es_map = self.get_map_expectation(pred_state)
                ep = 0.5 * ep_mem + 0.5 * ep_map
                es = 0.5 * es_mem + 0.5 * es_map

                # Day-18: trust prior — trusted nearby entities boost expected pleasure
                trust_bonus = sum(
                    self.entity_trust.get(name, 0.5) * prox
                    for name, prox in proximity.items()
                    if name in self.entity_trust
                ) * 0.2
                ep += trust_bonus

                # Day-14: secure base dampens expected stress near mother
                es_effective = es * (1.0 - 0.5 * secure_base)

                immediate = self._compute_state_value(ep, es_effective)

                # Day-15: penalise movement when tired
                if action < 4:
                    immediate -= energy_penalty

                # Depths 2+: fast map-only rollout
                future = self._rollout(pred_state, self.lookahead_depth - 1, self.lookahead_gamma)
                value  = immediate + self.lookahead_gamma * future

                if value > best_value:
                    best_value  = value
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
        step,
        proximity=None,
        present_entities=None,
        time_of_day=0.0,
    ):
        if proximity is None:
            proximity = {}
        if present_entities is None:
            present_entities = set()

        # Day-14: secure base — dampen incoming stress near mother
        secure_base = self.compute_secure_base_factor(proximity)
        stress      = stress * (1.0 - 0.5 * secure_base)

        # Habituation
        hab_factor   = self.get_habituation_factor(source)
        self.update_fatigue(source)
        hab_pleasure = pleasure * hab_factor
        hab_stress   = stress   * hab_factor

        # Core emotion update
        self.pleasure = 0.95 * self.pleasure + hab_pleasure
        self.stress   = 0.95 * self.stress   + hab_stress

        # Spatial map stores raw values
        self.update_spatial_map(state, pleasure, stress)

        # Memory-based expectations
        ep_state, es_state = self.long_term_memory.get_state_expectation(state)
        ep_src,   es_src   = self.long_term_memory.get_source_expectation(source)
        ep = 0.5 * ep_state + 0.5 * ep_src
        es = 0.5 * es_state + 0.5 * es_src

        # RPE — computed before updating expectations
        self.rpe_pleasure = pleasure - self.expected_pleasure
        self.rpe_stress   = stress   - self.expected_stress

        self.expected_pleasure += 0.1 * (ep - self.expected_pleasure)
        self.expected_stress   += 0.1 * (es - self.expected_stress)

        self.pleasure += 0.1 * self.expected_pleasure
        self.stress   += 0.1 * self.expected_stress
        self.pleasure += 0.2 * max(0.0, self.rpe_pleasure)
        self.stress   += 0.1 * max(0.0, -self.rpe_pleasure)

        # Curiosity — boosted near secure base (Day-14)
        # Multiplier applies only to the new contribution; the leak must stay < 1
        # so curiosity decays in the absence of new surprise.
        curiosity_multiplier = 1.0 + 0.5 * secure_base
        curiosity_input = (
            prediction_error.item()
            + abs(es - stress)
            + 0.5 * abs(self.rpe_pleasure)
        ) * curiosity_multiplier
        self.curiosity = 0.85 * self.curiosity + curiosity_input
        self.curiosity = min(self.curiosity, 10.0)

        # Drives
        self.curiosity_drive  = 1.0 + self.curiosity
        self.safety_drive     = 1.0 + self.stress
        self.attachment_drive = 1.0 + self.pleasure

        # Salience / attention
        self.compute_salience(prediction_error, pleasure, stress)

        # Day-13: separation anxiety + reunion bonus
        self.update_separation_anxiety(present_entities, step)

        # Day-15: energy — depletes on movement, restores with circadian rate
        if action < 4:
            self.energy = max(0.0, self.energy - 0.002)
        else:
            restore_rate = 0.001 * (1.0 + 0.5 * (1.0 - time_of_day))
            self.energy  = min(1.0, self.energy + restore_rate)

        # Day-15: mood — slow EMA of emotional balance
        self.mood = 0.995 * self.mood + 0.005 * (self.pleasure - self.stress)

        # Day-18: trust update from this encounter
        self.update_trust(source, pleasure, stress)

        self.working_memory.append(
            (state.detach().clone(), pleasure, stress, source, step)
        )

        if step - self.last_sleep_step >= self.sleep_interval:
            self.sleep(step)

    # --------------------------------------------------
    # Sleep consolidation

    def sleep(self, step):
        for state, pleasure, stress, source, t in self.working_memory:
            self.long_term_memory.store_experience(state, pleasure, stress, source, t)

        self.long_term_memory.sleep_and_consolidate(step)
        self.working_memory.clear()

        self.stress    *= 0.5
        self.curiosity *= 0.7
        self.attention *= 0.7
        self.energy     = 1.0  # Day-15: sleep fully restores energy

        for key in self.source_fatigue:
            self.source_fatigue[key] *= 0.5

        self.last_sleep_step = step
