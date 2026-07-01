import numpy as np
import random


class RichWorld:
    def __init__(self, size=7):
        self.size = size

        # ---------- Environment ----------
        self.environment = "home"
        self.env_timer   = 0
        self.ENV_HOME_DURATION    = 300
        self.ENV_OUTSIDE_DURATION = 100
        self.ENV_OUTSIDE_CHANCE   = 0.4

        # ---------- Temporal (Day-17) ----------
        self.step_count = 0
        self.DAY_LENGTH = 500

        # ---------- Social Entities ----------
        self.entities = {
            "mother": {
                "pleasure": 1.0, "stress": 0.0,
                "home_pos":    np.array([0, 0]),
                "outside_pos": np.array([3, 3]),
                "home_arrive":    1.000, "home_depart":    0.000,
                "outside_arrive": 0.800, "outside_depart": 0.005,
                "present": True,
            },
            "father": {
                "pleasure": 0.8, "stress": 0.0,
                "home_pos":    np.array([0, 1]),
                "outside_pos": np.array([3, 4]),
                "home_arrive":    0.008, "home_depart":    0.004,
                "outside_arrive": 0.004, "outside_depart": 0.015,
                "present": True,
            },
            "grandmother": {
                "pleasure": 0.9, "stress": 0.0,
                "home_pos":    np.array([1, 0]),
                "outside_pos": np.array([2, 3]),
                "home_arrive":    0.003, "home_depart":    0.020,
                "outside_arrive": 0.001, "outside_depart": 0.050,
                "present": False,
            },
            "stranger": {
                "pleasure": 0.1, "stress": 0.5,
                "home_pos":    np.array([5, 5]),
                "outside_pos": np.array([6, 5]),
                "home_arrive":    0.001, "home_depart":    0.050,
                "outside_arrive": 0.020, "outside_depart": 0.080,
                "present": False,
            },
        }

        # ---------- Danger Zones ----------
        self.danger_pos = {
            "home":    np.array([6, 6]),
            "outside": np.array([6, 6]),
        }

        # ---------- Ambient Sensory Events ----------
        self.ambient_events = {
            "loud_sound": (0.0, 0.8),
            "music":      (0.6, 0.0),
            "new_toy":    (0.7, 0.0),
            "rain":       (0.0, 0.3),
            "animal":     (0.5, 0.2),
        }

        self.ambient_probs = {
            "home": {
                "loud_sound": 0.02,
                "music":      0.05,
                "new_toy":    0.03,
                "rain":       0.01,
                "animal":     0.01,
            },
            "outside": {
                "loud_sound": 0.05,
                "music":      0.01,
                "new_toy":    0.01,
                "rain":       0.06,
                "animal":     0.08,
            },
        }

        self.reset()

    # --------------------------------------------------

    def reset(self):
        self.pos = np.random.randint(0, self.size, size=2)
        self.step_count = 0
        return self._get_state()

    # --------------------------------------------------

    def _update_entities(self):
        env = self.environment
        for e in self.entities.values():
            if e["present"]:
                if random.random() < e[f"{env}_depart"]:
                    e["present"] = False
            else:
                if random.random() < e[f"{env}_arrive"]:
                    e["present"] = True

    # --------------------------------------------------

    def _switch_environment(self):
        if self.environment == "home":
            self.environment = "outside"
            self.pos = np.array([self.size // 2, self.size // 2])
        else:
            self.environment = "home"
            self.pos = np.random.randint(0, self.size, size=2)
        self.env_timer = 0

    # --------------------------------------------------
    # Day-14: soft proximity signal for each nearby entity

    def get_proximity_signals(self, radius=2):
        signals = {}
        env = self.environment
        for name, e in self.entities.items():
            if e["present"]:
                dist = int(np.abs(self.pos - e[f"{env}_pos"]).sum())
                if dist <= radius:
                    signals[name] = 1.0 - dist / (radius + 1)
        return signals

    # --------------------------------------------------
    # Day-13: which entities are anywhere in the current environment

    def get_present_entities(self):
        return {name for name, e in self.entities.items() if e["present"]}

    # --------------------------------------------------

    def step(self, action, signal_active=False):
        move = {
            0: np.array([-1,  0]),
            1: np.array([ 1,  0]),
            2: np.array([ 0, -1]),
            3: np.array([ 0,  1]),
            4: np.array([ 0,  0]),
            5: np.array([ 0,  0]),  # signal: stay in place (Day-16)
        }

        self.pos = np.clip(self.pos + move[action], 0, self.size - 1)
        self.step_count += 1

        self.env_timer += 1
        if self.environment == "home" and self.env_timer >= self.ENV_HOME_DURATION:
            if random.random() < self.ENV_OUTSIDE_CHANCE:
                self._switch_environment()
        elif self.environment == "outside" and self.env_timer >= self.ENV_OUTSIDE_DURATION:
            self._switch_environment()

        self._update_entities()

        pleasure = 0.0
        stress   = 0.0
        source   = "none"

        env = self.environment

        # Social entity at exact position
        for name, e in self.entities.items():
            if e["present"] and np.array_equal(self.pos, e[f"{env}_pos"]):
                pleasure += e["pleasure"]
                stress   += e["stress"]
                source    = name
                break

        # Danger zone
        if source == "none" and np.array_equal(self.pos, self.danger_pos[env]):
            stress += 1.0
            source  = "danger"

        # Day-16: signal response — caring entity comforts if present anywhere
        if action == 5 and signal_active and source == "none":
            for name in ["mother", "father"]:
                e = self.entities[name]
                if e["present"]:
                    pleasure += e["pleasure"] * 0.6
                    source    = name
                    break

        # Ambient sensory event
        if source == "none":
            for event_name, prob in self.ambient_probs[env].items():
                if random.random() < prob:
                    p, s = self.ambient_events[event_name]
                    pleasure += p
                    stress   += s
                    source    = event_name
                    break

        proximity       = self.get_proximity_signals()
        present         = self.get_present_entities()
        time_of_day     = (self.step_count % self.DAY_LENGTH) / self.DAY_LENGTH

        return self._get_state(), pleasure, stress, source, proximity, present, time_of_day

    # --------------------------------------------------

    def _get_state(self):
        return self.pos.astype(np.float32) / (self.size - 1)
