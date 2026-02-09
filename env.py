import numpy as np
import random


class SimpleWorld:
    def __init__(self, size=5):
        self.size = size

        # fixed entities
        self.parent_pos = np.array([0, 0])
        self.danger_pos = np.array([size - 1, size - 1])

        # random life events (pleasure, stress)
        self.events = {
            i: (random.uniform(0, 2), random.uniform(0, 2))
            for i in range(1, 11)
        }

        self.reset()

    def reset(self):
        self.pos = np.random.randint(0, self.size, size=2)
        return self._get_state()

    def step(self, action):
        move = {
            0: np.array([-1, 0]),   # up
            1: np.array([1, 0]),    # down
            2: np.array([0, -1]),   # left
            3: np.array([0, 1]),    # right
            4: np.array([0, 0])     # stay
        }

        self.pos += move[action]
        self.pos = np.clip(self.pos, 0, self.size - 1)

        pleasure = 0.0
        stress = 0.0

        # parent comfort
        if np.array_equal(self.pos, self.parent_pos):
            pleasure += 1.0

        # danger
        if np.array_equal(self.pos, self.danger_pos):
            stress += 1.0

        # random life events (unlabeled sensations)
        if random.random() < 0.2:
            p, s = random.choice(list(self.events.values()))
            pleasure += p
            stress += s

        # IMPORTANT: return ONLY 3 values
        return self._get_state(), pleasure, stress

    def _get_state(self):
        return self.pos.astype(np.float32) / (self.size - 1)
