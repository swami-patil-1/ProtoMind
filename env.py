import numpy as np

class SimpleWorld:
    def __init__(self, size=5):
        self.size = size
        self.parent_pos = np.array([0, 0])
        self.danger_pos = np.array([size - 1, size - 1])
        self.reset()

    def reset(self):
        self.pos = np.random.randint(0, self.size, size=2)
        return self._get_state()

    def step(self, action):
        move = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
            4: np.array([0, 0])
        }

        self.pos += move[action]
        self.pos = np.clip(self.pos, 0, self.size - 1)

        reward = 0.0
        pain = 0.0

        if np.array_equal(self.pos, self.parent_pos):
            reward = 1.0  # comfort / feeding

        if np.array_equal(self.pos, self.danger_pos):
            pain = 1.0  # hurt

        return self._get_state(), reward, pain

    def _get_state(self):
        return self.pos.astype(np.float32) / (self.size - 1)
