import numpy as np

class SimpleWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.pos = np.random.randint(0, self.size, size=2)
        return self._get_state()

    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right, 4: stay
        move = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
            4: np.array([0, 0])
        }

        self.pos += move[action]
        self.pos = np.clip(self.pos, 0, self.size - 1)

        return self._get_state()

    def _get_state(self):
        return self.pos.astype(np.float32) / (self.size - 1)
