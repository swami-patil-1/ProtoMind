import time


class SleepMemory:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = []  # list of dicts

    def store_experience(self, state, action, pleasure, stress, timestamp):
        emotional_intensity = pleasure + stress

        self.memory.append({
            "state": state,
            "action": action,
            "intensity": emotional_intensity,
            "timestamp": timestamp,
            "score": 0.0
        })

    def sleep_and_consolidate(self, current_time):
        # compute LRU + emotion score
        for m in self.memory:
            recency = 1.0 / (1.0 + (current_time - m["timestamp"]))
            m["score"] = recency * m["intensity"]

        # keep only top memories
        self.memory.sort(key=lambda x: x["score"], reverse=True)
        self.memory = self.memory[:self.capacity]

    def __len__(self):
        return len(self.memory)
