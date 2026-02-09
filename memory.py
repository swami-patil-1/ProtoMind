class SleepMemory:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = []  # list of dicts

    def store_experience(self, state, action, pleasure, stress, timestamp):
        emotional_intensity = pleasure + stress

        self.memory.append({
            "state": state,
            "action": action,
            "pleasure": pleasure,
            "stress": stress,
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

    def get_expectations(self):
        """
        Returns bounded expected pleasure and stress
        as probabilities, not magnitudes.
        """
        if not self.memory:
            return 0.0, 0.0

        # normalize by intensity
        total_intensity = sum(m["intensity"] for m in self.memory) + 1e-6

        exp_pleasure = sum(m["pleasure"] for m in self.memory) / total_intensity
        exp_stress   = sum(m["stress"] for m in self.memory) / total_intensity

        # clamp to [0, 1]
        exp_pleasure = min(max(exp_pleasure, 0.0), 1.0)
        exp_stress   = min(max(exp_stress, 0.0), 1.0)

        return exp_pleasure, exp_stress


    def __len__(self):
        return len(self.memory)
