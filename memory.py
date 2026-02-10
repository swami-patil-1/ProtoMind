class SleepMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []  # list of dicts

    def store_experience(self, state, pleasure, stress, timestamp):
        intensity = pleasure + stress

        self.memory.append({
            "state": state,
            "pleasure": pleasure,
            "stress": stress,
            "intensity": intensity,
            "timestamp": timestamp,
            "score": 0.0
        })

    def sleep_and_consolidate(self, current_time):
        for m in self.memory:
            # LRU component
            recency = 1.0 / (1.0 + (current_time - m["timestamp"]))

            # frequency + intensity
            m["score"] = recency * (0.2 + m["intensity"])

        # keep best memories only
        self.memory.sort(key=lambda x: x["score"], reverse=True)
        self.memory = self.memory[:self.capacity]

    def get_state_expectation(self, state):
        """
        Returns expected (pleasure, stress) for a given state.
        """
        if not self.memory:
            return 0.0, 0.0

        total_weight = 0.0
        ep = 0.0
        es = 0.0

        for m in self.memory:
            dist = ((m["state"] - state) ** 2).sum().item()
            weight = 1.0 / (1.0 + dist)

            ep += weight * m["pleasure"]
            es += weight * m["stress"]
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.0

        # normalize (belief, not accumulation)
        ep /= (total_weight + 1e-6)
        es /= (total_weight + 1e-6)

        return min(ep, 1.0), min(es, 1.0)

    def __len__(self):
        return len(self.memory)
