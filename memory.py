class SleepMemory:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []

    def store_experience(self, state, pleasure, stress, source, timestamp):
        intensity = pleasure + stress

        self.memory.append({
            "state": state,
            "pleasure": pleasure,
            "stress": stress,
            "source": source,
            "intensity": intensity,
            "timestamp": timestamp,
            "score": 0.0
        })

    def sleep_and_consolidate(self, current_time):
        for m in self.memory:
            recency = 1.0 / (1.0 + (current_time - m["timestamp"]))
            m["score"] = recency * (0.2 + m["intensity"])

        self.memory.sort(key=lambda x: x["score"], reverse=True)
        self.memory = self.memory[:self.capacity]

    def get_state_expectation(self, state):
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

        ep /= total_weight
        es /= total_weight

        return min(ep, 1.0), min(es, 1.0)

    def get_source_expectation(self, source):
        relevant = [m for m in self.memory if m["source"] == source]

        if not relevant:
            return 0.0, 0.0

        ep = sum(m["pleasure"] for m in relevant) / len(relevant)
        es = sum(m["stress"] for m in relevant) / len(relevant)

        return min(ep, 1.0), min(es, 1.0)

    def __len__(self):
        return len(self.memory)
