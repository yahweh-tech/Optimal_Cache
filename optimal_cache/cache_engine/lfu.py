class LFUCache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = {}
        self.frequency = {}
        self.time = {}
        self.clock = 0

    def simulate(self, sequence):

        hits = 0
        misses = 0
        history = []

        for page in sequence:

            self.clock += 1

            # Hit
            if page in self.cache:
                hits += 1
                self.frequency[page] += 1
                self.time[page] = self.clock

                action = "HIT"

            # Miss
            else:

                misses += 1

                if len(self.cache) >= self.cache_size:

                    victim = min(
                        self.cache.keys(),
                        key=lambda x: (self.frequency[x], self.time[x])
                    )

                    del self.cache[victim]
                    del self.frequency[victim]
                    del self.time[victim]

                self.cache[page] = True
                self.frequency[page] = 1
                self.time[page] = self.clock

                action = "MISS"

            history.append({
                "page": page,
                "cache": list(self.cache.keys()),
                "frequency": dict(self.frequency),
                "action": action
            })

        return {
            "hits": hits,
            "misses": misses,
            "hit_ratio": hits / len(sequence),
            "history": history
        }