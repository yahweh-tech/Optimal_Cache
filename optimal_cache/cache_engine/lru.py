from collections import OrderedDict


class LRUCache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def simulate(self, sequence):
        hits = 0
        misses = 0
        history = []

        for page in sequence:

            # Cache Hit
            if page in self.cache:
                hits += 1
                self.cache.move_to_end(page)
                action = "HIT"

            # Cache Miss
            else:
                misses += 1

                if len(self.cache) >= self.cache_size:
                    evicted, _ = self.cache.popitem(last=False)
                else:
                    evicted = None

                self.cache[page] = True
                action = "MISS"

            history.append({
                "page": page,
                "cache": list(self.cache.keys()),
                "action": action
            })

        return {
            "hits": hits,
            "misses": misses,
            "hit_ratio": hits / len(sequence),
            "history": history
        }